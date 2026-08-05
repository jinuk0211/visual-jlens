# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Multimodal Jacobian-lens fitting and application.

The text-only fitter in :mod:`jlens.fitting` computes every output row of a
Jacobian.  That is useful for small models, but prohibitively expensive for a
4K-wide VLM.  This module instead uses the unbiased Rademacher estimator

``E_z [z (J^T z)^T] = J``.

Only the small VJP rows are written while fitting.  Dense lens matrices are
materialised one residual site at a time after all examples have completed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

LensMode = Literal["image_to_answer", "prompt_to_answer"]
HookKind = Literal["pre", "post"]
TargetReduction = Literal["mean", "sum"]

DEFAULT_MODES: tuple[LensMode, ...] = (
    "image_to_answer",
    "prompt_to_answer",
)


@dataclass(frozen=True)
class MultimodalSample:
    """One image/user/assistant example used to fit a multimodal lens."""

    sample_id: str
    image: Any
    user_text: str
    assistant_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PreparedMultimodalExample:
    """Model-ready inputs and explicit causal source/target masks.

    ``decoder_inputs`` contain already-fused, single-example tensors.  The
    adapter expands them for a probe microbatch without re-running its vision
    path.  Masks are one-dimensional and index the common sequence axis.
    """

    sample_id: str
    input_ids: torch.Tensor
    decoder_inputs: dict[str, Any]
    image_source_mask: torch.Tensor
    prompt_source_mask: torch.Tensor
    answer_target_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    native_inputs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, seq_len]")
        seq_len = self.input_ids.shape[1]
        for name in (
            "image_source_mask",
            "prompt_source_mask",
            "answer_target_mask",
        ):
            mask = getattr(self, name)
            if mask.dtype != torch.bool or mask.shape != (seq_len,):
                raise ValueError(f"{name} must be bool[{seq_len}], got {mask.shape}")
        if (self.image_source_mask & self.prompt_source_mask).any():
            raise ValueError("image and prompt source masks overlap")

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def n_visual_tokens(self) -> int:
        return int(self.image_source_mask.sum())

    def source_mask(self, mode: LensMode) -> torch.Tensor:
        if mode == "image_to_answer":
            return self.image_source_mask
        if mode == "prompt_to_answer":
            return self.prompt_source_mask
        raise ValueError(f"unknown lens mode {mode!r}")


@dataclass(frozen=True)
class ActivationSite:
    """A logical residual boundary and the hook that exposes it."""

    name: str
    module: nn.Module = field(repr=False, compare=False)
    hook: HookKind = "pre"
    block_index: int | None = None
    tag: str = "residual"
    is_target: bool = False


class MultimodalLensAdapter(Protocol):
    """Interface required by :func:`fit_multimodal`."""

    model_id: str
    revision: str | None
    n_layers: int
    d_model: int
    tokenizer: Any
    activation_sites: Sequence[ActivationSite]

    def prepare(
        self,
        sample: MultimodalSample,
        *,
        include_answer: bool = True,
        max_seq_len: int = 768,
    ) -> PreparedMultimodalExample: ...

    def expand_for_probes(
        self, prepared: PreparedMultimodalExample, batch_size: int
    ) -> dict[str, Any]: ...

    def forward_decoder(self, decoder_inputs: Mapping[str, Any]) -> Any: ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor: ...


class MultimodalActivationRecorder:
    """Capture named pre/post residual sites during one decoder forward.

    At ``start_graph_at`` the incoming residual is replaced with a detached
    leaf.  This deliberately cuts the graph through the frozen visual path and
    roots autograd at the fused decoder residual.
    """

    def __init__(
        self,
        sites: Sequence[ActivationSite],
        *,
        at: Sequence[str] | None = None,
        start_graph_at: str | None = None,
    ) -> None:
        by_name = {site.name: site for site in sites}
        if len(by_name) != len(sites):
            raise ValueError("activation site names must be unique")
        requested = list(by_name) if at is None else list(dict.fromkeys(at))
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ValueError(f"unknown activation sites: {missing}")
        if start_graph_at is not None:
            if start_graph_at not in by_name:
                raise ValueError(f"unknown graph root {start_graph_at!r}")
            if start_graph_at not in requested:
                requested.insert(0, start_graph_at)
        self._sites = [by_name[name] for name in requested]
        self._start_graph_at = start_graph_at
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _first_tensor(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor:
        if args and torch.is_tensor(args[0]):
            return args[0]
        hidden = kwargs.get("hidden_states")
        if torch.is_tensor(hidden):
            return hidden
        raise TypeError("residual block did not receive a tensor as its first input")

    def _pre_hook(self, site: ActivationSite):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            tensor = self._first_tensor(args, kwargs)
            if site.name == self._start_graph_at:
                tensor = tensor.detach().requires_grad_(True)
                if args and torch.is_tensor(args[0]):
                    args = (tensor, *args[1:])
                else:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = tensor
            self.activations[site.name] = tensor
            return args, kwargs

        return hook

    def _post_hook(self, site: ActivationSite):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            self.activations[site.name] = tensor

        return hook

    def __enter__(self) -> MultimodalActivationRecorder:
        try:
            for site in self._sites:
                if site.hook == "pre":
                    handle = site.module.register_forward_pre_hook(
                        self._pre_hook(site), with_kwargs=True
                    )
                elif site.hook == "post":
                    handle = site.module.register_forward_hook(self._post_hook(site))
                else:  # pragma: no cover - protected by HookKind typing
                    raise ValueError(f"unknown hook kind {site.hook!r}")
                self._handles.append(handle)
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


@dataclass(frozen=True)
class EstimatorConfig:
    """Configuration for the randomized VLM Jacobian estimator."""

    probes_per_example: int = 4
    probe_microbatch: int | Literal["auto"] = "auto"
    target_reduction: TargetReduction = "mean"
    seed: int = 0
    checkpoint_every: int = 25
    max_seq_len: int = 768
    materialize_chunk_size: int = 64
    stability_vectors: int = 64
    stability_threshold: float = 0.95
    max_probes_per_example: int = 8

    def __post_init__(self) -> None:
        if self.probes_per_example <= 0:
            raise ValueError("probes_per_example must be positive")
        if self.probe_microbatch != "auto" and self.probe_microbatch <= 0:
            raise ValueError("probe_microbatch must be positive or 'auto'")
        if self.target_reduction not in ("mean", "sum"):
            raise ValueError("target_reduction must be 'mean' or 'sum'")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.stability_vectors <= 0:
            raise ValueError("stability_vectors must be positive")
        if not -1 <= self.stability_threshold <= 1:
            raise ValueError("stability_threshold must be in [-1, 1]")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(value), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sample_manifest_hash(samples: Sequence[MultimodalSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = {
            "sample_id": sample.sample_id,
            "user_text": sample.user_text,
            "assistant_text": sample.assistant_text,
            "metadata": _jsonable(sample.metadata),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_probe_seed(seed: int, model_id: str, sample_id: str) -> int:
    raw = f"{seed}\0{model_id}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def rademacher_probes(
    d_model: int,
    count: int,
    *,
    seed: int,
    model_id: str,
    sample_id: str,
) -> torch.Tensor:
    """Deterministically generate CPU float32 Rademacher vectors."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_probe_seed(seed, model_id, sample_id))
    return (
        torch.randint(0, 2, (count, d_model), generator=generator)
        .mul_(2)
        .sub_(1)
        .float()
    )


class _SketchStore:
    """Disk-backed probe/VJP rows with exact sample-boundary resume."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        modes: Sequence[LensMode],
        site_names: Sequence[str],
        max_probes: int,
        d_model: int,
        signature: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        self.modes = tuple(modes)
        self.site_names = tuple(site_names)
        self.max_probes = max_probes
        self.d_model = d_model
        expected = {
            "format_version": self.FORMAT_VERSION,
            "signature": _jsonable(signature),
            "modes": list(self.modes),
            "site_names": list(self.site_names),
            "max_probes": max_probes,
            "d_model": d_model,
        }
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if state.get(key) != value:
                    raise ValueError(
                        f"sketch checkpoint mismatch for {key}: "
                        f"found {state.get(key)!r}, expected {value!r}"
                    )
            self.next_sample = int(state["next_sample"])
            self.n_examples = int(state["n_examples"])
            self.n_probes = int(state["n_probes"])
            self.skipped = list(state.get("skipped", []))
        else:
            self.next_sample = 0
            self.n_examples = 0
            self.n_probes = 0
            self.skipped: list[dict[str, Any]] = []
            _atomic_json(
                self.state_path,
                {
                    **expected,
                    "next_sample": 0,
                    "n_examples": 0,
                    "n_probes": 0,
                    "skipped": [],
                },
            )
        self._expected = expected
        mode = "r+" if (root / "probes.npy").exists() else "w+"
        self.probes = np.lib.format.open_memmap(
            root / "probes.npy",
            mode=mode,
            dtype=np.float16,
            shape=(max_probes, d_model),
        )
        self.gradients: dict[LensMode, dict[str, np.memmap]] = {}
        for lens_mode in self.modes:
            self.gradients[lens_mode] = {}
            for site in self.site_names:
                path = root / f"grad-{lens_mode}-{site}.npy"
                file_mode = "r+" if path.exists() else "w+"
                self.gradients[lens_mode][site] = np.lib.format.open_memmap(
                    path,
                    mode=file_mode,
                    dtype=np.float16,
                    shape=(max_probes, d_model),
                )

    def append(
        self,
        probes: torch.Tensor,
        gradients: Mapping[LensMode, Mapping[str, torch.Tensor]],
    ) -> None:
        count = probes.shape[0]
        start, end = self.n_probes, self.n_probes + count
        if end > self.max_probes:
            raise ValueError("sketch store capacity exceeded")
        self.probes[start:end] = probes.detach().cpu().numpy().astype(np.float16)
        for mode in self.modes:
            for site in self.site_names:
                rows = gradients[mode][site]
                if rows.shape != (count, self.d_model):
                    raise ValueError(
                        f"gradient {mode}/{site} has shape {rows.shape}, "
                        f"expected {(count, self.d_model)}"
                    )
                self.gradients[mode][site][start:end] = (
                    rows.detach().cpu().numpy().astype(np.float16)
                )
        self.n_probes = end

    def rollback_probes(self, n_probes: int) -> None:
        """Forget uncheckpointed rows from a partially failed example."""

        if not 0 <= n_probes <= self.n_probes:
            raise ValueError("invalid sketch rollback position")
        self.n_probes = n_probes

    def finish_sample(
        self, sample_index: int, *, success: bool, error: str | None = None
    ) -> None:
        self.next_sample = sample_index + 1
        if success:
            self.n_examples += 1
        else:
            self.skipped.append(
                {"sample_index": sample_index, "error": error or "unknown"}
            )

    def checkpoint(self) -> None:
        self.probes.flush()
        for by_site in self.gradients.values():
            for array in by_site.values():
                array.flush()
        _atomic_json(
            self.state_path,
            {
                **self._expected,
                "next_sample": self.next_sample,
                "n_examples": self.n_examples,
                "n_probes": self.n_probes,
                "skipped": self.skipped,
            },
        )

    def materialize(
        self,
        *,
        device: torch.device,
        chunk_size: int,
        extra_stores: Sequence[_SketchStore] = (),
    ) -> dict[LensMode, dict[str, torch.Tensor]]:
        stores = (self, *extra_stores)
        for store in stores[1:]:
            if (
                store.modes != self.modes
                or store.site_names != self.site_names
                or store.d_model != self.d_model
            ):
                raise ValueError("cannot combine incompatible sketch stores")
        total_probes = sum(store.n_probes for store in stores)
        if total_probes == 0:
            raise ValueError("cannot materialize an empty sketch")
        result: dict[LensMode, dict[str, torch.Tensor]] = {
            mode: {} for mode in self.modes
        }
        for mode in self.modes:
            for site in self.site_names:
                logger.info("materialize: %s/%s", mode, site)
                total = torch.zeros(
                    self.d_model, self.d_model, dtype=torch.float32, device=device
                )
                for store in stores:
                    gradients = store.gradients[mode][site]
                    for start in range(0, store.n_probes, chunk_size):
                        end = min(start + chunk_size, store.n_probes)
                        z = torch.from_numpy(
                            np.array(store.probes[start:end], copy=True)
                        ).to(device=device, dtype=torch.float32)
                        g = torch.from_numpy(
                            np.array(gradients[start:end], copy=True)
                        ).to(device=device, dtype=torch.float32)
                        total.addmm_(z.T, g)
                result[mode][site] = (total / total_probes).cpu()
                del total
        return result

    def projected_actions(
        self,
        vectors: torch.Tensor,
        *,
        device: torch.device,
        chunk_size: int,
        start: int = 0,
        end: int | None = None,
    ) -> dict[LensMode, dict[str, torch.Tensor]]:
        """Compute ``J @ vectors`` directly from sketch rows, without dense J."""

        end = self.n_probes if end is None else end
        if not 0 <= start < end <= self.n_probes:
            raise ValueError("invalid projected-action probe range")
        count = end - start
        vectors = vectors.to(device=device, dtype=torch.float32)
        result: dict[LensMode, dict[str, torch.Tensor]] = {
            mode: {} for mode in self.modes
        }
        for mode in self.modes:
            for site in self.site_names:
                output = torch.zeros(
                    self.d_model,
                    vectors.shape[1],
                    device=device,
                    dtype=torch.float32,
                )
                gradients = self.gradients[mode][site]
                for chunk_start in range(start, end, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, end)
                    z = torch.from_numpy(
                        np.array(self.probes[chunk_start:chunk_end], copy=True)
                    ).to(device=device, dtype=torch.float32)
                    g = torch.from_numpy(
                        np.array(gradients[chunk_start:chunk_end], copy=True)
                    ).to(device=device, dtype=torch.float32)
                    output.addmm_(z.T, g @ vectors)
                result[mode][site] = (output / count).cpu()
        return result


def _fixed_validation_vectors(d_model: int, count: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x4A4C454E53)
    return torch.randint(0, 2, (d_model, count), generator=generator).mul_(2).sub_(
        1
    ).float() / math.sqrt(d_model)


def _action_stability(
    first: Mapping[LensMode, Mapping[str, torch.Tensor]],
    second: Mapping[LensMode, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for mode, by_site in first.items():
        scores[mode] = {}
        for site, first_action in by_site.items():
            second_action = second[mode][site]
            cosines = torch.nn.functional.cosine_similarity(
                first_action.float(), second_action.float(), dim=0, eps=1e-8
            )
            scores[mode][site] = float(cosines.median())
    return scores


def _half_sketch_stability(
    store: _SketchStore,
    *,
    vectors: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    midpoint = store.n_probes // 2
    if midpoint == 0 or midpoint == store.n_probes:
        return {
            mode: {site: float("nan") for site in store.site_names}
            for mode in store.modes
        }
    first = store.projected_actions(
        vectors, device=device, chunk_size=chunk_size, start=0, end=midpoint
    )
    second = store.projected_actions(
        vectors,
        device=device,
        chunk_size=chunk_size,
        start=midpoint,
        end=store.n_probes,
    )
    return _action_stability(first, second)


def _cross_sketch_stability(
    first_store: _SketchStore,
    second_store: _SketchStore,
    *,
    vectors: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    first = first_store.projected_actions(vectors, device=device, chunk_size=chunk_size)
    second = second_store.projected_actions(
        vectors, device=device, chunk_size=chunk_size
    )
    return _action_stability(first, second)


def _unstable_sites(
    scores: Mapping[str, Mapping[str, float]], threshold: float
) -> list[str]:
    unstable = []
    for mode, by_site in scores.items():
        for site, score in by_site.items():
            if not math.isfinite(score) or score < threshold:
                unstable.append(f"{mode}/{site}")
    return unstable


class MultimodalJacobianLensBundle:
    """Dense mode/site Jacobians plus reproducibility metadata."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        jacobians: Mapping[LensMode, Mapping[str, torch.Tensor]],
        *,
        d_model: int,
        n_examples: int,
        n_probes: int,
        site_metadata: Mapping[str, Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> None:
        self.jacobians = {
            mode: {site: matrix.float() for site, matrix in by_site.items()}
            for mode, by_site in jacobians.items()
        }
        self.d_model = d_model
        self.n_examples = n_examples
        self.n_probes = n_probes
        self.site_metadata = {k: dict(v) for k, v in site_metadata.items()}
        self.metadata = dict(metadata)
        for mode, by_site in self.jacobians.items():
            for site, matrix in by_site.items():
                if matrix.shape != (d_model, d_model):
                    raise ValueError(f"{mode}/{site} is not [{d_model}, {d_model}]")

    @property
    def modes(self) -> list[str]:
        return sorted(self.jacobians)

    @property
    def source_sites(self) -> list[str]:
        if not self.jacobians:
            return []
        return list(next(iter(self.jacobians.values())))

    def transport(
        self, residual: torch.Tensor, *, mode: LensMode, site: str
    ) -> torch.Tensor:
        matrix = self.jacobians[mode][site].to(
            device=residual.device, dtype=residual.dtype
        )
        return residual @ matrix.T

    def save(self, path: str | os.PathLike[str], *, sites_per_shard: int = 8) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError("saving a multimodal lens requires safetensors") from exc

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        weight_index: dict[str, dict[str, str]] = {}
        for mode, by_site in self.jacobians.items():
            weight_index[mode] = {}
            sites = list(by_site)
            for shard_index, start in enumerate(range(0, len(sites), sites_per_shard)):
                shard_sites = sites[start : start + sites_per_shard]
                filename = f"lens-{mode}-{shard_index:03d}.safetensors"
                tmp = root / f"{filename}.tmp.{os.getpid()}"
                tensors = {
                    site: by_site[site].half().contiguous() for site in shard_sites
                }
                save_file(tensors, str(tmp))
                os.replace(tmp, root / filename)
                for site in shard_sites:
                    weight_index[mode][site] = filename
        _atomic_json(
            root / "manifest.json",
            {
                "format_version": self.FORMAT_VERSION,
                "d_model": self.d_model,
                "n_examples": self.n_examples,
                "n_probes": self.n_probes,
                "source_sites": self.source_sites,
                "site_metadata": self.site_metadata,
                "metadata": self.metadata,
                "weights": weight_index,
            },
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        modes: Sequence[LensMode] | None = None,
        sites: Sequence[str] | None = None,
    ) -> MultimodalJacobianLensBundle:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "loading a multimodal lens requires safetensors"
            ) from exc

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported multimodal lens format")
        wanted_modes = set(modes or manifest["weights"])
        wanted_sites = None if sites is None else set(sites)
        source_order = manifest.get("source_sites", list(manifest["site_metadata"]))
        jacobians: dict[LensMode, dict[str, torch.Tensor]] = {}
        for mode, index in manifest["weights"].items():
            if mode not in wanted_modes:
                continue
            selected = {
                site: index[site]
                for site in source_order
                if site in index and (wanted_sites is None or site in wanted_sites)
            }
            jacobians[mode] = {}
            for filename in sorted(set(selected.values())):
                shard = load_file(str(root / filename), device="cpu")
                for site, site_file in selected.items():
                    if site_file == filename:
                        jacobians[mode][site] = shard[site]
        return cls(
            jacobians=jacobians,
            d_model=manifest["d_model"],
            n_examples=manifest["n_examples"],
            n_probes=manifest["n_probes"],
            site_metadata=manifest["site_metadata"],
            metadata=manifest["metadata"],
        )

    @torch.no_grad()
    def readout(
        self,
        adapter: MultimodalLensAdapter,
        sample: MultimodalSample,
        *,
        mode: LensMode,
        sites: Sequence[str] | None = None,
        max_seq_len: int = 768,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Read a fitted lens from a prompt-only multimodal forward."""

        requested = list(sites or self.jacobians[mode])
        unknown = sorted(set(requested) - set(self.jacobians[mode]))
        if unknown:
            raise ValueError(f"sites not present in bundle: {unknown}")
        prepared = adapter.prepare(
            sample, include_answer=False, max_seq_len=max_seq_len
        )
        mask = prepared.source_mask(mode)
        if not mask.any():
            raise ValueError(f"empty source mask for {mode}")
        decoder_inputs = adapter.expand_for_probes(prepared, 1)
        with MultimodalActivationRecorder(
            adapter.activation_sites, at=requested
        ) as recorder:
            adapter.forward_decoder(decoder_inputs)
        logits: dict[str, torch.Tensor] = {}
        for site in requested:
            residual = recorder.activations[site][
                0, mask.to(recorder.activations[site].device)
            ]
            residual = residual.float().mean(dim=0, keepdim=True)
            transported = self.transport(residual, mode=mode, site=site)
            logits[site] = adapter.unembed(transported).float().cpu()
        return logits, prepared.input_ids.cpu()


def _source_and_target_sites(
    adapter: MultimodalLensAdapter,
) -> tuple[list[ActivationSite], ActivationSite]:
    targets = [site for site in adapter.activation_sites if site.is_target]
    if len(targets) != 1:
        raise ValueError("adapter must expose exactly one target activation site")
    sources = [site for site in adapter.activation_sites if not site.is_target]
    if not sources:
        raise ValueError("adapter must expose at least one source site")
    if sources[0].name != "embed":
        raise ValueError("the first source site must be the fused 'embed' residual")
    return sources, targets[0]


def _probe_vjp(
    adapter: MultimodalLensAdapter,
    prepared: PreparedMultimodalExample,
    probes: torch.Tensor,
    *,
    modes: Sequence[LensMode],
    target_reduction: TargetReduction,
) -> dict[LensMode, dict[str, torch.Tensor]]:
    sources, target = _source_and_target_sites(adapter)
    batch_size = probes.shape[0]
    decoder_inputs = adapter.expand_for_probes(prepared, batch_size)
    names = [site.name for site in sources] + [target.name]
    with (
        MultimodalActivationRecorder(
            adapter.activation_sites,
            at=names,
            start_graph_at=sources[0].name,
        ) as recorder,
        torch.enable_grad(),
    ):
        adapter.forward_decoder(decoder_inputs)
        target_activation = recorder.activations[target.name]
        source_activations = [recorder.activations[site.name] for site in sources]
        target_positions = prepared.answer_target_mask.nonzero(as_tuple=True)[0]
        if len(target_positions) == 0:
            raise ValueError("answer_target_mask is empty")
        target_positions = target_positions.to(target_activation.device)
        z = probes.to(device=target_activation.device, dtype=target_activation.dtype)
        cotangent = torch.zeros_like(target_activation)
        scale = 1.0 / len(target_positions) if target_reduction == "mean" else 1.0
        cotangent[:, target_positions, :] = z[:, None, :] * scale
        grads = torch.autograd.grad(
            outputs=target_activation,
            inputs=source_activations,
            grad_outputs=cotangent,
            retain_graph=False,
        )

    result: dict[LensMode, dict[str, torch.Tensor]] = {mode: {} for mode in modes}
    for site, grad in zip(sources, grads, strict=True):
        for mode in modes:
            positions = prepared.source_mask(mode).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                raise ValueError(f"source mask for {mode} is empty")
            rows = grad[:, positions.to(grad.device), :].float().mean(dim=1)
            result[mode][site.name] = rows.cpu()
    return result


def auto_tune_probe_microbatch(
    adapter: MultimodalLensAdapter,
    samples: Sequence[MultimodalSample],
    config: EstimatorConfig,
    *,
    max_peak_bytes: int = 72 * 1024**3,
) -> int:
    """Choose 4, 2, or 1 probes using up to two dry-run examples."""

    if not torch.cuda.is_available():
        return 1
    candidates = [n for n in (4, 2, 1) if n <= config.probes_per_example]
    if not candidates:
        candidates = [1]
    examples = list(samples[:2])
    for candidate in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for sample in examples:
                prepared = adapter.prepare(
                    sample, include_answer=True, max_seq_len=config.max_seq_len
                )
                probes = rademacher_probes(
                    adapter.d_model,
                    candidate,
                    seed=config.seed,
                    model_id=adapter.model_id,
                    sample_id=sample.sample_id,
                )
                _probe_vjp(
                    adapter,
                    prepared,
                    probes,
                    modes=DEFAULT_MODES,
                    target_reduction=config.target_reduction,
                )
            peak = torch.cuda.max_memory_allocated()
            logger.info(
                "probe microbatch dry run: batch=%d peak=%.1f GiB",
                candidate,
                peak / 1024**3,
            )
            if peak <= max_peak_bytes:
                return candidate
        except torch.OutOfMemoryError:
            logger.warning("probe microbatch %d OOM; trying a smaller batch", candidate)
        finally:
            torch.cuda.empty_cache()
    return 1


def fit_multimodal(
    adapter: MultimodalLensAdapter,
    samples: Sequence[MultimodalSample],
    output_dir: str | os.PathLike[str],
    *,
    modes: Sequence[LensMode] = DEFAULT_MODES,
    config: EstimatorConfig | None = None,
    materialize_device: str | torch.device | None = None,
) -> MultimodalJacobianLensBundle:
    """Fit image/prompt-to-answer Jacobian lenses with resumable sketches."""

    if not samples:
        raise ValueError("fit_multimodal needs at least one sample")
    config = config or EstimatorConfig()
    modes = tuple(dict.fromkeys(modes))
    unknown_modes = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown_modes:
        raise ValueError(f"unknown lens modes: {unknown_modes}")
    sources, target = _source_and_target_sites(adapter)
    site_names = [site.name for site in sources]
    manifest_hash = _sample_manifest_hash(samples)
    signature = {
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "manifest_hash": manifest_hash,
        "probes_per_example": config.probes_per_example,
        "target_reduction": config.target_reduction,
        "seed": config.seed,
        "max_seq_len": config.max_seq_len,
    }
    output_root = Path(output_dir)
    store = _SketchStore(
        output_root / "sketch",
        modes=modes,
        site_names=site_names,
        max_probes=len(samples) * config.probes_per_example,
        d_model=adapter.d_model,
        signature=signature,
    )
    microbatch = (
        auto_tune_probe_microbatch(adapter, samples, config)
        if config.probe_microbatch == "auto"
        else min(config.probe_microbatch, config.probes_per_example)
    )

    def collect(
        destination: _SketchStore,
        *,
        probe_count: int,
        probe_seed: int,
        pass_name: str,
    ) -> None:
        pass_microbatch = min(microbatch, probe_count)
        durations: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(
            "fit_multimodal[%s]: model=%s examples=%d sites=%d "
            "probes/example=%d microbatch=%d",
            pass_name,
            adapter.model_id,
            len(samples),
            len(sources),
            probe_count,
            pass_microbatch,
        )
        for sample_index, sample in enumerate(samples):
            if sample_index < destination.next_sample:
                continue
            sample_probe_start = destination.n_probes
            started_at = time.perf_counter()
            try:
                prepared = adapter.prepare(
                    sample, include_answer=True, max_seq_len=config.max_seq_len
                )
                probes = rademacher_probes(
                    adapter.d_model,
                    probe_count,
                    seed=probe_seed,
                    model_id=adapter.model_id,
                    sample_id=sample.sample_id,
                )
                for start in range(0, probe_count, pass_microbatch):
                    probe_batch = probes[start : start + pass_microbatch]
                    gradients = _probe_vjp(
                        adapter,
                        prepared,
                        probe_batch,
                        modes=modes,
                        target_reduction=config.target_reduction,
                    )
                    destination.append(probe_batch, gradients)
                destination.finish_sample(sample_index, success=True)
                duration = time.perf_counter() - started_at
                durations.append(duration)
                logger.info(
                    "  [%s] example %d/%d id=%s seq=%d visual=%d %.1fs",
                    pass_name,
                    sample_index + 1,
                    len(samples),
                    sample.sample_id,
                    prepared.seq_len,
                    prepared.n_visual_tokens,
                    duration,
                )
            except ValueError as exc:
                destination.rollback_probes(sample_probe_start)
                logger.warning(
                    "  [%s] skipping example %s: %s",
                    pass_name,
                    sample.sample_id,
                    exc,
                )
                destination.finish_sample(sample_index, success=False, error=str(exc))
            if destination.next_sample % config.checkpoint_every == 0:
                destination.checkpoint()
        destination.checkpoint()
        if durations:
            median_seconds = sorted(durations)[len(durations) // 2]
            peak_gib = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            logger.info(
                "fit_multimodal[%s] summary: median=%.1fs/example "
                "projected=%.2fh peak_allocated=%.1fGiB",
                pass_name,
                median_seconds,
                median_seconds * len(samples) / 3600,
                peak_gib,
            )

    collect(
        store,
        probe_count=config.probes_per_example,
        probe_seed=config.seed,
        pass_name="initial",
    )
    if store.n_examples == 0:
        raise ValueError("no multimodal examples were successfully fitted")

    if materialize_device is None:
        materialize_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        materialize_device = torch.device(materialize_device)
    validation_vectors = _fixed_validation_vectors(
        adapter.d_model, config.stability_vectors, seed=config.seed
    )
    stability = _half_sketch_stability(
        store,
        vectors=validation_vectors,
        device=materialize_device,
        chunk_size=config.materialize_chunk_size,
    )
    initial_unstable = _unstable_sites(stability, config.stability_threshold)
    extra_store: _SketchStore | None = None
    max_probes = max(config.probes_per_example, config.max_probes_per_example)
    extra_count = max_probes - config.probes_per_example
    if initial_unstable and extra_count > 0:
        logger.warning(
            "sketch stability below %.3f at %d mode/sites; collecting %d "
            "additional probes/example",
            config.stability_threshold,
            len(initial_unstable),
            extra_count,
        )
        extra_seed = config.seed ^ 0x5A17A5E
        extra_signature = {
            **signature,
            "adaptive_pass": 2,
            "probes_per_example": extra_count,
            "seed": extra_seed,
        }
        extra_store = _SketchStore(
            output_root / "sketch-extra",
            modes=modes,
            site_names=site_names,
            max_probes=len(samples) * extra_count,
            d_model=adapter.d_model,
            signature=extra_signature,
        )
        collect(
            extra_store,
            probe_count=extra_count,
            probe_seed=extra_seed,
            pass_name="adaptive",
        )
        if extra_store.n_probes:
            stability = _cross_sketch_stability(
                store,
                extra_store,
                vectors=validation_vectors,
                device=materialize_device,
                chunk_size=config.materialize_chunk_size,
            )
    final_unstable = _unstable_sites(stability, config.stability_threshold)
    if final_unstable:
        logger.warning(
            "%d mode/sites remain unstable after fitting", len(final_unstable)
        )
    extra_stores = (
        (extra_store,) if extra_store is not None and extra_store.n_probes else ()
    )
    jacobians = store.materialize(
        device=materialize_device,
        chunk_size=config.materialize_chunk_size,
        extra_stores=extra_stores,
    )
    site_metadata = {
        site.name: {
            "block_index": site.block_index,
            "tag": site.tag,
            "hook": site.hook,
            "stability": {mode: stability[mode][site.name] for mode in modes},
            "stable": all(
                math.isfinite(stability[mode][site.name])
                and stability[mode][site.name] >= config.stability_threshold
                for mode in modes
            ),
        }
        for site in sources
    }
    total_probes = store.n_probes + sum(item.n_probes for item in extra_stores)
    metadata = {
        **signature,
        "target_site": target.name,
        "modes": list(modes),
        "skipped": {
            "initial": store.skipped,
            "adaptive": extra_store.skipped if extra_store is not None else [],
        },
        "estimator": "rademacher_vjp",
        "stability_threshold": config.stability_threshold,
        "unstable_sites": final_unstable,
        "adaptive_probe_count": extra_count if extra_store is not None else 0,
    }
    bundle = MultimodalJacobianLensBundle(
        jacobians,
        d_model=adapter.d_model,
        n_examples=store.n_examples,
        n_probes=total_probes,
        site_metadata=site_metadata,
        metadata=metadata,
    )
    bundle.save(output_root / "lens")
    return bundle


def _distribution_metrics(
    lens_logits: torch.Tensor, native_logits: torch.Tensor, *, top_k: int
) -> dict[str, float]:
    lens = lens_logits.float().flatten()
    native = native_logits.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(lens, native, dim=0).item()
    native_logp = torch.log_softmax(native, dim=0)
    lens_logp = torch.log_softmax(lens, dim=0)
    kl = torch.sum(native_logp.exp() * (native_logp - lens_logp)).item()
    k = min(top_k, lens.numel())
    lens_top = set(torch.topk(lens, k).indices.tolist())
    native_top = set(torch.topk(native, k).indices.tolist())
    return {
        "logit_cosine": cosine,
        "kl_native_to_lens": kl,
        "top_k_overlap": len(lens_top & native_top) / k,
    }


@torch.no_grad()
def evaluate_multimodal_sample(
    adapter: MultimodalLensAdapter,
    bundle: MultimodalJacobianLensBundle,
    sample: MultimodalSample,
    *,
    modes: Sequence[LensMode] = DEFAULT_MODES,
    sites: Sequence[str] | None = None,
    max_seq_len: int = 768,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Compare lens logits with the native answer-position distribution."""

    requested = list(sites or bundle.source_sites)
    prepared = adapter.prepare(sample, include_answer=True, max_seq_len=max_seq_len)
    if not prepared.answer_target_mask.any():
        raise ValueError("evaluation sample has no answer target positions")
    _, target = _source_and_target_sites(adapter)
    decoder_inputs = adapter.expand_for_probes(prepared, 1)
    with MultimodalActivationRecorder(
        adapter.activation_sites, at=[*requested, target.name]
    ) as recorder:
        adapter.forward_decoder(decoder_inputs)
    target_residual = recorder.activations[target.name][
        0, prepared.answer_target_mask.to(recorder.activations[target.name].device)
    ]
    native_logits = adapter.unembed(target_residual.float()).float().mean(dim=0).cpu()
    rows: list[dict[str, Any]] = []
    for mode in modes:
        mask = prepared.source_mask(mode)
        for site in requested:
            residual = recorder.activations[site][
                0, mask.to(recorder.activations[site].device)
            ]
            residual = residual.float().mean(dim=0, keepdim=True)
            lens_logits = (
                adapter.unembed(bundle.transport(residual, mode=mode, site=site))
                .float()[0]
                .cpu()
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "mode": mode,
                    "site": site,
                    **_distribution_metrics(lens_logits, native_logits, top_k=top_k),
                }
            )
    return rows
