# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic The Cauldron manifest preparation for VLM fitting."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from jlens.multimodal import MultimodalSample

CAULDRON_REPO = "HuggingFaceM4/the_cauldron"
CAULDRON_QUOTAS: dict[str, dict[str, int]] = {
    "vqav2": {"fit": 250, "eval": 50},
    "aokvqa": {"fit": 100, "eval": 20},
    "vsr": {"fit": 150, "eval": 30},
    "docvqa": {"fit": 150, "eval": 30},
    "chartqa": {"fit": 150, "eval": 30},
    "textcaps": {"fit": 100, "eval": 20},
    "screen2words": {"fit": 100, "eval": 20},
}


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The Cauldron helpers require the 'datasets' extra: "
            "pip install 'jlens[dev]'"
        ) from exc
    return load_dataset


def _stable_int(*parts: Any) -> int:
    raw = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _image_hash(image: Any) -> str:
    digest = hashlib.sha256()
    if hasattr(image, "mode") and hasattr(image, "size") and hasattr(image, "tobytes"):
        digest.update(str(image.mode).encode())
        digest.update(str(tuple(image.size)).encode())
        digest.update(image.tobytes())
        return digest.hexdigest()
    if isinstance(image, dict) and image.get("bytes") is not None:
        digest.update(image["bytes"])
        return digest.hexdigest()
    if isinstance(image, (bytes, bytearray)):
        digest.update(image)
        return digest.hexdigest()
    digest.update(repr(image).encode())
    return digest.hexdigest()


def _conversation(row: dict[str, Any], *, seed: int, config: str, row_index: int):
    texts = row.get("texts")
    if isinstance(texts, dict):
        texts = [texts]
    if not isinstance(texts, list) or not texts:
        raise ValueError("row has no conversations")
    index = _stable_int(seed, config, row_index, "conversation") % len(texts)
    conversation = texts[index]
    user = str(conversation.get("user", "")).strip()
    assistant = str(conversation.get("assistant", "")).strip()
    if not user or not assistant:
        raise ValueError("conversation has an empty user or assistant turn")
    return index, user, assistant, conversation.get("source")


def _resolved_revision(repo_id: str, revision: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision)
    return str(info.sha)


def build_cauldron_manifest(
    output_path: str | os.PathLike[str],
    *,
    seed: int = 0,
    revision: str | None = None,
    quotas: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Create the fixed 1,000-fit/200-eval JSONL manifest.

    A hash assigns each row to fit or eval before quota sampling.  The function
    streams each dataset configuration and stores only provenance plus an image
    digest; image bytes are never written to the manifest.
    """

    load_dataset = _require_datasets()
    quotas = quotas or CAULDRON_QUOTAS
    resolved_revision = _resolved_revision(CAULDRON_REPO, revision)
    records: list[dict[str, Any]] = []
    for config, split_quotas in quotas.items():
        counts = {"fit": 0, "eval": 0}
        dataset = load_dataset(
            CAULDRON_REPO,
            config,
            split="train",
            revision=resolved_revision,
            streaming=True,
        )
        for row_index, row in enumerate(dataset):
            # One fifth is held out before either quota is sampled.
            split = (
                "eval"
                if _stable_int(seed, config, row_index, "split") % 5 == 0
                else "fit"
            )
            if counts[split] >= split_quotas[split]:
                if all(counts[name] >= split_quotas[name] for name in counts):
                    break
                continue
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1:
                continue
            try:
                text_index, user, assistant, source = _conversation(
                    row, seed=seed, config=config, row_index=row_index
                )
            except ValueError:
                continue
            image = images[0]
            sample_id = f"cauldron/{config}/{row_index}/{text_index}"
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": CAULDRON_REPO,
                    "revision": resolved_revision,
                    "config": config,
                    "source": source,
                    "row_index": row_index,
                    "image_index": 0,
                    "image_sha256": _image_hash(image),
                    "text_index": text_index,
                    "user_text": user,
                    "assistant_text": assistant,
                    "split": split,
                }
            )
            counts[split] += 1
        missing = {
            split: split_quotas[split] - counts[split]
            for split in counts
            if counts[split] < split_quotas[split]
        }
        if missing:
            raise RuntimeError(
                f"{config} stream ended before quotas were met: {missing}"
            )

    # Keep each split deterministically shuffled so prefixes (notably the
    # 32-example pilot) are not a single dataset configuration.
    records.sort(
        key=lambda row: (
            row["split"],
            _stable_int(seed, row["sample_id"], "manifest-order"),
        )
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    return records


def read_manifest(
    path: str | os.PathLike[str], *, split: str | None = None
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split is not None:
        records = [record for record in records if record["split"] == split]
    return records


def iter_cauldron_samples(
    manifest_path: str | os.PathLike[str],
    *,
    split: str = "fit",
    limit: int | None = None,
) -> Iterator[MultimodalSample]:
    """Reload manifest images from the pinned dataset revision and verify them."""

    load_dataset = _require_datasets()
    records = read_manifest(manifest_path, split=split)
    if limit is not None:
        records = records[:limit]
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_config[record["config"]].append(record)
    for config, config_records in by_config.items():
        revision_values = {record["revision"] for record in config_records}
        if len(revision_values) != 1:
            raise ValueError(f"manifest mixes revisions for {config}")
        revision = next(iter(revision_values))
        wanted = {int(record["row_index"]): record for record in config_records}
        dataset = load_dataset(
            CAULDRON_REPO,
            config,
            split="train",
            revision=revision,
            streaming=True,
        )
        for row_index, row in enumerate(dataset):
            record = wanted.get(row_index)
            if record is None:
                if wanted and row_index <= max(wanted):
                    continue
                if not wanted or row_index > max(wanted):
                    break
            image = row["images"][int(record["image_index"])]
            actual_hash = _image_hash(image)
            if actual_hash != record["image_sha256"]:
                raise ValueError(
                    f"image hash mismatch for {record['sample_id']}: "
                    f"{actual_hash} != {record['image_sha256']}"
                )
            if hasattr(image, "copy"):
                image = image.copy()
            yield MultimodalSample(
                sample_id=record["sample_id"],
                image=image,
                user_text=record["user_text"],
                assistant_text=record["assistant_text"],
                metadata={
                    "dataset": record["dataset"],
                    "revision": revision,
                    "config": config,
                    "source": record.get("source"),
                    "row_index": row_index,
                    "image_sha256": actual_hash,
                    "split": split,
                },
            )
            del wanted[row_index]
            if not wanted:
                break
        if wanted:
            raise RuntimeError(
                f"dataset stream ended before manifest rows were found: {sorted(wanted)[:5]}"
            )


def load_cauldron_samples(
    manifest_path: str | os.PathLike[str],
    *,
    split: str = "fit",
    limit: int | None = None,
) -> Sequence[MultimodalSample]:
    return list(iter_cauldron_samples(manifest_path, split=split, limit=limit))
