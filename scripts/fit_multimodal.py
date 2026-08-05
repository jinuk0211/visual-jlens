"""Fit a Qwen3-VL or Gemma 4 Unified multimodal Jacobian lens."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from jlens import configure_logging
from jlens.cauldron import load_cauldron_samples
from jlens.multimodal import EstimatorConfig, fit_multimodal
from jlens.multimodal_hf import from_hf_multimodal, validate_adapter_parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--max-probes", type=int, default=8)
    parser.add_argument("--probe-microbatch", default="auto")
    parser.add_argument("--target-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--stability-threshold", type=float, default=0.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-free-disk-gib", type=float, default=100.0)
    parser.add_argument("--skip-parity-check", action="store_true")
    args = parser.parse_args()
    configure_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / 1024**3
    if free_gib < args.min_free_disk_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free at {output_dir}; "
            f"need {args.min_free_disk_gib:.1f} GiB "
            "(--min-free-disk-gib 0 disables this guard)"
        )

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "This model requires a recent Transformers build with "
            "AutoModelForMultimodalLM"
        ) from exc

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    load_kwargs = {
        "revision": args.revision,
        "dtype": torch.bfloat16,
        "device_map": {"": args.device},
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
    adapter = from_hf_multimodal(model, processor)
    samples = load_cauldron_samples(args.manifest, split="fit", limit=args.limit)
    if not args.skip_parity_check:
        reports = validate_adapter_parity(
            adapter, list(samples[:10]), max_seq_len=args.max_seq_len
        )
        worst = max(float(report["max_abs_error"]) for report in reports)
        print(f"adapter parity passed on {len(reports)} examples (max_abs={worst:.4g})")
    microbatch: int | str = args.probe_microbatch
    if microbatch != "auto":
        microbatch = int(microbatch)
    bundle = fit_multimodal(
        adapter,
        samples,
        args.output_dir,
        config=EstimatorConfig(
            probes_per_example=args.probes,
            probe_microbatch=microbatch,
            target_reduction=args.target_reduction,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            max_seq_len=args.max_seq_len,
            stability_threshold=args.stability_threshold,
            max_probes_per_example=args.max_probes,
        ),
        materialize_device=args.device,
    )
    print(
        f"saved {args.output_dir}/lens: {bundle.n_examples} examples, "
        f"{bundle.n_probes} probes"
    )


if __name__ == "__main__":
    main()
