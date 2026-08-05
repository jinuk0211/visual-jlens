"""Evaluate a multimodal lens on the held-out manifest split."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from jlens import configure_logging
from jlens.cauldron import load_cauldron_samples
from jlens.multimodal import (
    MultimodalJacobianLensBundle,
    evaluate_multimodal_sample,
)
from jlens.multimodal_hf import from_hf_multimodal


def _gray(image):
    try:
        from PIL import Image

        return Image.new("RGB", image.size, color=(127, 127, 127))
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("gray-image controls require Pillow images") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    configure_logging()

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    adapter = from_hf_multimodal(model, processor)
    bundle = MultimodalJacobianLensBundle.load(args.lens)
    samples = list(load_cauldron_samples(args.manifest, split="eval", limit=args.limit))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            conditions = [("original", sample)]
            if args.controls:
                conditions.extend(
                    [
                        ("gray", replace(sample, image=_gray(sample.image))),
                        (
                            "swapped",
                            replace(
                                sample, image=samples[(index + 1) % len(samples)].image
                            ),
                        ),
                    ]
                )
            for condition, conditioned_sample in conditions:
                rows = evaluate_multimodal_sample(
                    adapter, bundle, conditioned_sample, top_k=args.top_k
                )
                for row in rows:
                    row["condition"] = condition
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
