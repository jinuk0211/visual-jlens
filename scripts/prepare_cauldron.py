"""Build the pinned The Cauldron fit/eval manifest."""

from __future__ import annotations

import argparse

from jlens.cauldron import build_cauldron_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Destination JSONL manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    records = build_cauldron_manifest(
        args.output, seed=args.seed, revision=args.revision
    )
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("fit", "eval")
    }
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
