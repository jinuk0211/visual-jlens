"""Aggregate held-out metrics and render an architecture-annotated SVG plot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from html import escape
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Evaluation JSONL")
    parser.add_argument("--lens", required=True, help="Lens artifact directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="logit_cosine")
    args = parser.parse_args()

    lens_manifest = json.loads(
        (Path(args.lens) / "manifest.json").read_text(encoding="utf-8")
    )
    sites = lens_manifest["source_sites"]
    site_index = {site: index for index, site in enumerate(sites)}
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for line in Path(args.eval).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        groups[(row.get("condition", "original"), row["mode"], row["site"])].append(
            float(row[args.metric])
        )
    summary = [
        {
            "condition": condition,
            "mode": mode,
            "site": site,
            "site_index": site_index[site],
            "tag": lens_manifest["site_metadata"][site]["tag"],
            args.metric: sum(values) / len(values),
            "n": len(values),
        }
        for (condition, mode, site), values in groups.items()
    ]
    summary.sort(key=lambda row: (row["condition"], row["mode"], row["site_index"]))
    if not summary:
        raise ValueError("evaluation file contains no metric rows")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    width, height = 1100, 560
    left, right, top, bottom = 70, 30, 45, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [row[args.metric] for row in summary]
    y_min = -1.0 if args.metric == "logit_cosine" else min(values)
    y_max = 1.0 if args.metric == "logit_cosine" else max(values)
    if y_max == y_min:
        y_max = y_min + 1

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + plot_width * index / max(1, len(sites) - 1)
        y = top + plot_height * (y_max - value) / (y_max - y_min)
        return x, y

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{escape(args.metric)} by residual site</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for index, site in enumerate(sites):
        tag = lens_manifest["site_metadata"][site]["tag"]
        if tag in {"deepstack", "global"}:
            x, _ = xy(index, y_min)
            lines.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#999" stroke-dasharray="4 4"/>'
            )
            lines.append(
                f'<text x="{x + 3:.1f}" y="{top + 12}" font-family="sans-serif" font-size="10">{escape(tag)}</text>'
            )
    series: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary:
        series[(row["condition"], row["mode"])].append(row)
    for series_index, ((condition, mode), rows) in enumerate(sorted(series.items())):
        color = colors[series_index % len(colors)]
        points = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (xy(row["site_index"], row[args.metric]) for row in rows)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_y = top + 18 * series_index
        lines.append(
            f'<text x="{left + 10}" y="{legend_y}" font-family="sans-serif" font-size="11" fill="{color}">{escape(condition)} / {escape(mode)}</text>'
        )
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        _, y = xy(0, value)
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>'
        )
    lines.append(
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="12">fused decoder residual site</text>'
    )
    lines.append("</svg>")
    (output_dir / "layer_plot.svg").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} and {output_dir / 'layer_plot.svg'}")


if __name__ == "__main__":
    main()
