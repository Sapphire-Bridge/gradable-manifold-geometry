from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"
VARIANTS = ("natural", "neutral", "iso_ratio", "artificial", "fictional_semantic", "counter_natural")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else float("nan")
    m = _mean(vals)
    return float(math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)))


def _fmt(x: float, digits: int = 3) -> str:
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def _load_variant(variant: str) -> Dict[str, Any]:
    version = f"v2_{variant}"
    path = DEFAULT_ROOT / f"gradable_size_{version}_behavior_recompute.json"
    if not path.exists():
        return {
            "variant": variant,
            "version": version,
            "missing": True,
            "path": str(path),
        }
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    row = summary["domains"]["size"]
    unique = row["side_metrics_unique"]
    corr = unique["corr_ordered_score_vs_predictor"]
    ci = corr.get("bootstrap_prompt95", {})
    direction = row["shift_direction_match"]
    return {
        "variant": variant,
        "version": version,
        "n_pairs": row["counts"]["n_pair_rows_recomputed"],
        "n_unique_sides": row["counts"]["n_unique_sides"],
        "r": corr["r"],
        "ci_lo": ci.get("lo", float("nan")),
        "ci_hi": ci.get("hi", float("nan")),
        "p": corr.get("permutation_p_two_sided", float("nan")),
        "within_target_p": corr.get("within_target_permutation_p_two_sided", float("nan")),
        "direction_count": direction["count"],
        "direction_n": direction["n"],
        "direction_rate": direction["rate"],
        "argmax_counts": unique["argmax_label_counts"],
        "warnings": row.get("warnings", []),
        "missing": False,
    }


def _iso_ratio_diagnostics() -> Dict[str, Any]:
    path = DEFAULT_ROOT / "gradable_size_v2_iso_ratio_behavior_recompute.sides.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        predictor = float(row["predictor"])
        ratio = math.exp(predictor)
        grouped[f"{ratio:.6g}"].append(row)
    bins: List[Dict[str, Any]] = []
    for ratio, group in sorted(grouped.items(), key=lambda kv: float(kv[0])):
        scores = [float(r["ordered_score"]) for r in group]
        bins.append(
            {
                "ratio": ratio,
                "n": len(group),
                "mean_ordered_score": _mean(scores),
                "sd_ordered_score": _std(scores),
                "values": sorted({int(float(r["value"])) for r in group}),
                "standards": sorted({int(float(r["standard"])) for r in group}),
                "expected_labels": dict(sorted({lab: sum(1 for r in group if r["expected_label"] == lab) for lab in {r["expected_label"] for r in group}}.items())),
            }
        )
    return {"bins": bins}


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Size v2 Behavioral Lock-In Summary",
        "",
        "Primary metric: unique-side `corr(ordered_score, log(value / standard))`.",
        "",
        "| variant | n pairs | unique sides | r | 95% CI | perm. p | within-target p | direction match | argmax counts |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["variants"]:
        if row.get("missing"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["variant"]),
                        "missing",
                        "missing",
                        "nan",
                        "missing",
                        "nan",
                        "nan",
                        "missing",
                        f"missing recompute: {row['path']}",
                    ]
                )
                + " |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["n_pairs"]),
                    str(row["n_unique_sides"]),
                    _fmt(float(row["r"])),
                    f"[{_fmt(float(row['ci_lo']))}, {_fmt(float(row['ci_hi']))}]",
                    _fmt(float(row["p"]), 5),
                    _fmt(float(row["within_target_p"]), 5),
                    f"{row['direction_count']}/{row['direction_n']}={_fmt(float(row['direction_rate']))}",
                    ", ".join(f"{k}:{v}" for k, v in sorted(row["argmax_counts"].items())),
                ]
            )
            + " |"
        )
    iso = summary.get("iso_ratio", {})
    if iso:
        lines.extend(
            [
                "",
                "## Iso-Ratio Diagnostics",
                "",
                "Rows with the same ratio should have similar ordered scores across different absolute values/standards. This is diagnostic, not a pass/fail gate by itself.",
                "",
                "| ratio | n | mean ordered score | sd ordered score | values | standards |",
                "| ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in iso.get("bins", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["ratio"]),
                        str(row["n"]),
                        _fmt(float(row["mean_ordered_score"])),
                        _fmt(float(row["sd_ordered_score"])),
                        ",".join(str(x) for x in row["values"]),
                        ",".join(str(x) for x in row["standards"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Gate interpretation:",
            "",
            "```text",
            "Go to confirmatory raw patching only if natural, iso_ratio, and at least one semantic-control variant pass.",
            "Semantic-control variants: fictional_semantic and counter_natural.",
            "Use r >= 0.40 with CI excluding 0 as the strict behavior gate.",
            "Treat artificial/inverted standards as a stress test; partial survival is already informative.",
            "If bare neutral fails but fictional_semantic passes, interpret the effect as semantically scaffolded calibration, not context-free ratio arithmetic.",
            "If iso_ratio fails, do not proceed to geometry yet.",
            "```",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize size v2 behavioral-lock-in recompute outputs.")
    p.add_argument("--out_md", type=str, default=str(DEFAULT_ROOT / "gradable_size_v2_behavior_lockin.summary.md"))
    p.add_argument("--out_json", type=str, default=str(DEFAULT_ROOT / "gradable_size_v2_behavior_lockin.summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = {
        "schema_version": "gradable_size_v2_lockin_summary_v1",
        "variants": [_load_variant(variant) for variant in VARIANTS],
        "iso_ratio": _iso_ratio_diagnostics(),
    }
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Wrote summary JSON: {out_json}")
    print(f"Wrote summary Markdown: {out_md}")


if __name__ == "__main__":
    main()
