from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHOICES = {
    "tiny": [" tiny"],
    "small": [" small"],
    "large": [" large"],
    "huge": [" huge"],
}


# (context, typical adult body length in centimeters)
CONTEXTS = [
    ("an ant", 1),
    ("a house mouse", 9),
    ("a domestic cat", 46),
    ("a labrador dog", 60),
    ("an adult human", 170),
    ("an african elephant", 350),
]


# scalar values in centimeters
VALUES = [1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500]


def label_for_ratio(ratio: float) -> str | None:
    # ratio = value / standard.
    # ratio <= 0.5 -> "tiny"; 0.5 < r <= 0.85 -> "small";
    # 1.15 <= r < 2.0 -> "large"; r >= 2.0 -> "huge"; gap 0.85<r<1.15 -> borderline.
    if ratio <= 0.5:
        return "tiny"
    if 0.5 < ratio <= 0.85:
        return "small"
    if 1.15 <= ratio < 2.0:
        return "large"
    if ratio >= 2.0:
        return "huge"
    return None


def prompt_for(context: str, standard: int, value: int) -> str:
    return (
        f"The normal body length for {context} is {standard} centimeters. "
        f"Compared only to that {standard}-centimeter baseline for {context}, "
        f"a body length of {value} centimeters feels"
    )


def slug(raw: str) -> str:
    return str(raw).replace(" ", "_").replace("-", "_")


def build_items() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in VALUES:
        sides: List[Dict[str, Any]] = []
        for context, standard in CONTEXTS:
            ratio = float(value) / float(standard)
            label = label_for_ratio(ratio)
            if label is None:
                continue
            target = f"{value} centimeters"
            prompt = prompt_for(context, int(standard), int(value))
            if prompt.count(target) != 1:
                continue
            sides.append(
                {
                    "context": str(context),
                    "standard": int(standard),
                    "value": int(value),
                    "target": target,
                    "prompt": prompt,
                    "expected_label": str(label),
                    "ratio_from_standard": float(ratio),
                }
            )
        for i, a in enumerate(sides):
            for b in sides[i + 1 :]:
                if a["expected_label"] == b["expected_label"]:
                    continue
                value_tag = f"p{int(value):04d}"
                pair_id = f"size-v1_1-{value_tag}-{slug(a['context'])}-vs-{slug(b['context'])}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "target": str(a["target"]),
                        "target_occurrence": 0,
                        "a": {"prompt": str(a["prompt"]), "expected_label": str(a["expected_label"])},
                        "b": {"prompt": str(b["prompt"]), "expected_label": str(b["expected_label"])},
                        "choices": CHOICES,
                        "metadata": {
                            "type": "gradable_scalar",
                            "dimension": "size",
                            "unit": "centimeters",
                            "value": int(value),
                            "predicate_family": "tiny_huge",
                            "comparison_a": str(a["context"]),
                            "comparison_b": str(b["context"]),
                            "standard_a": int(a["standard"]),
                            "standard_b": int(b["standard"]),
                            "ratio_a": float(a["ratio_from_standard"]),
                            "ratio_b": float(b["ratio_from_standard"]),
                            "standard_type": "comparison_class_explicit",
                            "scale_type": "relative_open_size",
                            "control_type": "standard_explicit",
                            "is_borderline": False,
                            "is_artificial_norm": False,
                            "design": "same_scalar_different_explicit_standard",
                            "source": "bierwisch_kennedy_size_v1_1",
                        },
                    }
                )
    return rows


def main() -> None:
    root = ROOT / "results" / "manifold_groups_poc"
    out_jsonl = root / "gradable_size_disamb_pairs_v1_1.jsonl"
    out_csv = root / "gradable_size_grid_v1_1.csv"
    rows = build_items()
    root.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        md = dict(row["metadata"])
        flat_rows.append(
            {
                "pair_id": row["pair_id"],
                "target": row["target"],
                "a_prompt": row["a"]["prompt"],
                "b_prompt": row["b"]["prompt"],
                "a_expected_label": row["a"]["expected_label"],
                "b_expected_label": row["b"]["expected_label"],
                **md,
            }
        )
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in flat_rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {len(rows)} pairs: {out_jsonl}")
    print(f"Wrote grid CSV: {out_csv}")


if __name__ == "__main__":
    main()
