from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHOICES = {
    "cold": [" cold"],
    "cool": [" cool"],
    "warm": [" warm"],
    "hot": [" hot"],
}

CONTEXTS = [
    ("freezer air", -18),
    ("room air", 21),
    ("swimming-pool water", 27),
    ("bath water", 38),
    ("freshly served tea", 80),
    ("heated oven air", 180),
]

VALUES = [-30, -10, 5, 15, 25, 35, 45, 55, 85, 120, 200]


def label_for_delta(delta: float) -> str | None:
    if delta <= -20:
        return "cold"
    if -20 < delta <= -4:
        return "cool"
    if 4 <= delta < 20:
        return "warm"
    if delta >= 20:
        return "hot"
    return None


def prompt_for(context: str, standard: int, value: int) -> str:
    return (
        f"The normal temperature for {context} is {standard} degrees Celsius. "
        f"Compared only to that {standard}-degree baseline for {context}, "
        f"a temperature of {value} degrees Celsius feels"
    )


def slug(raw: str) -> str:
    return str(raw).replace(" ", "_").replace("-", "_")


def stable_flip(key: str) -> bool:
    h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2 == 1


def oriented_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    pair_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    if stable_flip(pair_key):
        return right, left, True
    return left, right, False


def build_items() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in VALUES:
        sides: List[Dict[str, Any]] = []
        for context, standard in CONTEXTS:
            label = label_for_delta(float(value - standard))
            if label is None:
                continue
            target = f"{value} degrees Celsius"
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
                    "delta_from_standard": int(value - standard),
                }
            )
        for i, left in enumerate(sides):
            for right in sides[i + 1 :]:
                if left["expected_label"] == right["expected_label"]:
                    continue
                value_tag = f"m{abs(int(value)):03d}" if int(value) < 0 else f"p{int(value):03d}"
                base_key = f"temp-v1_2-{value_tag}-{slug(left['context'])}-vs-{slug(right['context'])}"
                a, b, flipped = oriented_pair(left, right, pair_key=base_key)
                pair_id = f"temp-v1_2-{value_tag}-{slug(a['context'])}-vs-{slug(b['context'])}"
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
                            "dimension": "temperature",
                            "unit": "celsius",
                            "value": int(value),
                            "predicate_family": "warm_cold",
                            "comparison_a": str(a["context"]),
                            "comparison_b": str(b["context"]),
                            "standard_a": int(a["standard"]),
                            "standard_b": int(b["standard"]),
                            "delta_a": int(a["delta_from_standard"]),
                            "delta_b": int(b["delta_from_standard"]),
                            "pair_orientation_base_key": str(base_key),
                            "pair_flipped": bool(flipped),
                            "standard_type": "comparison_class_explicit",
                            "scale_type": "relative_open_temperature",
                            "control_type": "standard_explicit",
                            "is_borderline": False,
                            "is_artificial_norm": False,
                            "design": "same_scalar_different_explicit_standard_balanced_orientation",
                            "source": "bierwisch_kennedy_temperature_v1_2",
                        },
                    }
                )
    return rows


def main() -> None:
    root = ROOT / "results" / "manifold_groups_poc"
    out_jsonl = root / "gradable_temperature_disamb_pairs_v1_2.jsonl"
    out_csv = root / "gradable_temperature_grid_v1_2.csv"
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
    n_flipped = sum(1 for r in rows if bool(r["metadata"].get("pair_flipped")))
    print(f"Wrote {len(rows)} pairs: {out_jsonl}")
    print(f"Wrote grid CSV: {out_csv}")
    print(f"Flipped pairs: {n_flipped}/{len(rows)}")


if __name__ == "__main__":
    main()
