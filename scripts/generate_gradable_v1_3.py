"""Paraphrase sweep generator for v1.3.

Holds the same per-domain CONTEXTS, VALUES, and label-threshold logic as the
v1_2 generators, but supports three paraphrase variants (a/b/c) of the
prompt template that elicit the explicit comparison standard differently.
Output naming: gradable_{domain}_disamb_pairs_v1_3_{paraphrase}.jsonl etc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "temperature": {
        "choices": {
            "cold": [" cold"],
            "cool": [" cool"],
            "warm": [" warm"],
            "hot": [" hot"],
        },
        "contexts": [
            ("freezer air", -18),
            ("room air", 21),
            ("swimming-pool water", 27),
            ("bath water", 38),
            ("freshly served tea", 80),
            ("heated oven air", 180),
        ],
        "values": [-30, -10, 5, 15, 25, 35, 45, 55, 85, 120, 200],
        "unit": "degrees Celsius",
        "unit_short": "degree",
        "dimension_word": "temperature",
        "predicate_family": "warm_cold",
        "scale_type": "relative_open_temperature",
        "pair_id_prefix": "temp",
        "source_prefix": "bierwisch_kennedy_temperature",
        "predictor_kind": "delta",
        "label_function_kind": "delta",
    },
    "size": {
        "choices": {
            "tiny": [" tiny"],
            "small": [" small"],
            "large": [" large"],
            "huge": [" huge"],
        },
        "contexts": [
            ("an ant", 1),
            ("a house mouse", 9),
            ("a domestic cat", 46),
            ("a labrador dog", 60),
            ("an adult human", 170),
            ("an african elephant", 350),
        ],
        "values": [1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500],
        "unit": "centimeters",
        "unit_short": "centimeter",
        "dimension_word": "body length",
        "predicate_family": "tiny_huge",
        "scale_type": "relative_open_size",
        "pair_id_prefix": "size",
        "source_prefix": "bierwisch_kennedy_size",
        "predictor_kind": "ratio",
        "label_function_kind": "ratio",
    },
    "age": {
        "choices": {
            "young": [" young"],
            "youthful": [" youthful"],
            "mature": [" mature"],
            "old": [" old"],
        },
        "contexts": [
            ("a domestic mouse", 2),
            ("a domestic dog", 12),
            ("a horse", 28),
            ("a human", 80),
            ("a galapagos tortoise", 150),
            ("a bowhead whale", 200),
        ],
        "values": [1, 3, 6, 10, 18, 30, 55, 90, 130, 180],
        "unit": "years",
        "unit_short": "year",
        "dimension_word": "lifespan",
        "predicate_family": "young_old",
        "scale_type": "relative_open_age",
        "pair_id_prefix": "age",
        "source_prefix": "bierwisch_kennedy_age",
        "predictor_kind": "ratio",
        "label_function_kind": "ratio",
    },
}


def label_for_temperature_delta(delta: float) -> str | None:
    if delta <= -20:
        return "cold"
    if -20 < delta <= -4:
        return "cool"
    if 4 <= delta < 20:
        return "warm"
    if delta >= 20:
        return "hot"
    return None


def label_for_size_ratio(ratio: float) -> str | None:
    if ratio <= 0.5:
        return "tiny"
    if 0.5 < ratio <= 0.85:
        return "small"
    if 1.15 <= ratio < 2.0:
        return "large"
    if ratio >= 2.0:
        return "huge"
    return None


def label_for_age_ratio(ratio: float) -> str | None:
    if ratio <= 0.15:
        return "young"
    if 0.15 < ratio <= 0.40:
        return "youthful"
    if 0.55 <= ratio < 0.85:
        return "mature"
    if ratio >= 0.85:
        return "old"
    return None


LABEL_FUNCTIONS: Dict[str, Callable[[float], str | None]] = {
    "temperature": label_for_temperature_delta,
    "size": label_for_size_ratio,
    "age": label_for_age_ratio,
}


# --- Paraphrase templates ------------------------------------------------
# Each paraphrase is a function (context, standard, value, cfg) -> str
# Style:
#   A: "Relative to ... reference, ... is best described as" (analytical)
#   B: "If ... is ..., then ... would best be called" (conditional)
#   C: "For ..., normal is ... Compared to that, ... is" (elliptic)


def paraphrase_a(context: str, standard: int, value: int, cfg: Dict[str, Any]) -> str:
    unit = cfg["unit"]
    unit_short = cfg["unit_short"]
    dim = cfg["dimension_word"]
    if cfg["pair_id_prefix"] == "age":
        head = f"For {context}, a typical {dim} is {standard} {unit}."
        ref = f"Relative to that {standard}-{unit_short} reference,"
        tail = f"an individual aged {value} {unit} is best described as"
    else:
        head = f"For {context}, a typical {dim} is {standard} {unit}."
        ref = f"Relative to that {standard}-{unit_short} reference,"
        tail = f"a {dim} of {value} {unit} is best described as"
    return f"{head} {ref} {tail}"


def paraphrase_b(context: str, standard: int, value: int, cfg: Dict[str, Any]) -> str:
    unit = cfg["unit"]
    dim = cfg["dimension_word"]
    if cfg["pair_id_prefix"] == "age":
        return (
            f"If the normal {dim} for {context} is {standard} {unit}, "
            f"then someone aged {value} {unit} would best be called"
        )
    return (
        f"If the normal {dim} for {context} is {standard} {unit}, "
        f"then {value} {unit} would best be called"
    )


def paraphrase_c(context: str, standard: int, value: int, cfg: Dict[str, Any]) -> str:
    unit = cfg["unit"]
    return (
        f"For {context}, normal is {standard} {unit}. "
        f"Compared to that, {value} {unit} is"
    )


PARAPHRASE_TEMPLATES: Dict[str, Callable[[str, int, int, Dict[str, Any]], str]] = {
    "a": paraphrase_a,
    "b": paraphrase_b,
    "c": paraphrase_c,
}


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


def build_items(domain: str, paraphrase: str) -> List[Dict[str, Any]]:
    cfg = DOMAIN_CONFIGS[domain]
    label_fn = LABEL_FUNCTIONS[domain]
    template_fn = PARAPHRASE_TEMPLATES[paraphrase]
    contexts = cfg["contexts"]
    values = cfg["values"]
    unit = cfg["unit"]
    pair_prefix = cfg["pair_id_prefix"]
    rows: List[Dict[str, Any]] = []

    for value in values:
        sides: List[Dict[str, Any]] = []
        for context, standard in contexts:
            if cfg["label_function_kind"] == "delta":
                label = label_fn(float(value - standard))
                metric_value = int(value - standard)
                metric_key = "delta_from_standard"
            else:
                label = label_fn(float(value) / float(standard))
                metric_value = float(value) / float(standard)
                metric_key = "ratio_from_standard"
            if label is None:
                continue
            if domain == "age":
                target = f"{value} {unit}"
            else:
                target = f"{value} {unit}"
            prompt = template_fn(context, int(standard), int(value), cfg)
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
                    metric_key: metric_value,
                }
            )
        for i, left in enumerate(sides):
            for right in sides[i + 1 :]:
                if left["expected_label"] == right["expected_label"]:
                    continue
                if cfg["label_function_kind"] == "delta":
                    value_tag = f"m{abs(int(value)):03d}" if int(value) < 0 else f"p{int(value):03d}"
                else:
                    value_tag = f"p{int(value):04d}"
                base_key = (
                    f"{pair_prefix}-v1_3{paraphrase}-{value_tag}-"
                    f"{slug(left['context'])}-vs-{slug(right['context'])}"
                )
                a, b, flipped = oriented_pair(left, right, pair_key=base_key)
                pair_id = (
                    f"{pair_prefix}-v1_3{paraphrase}-{value_tag}-"
                    f"{slug(a['context'])}-vs-{slug(b['context'])}"
                )
                md: Dict[str, Any] = {
                    "type": "gradable_scalar",
                    "dimension": pair_prefix if pair_prefix != "temp" else "temperature",
                    "unit": cfg["unit"].split()[-1] if pair_prefix == "temp" else cfg["unit"],
                    "value": int(value),
                    "predicate_family": cfg["predicate_family"],
                    "comparison_a": str(a["context"]),
                    "comparison_b": str(b["context"]),
                    "standard_a": int(a["standard"]),
                    "standard_b": int(b["standard"]),
                    "pair_orientation_base_key": str(base_key),
                    "pair_flipped": bool(flipped),
                    "standard_type": "comparison_class_explicit",
                    "scale_type": cfg["scale_type"],
                    "control_type": "standard_explicit",
                    "is_borderline": False,
                    "is_artificial_norm": False,
                    "design": "same_scalar_different_explicit_standard_balanced_orientation_paraphrase_sweep",
                    "source": f"{cfg['source_prefix']}_v1_3_{paraphrase}",
                    "paraphrase": paraphrase,
                }
                if cfg["label_function_kind"] == "delta":
                    md["delta_a"] = int(a["delta_from_standard"])
                    md["delta_b"] = int(b["delta_from_standard"])
                else:
                    md["ratio_a"] = float(a["ratio_from_standard"])
                    md["ratio_b"] = float(b["ratio_from_standard"])
                rows.append(
                    {
                        "pair_id": pair_id,
                        "target": str(a["target"]),
                        "target_occurrence": 0,
                        "a": {"prompt": str(a["prompt"]), "expected_label": str(a["expected_label"])},
                        "b": {"prompt": str(b["prompt"]), "expected_label": str(b["expected_label"])},
                        "choices": cfg["choices"],
                        "metadata": md,
                    }
                )
    return rows


def write_outputs(domain: str, paraphrase: str, rows: List[Dict[str, Any]]) -> None:
    root = ROOT / "results" / "manifold_groups_poc"
    version = f"v1_3_{paraphrase}"
    out_jsonl = root / f"gradable_{domain}_disamb_pairs_{version}.jsonl"
    out_csv = root / f"gradable_{domain}_grid_{version}.csv"
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
    print(f"[{domain}/{paraphrase}] Wrote {len(rows)} pairs: {out_jsonl}")
    print(f"[{domain}/{paraphrase}] Wrote grid CSV: {out_csv}")
    print(f"[{domain}/{paraphrase}] Flipped pairs: {n_flipped}/{len(rows)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate v1.3 paraphrase-swept gradable predicate datasets.")
    p.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS.keys()), required=True)
    p.add_argument("--paraphrase", choices=sorted(PARAPHRASE_TEMPLATES.keys()), required=True)
    args = p.parse_args()
    rows = build_items(args.domain, args.paraphrase)
    write_outputs(args.domain, args.paraphrase, rows)


if __name__ == "__main__":
    main()
