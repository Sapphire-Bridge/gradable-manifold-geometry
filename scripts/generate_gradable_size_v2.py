"""Generate size-only v2 behavioral-lock-in datasets.

v2 is the next gated step after the v1.2/v1.3 sweep:

- natural: original world-anchored size contexts
- neutral: Class A/B/C-style contexts with the same standards
- iso_ratio: a controlled grid with repeated value/standard ratios
- artificial: world-anchored names with inverted/artificial standards
- fictional_semantic: invented but semantically scaffolded object classes
- counter_natural: familiar object names with explicitly fictional/inverted norms

All variants use the v1.2 explicit-baseline prompt family because that is the
current positive behavioral regime. The point of v2 is not representational
geometry yet; it is to lock in a robust size behavior before raw patching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


READOUTS: Dict[str, Dict[str, Any]] = {
    "adjective": {
        "label_order": ("tiny", "small", "large", "huge"),
        "choices": {
            "tiny": [" tiny"],
            "small": [" small"],
            "large": [" large"],
            "huge": [" huge"],
        },
        "prompt_end": "feels",
    },
    "adjective_counts": {
        "label_order": ("tiny", "small", "large", "huge"),
        "choices": {
            "tiny": [" tiny"],
            "small": [" small"],
            "large": [" large"],
            "huge": [" huge"],
        },
        "prompt_end": "counts as",
    },
    "normality4": {
        "label_order": ("far_below_normal", "below_normal", "above_normal", "far_above_normal"),
        "choices": {
            "far_below_normal": [" far below normal"],
            "below_normal": [" below normal"],
            "above_normal": [" above normal"],
            "far_above_normal": [" far above normal"],
        },
        "prompt_end": "counts as",
    },
    "binary": {
        "label_order": ("below_normal", "above_normal"),
        "choices": {
            "below_normal": [" below normal"],
            "above_normal": [" above normal"],
        },
        "prompt_end": "counts as",
    },
    "comparative4": {
        "label_order": (
            "much_shorter_than_normal",
            "slightly_shorter_than_normal",
            "slightly_longer_than_normal",
            "much_longer_than_normal",
        ),
        "choices": {
            "much_shorter_than_normal": [" much shorter than normal"],
            "slightly_shorter_than_normal": [" slightly shorter than normal"],
            "slightly_longer_than_normal": [" slightly longer than normal"],
            "much_longer_than_normal": [" much longer than normal"],
        },
        "prompt_end": "is",
    },
    "comparative2": {
        "label_order": ("shorter_than_normal", "longer_than_normal"),
        "choices": {
            "shorter_than_normal": [" shorter than normal"],
            "longer_than_normal": [" longer than normal"],
        },
        "prompt_end": "is",
    },
}

LABEL_ORDER = tuple(READOUTS["adjective"]["label_order"])

VARIANTS: Dict[str, Dict[str, Any]] = {
    "natural": {
        "contexts": [
            ("an ant", 1),
            ("a house mouse", 9),
            ("a domestic cat", 46),
            ("a labrador dog", 60),
            ("an adult human", 170),
            ("an african elephant", 350),
        ],
        "values": [1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500],
        "context_type": "natural_world_anchor",
        "description": "Original world-anchored size standards.",
    },
    "neutral": {
        "contexts": [
            ("Class A objects", 1),
            ("Class B objects", 9),
            ("Class C objects", 46),
            ("Class D objects", 60),
            ("Class E objects", 170),
            ("Class F objects", 350),
        ],
        "values": [1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500],
        "context_type": "neutral_class_labels",
        "description": "Neutral labels with the natural-context standards.",
    },
    "iso_ratio": {
        "contexts": [
            ("Scale class 30 objects", 30),
            ("Scale class 60 objects", 60),
            ("Scale class 120 objects", 120),
            ("Scale class 240 objects", 240),
        ],
        "values": [15, 30, 45, 60, 90, 120, 180, 240, 360, 480],
        "context_type": "neutral_iso_ratio_grid",
        "description": "Neutral grid with repeated value/standard ratios across absolute scales.",
    },
    "artificial": {
        "contexts": [
            ("ant-like objects in the toy lab", 300),
            ("mouse-like objects in the toy lab", 170),
            ("human-like objects in the toy lab", 60),
            ("elephant-like objects in the toy lab", 30),
        ],
        "values": [15, 25, 40, 60, 90, 130, 200, 320, 500],
        "context_type": "artificial_inverted_world_anchor",
        "description": "World-anchored nouns with deliberately inverted/artificial standards.",
    },
    "fictional_semantic": {
        "contexts": [
            ("Type A dax rods", 1),
            ("Type B dax rods", 9),
            ("Type C dax rods", 46),
            ("Type D dax rods", 60),
            ("Type E dax rods", 170),
            ("Type F dax rods", 350),
        ],
        "values": [1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500],
        "context_type": "fictional_semantic_scaffold",
        "description": "Invented rod-like objects with explicit dimensional semantics but no world-size prior.",
    },
    "counter_natural": {
        "contexts": [
            ("adult humans in this fictional setting", 30),
            ("labrador dogs in this fictional setting", 170),
            ("house mice in this fictional setting", 350),
            ("african elephants in this fictional setting", 60),
        ],
        "values": [15, 25, 40, 60, 90, 130, 200, 320, 500],
        "context_type": "counter_natural_explicit_norm",
        "description": "Familiar object names with fictional standards that conflict with real-world size priors.",
    },
}


def label_for_ratio(ratio: float, *, readout_family: str) -> str | None:
    if readout_family == "comparative2":
        if ratio < 0.85:
            return "shorter_than_normal"
        if ratio > 1.15:
            return "longer_than_normal"
        return None
    if readout_family == "comparative4":
        if ratio <= 0.5:
            return "much_shorter_than_normal"
        if 0.5 < ratio <= 0.85:
            return "slightly_shorter_than_normal"
        if 1.15 <= ratio < 2.0:
            return "slightly_longer_than_normal"
        if ratio >= 2.0:
            return "much_longer_than_normal"
        return None
    if readout_family == "binary":
        if ratio < 0.85:
            return "below_normal"
        if ratio > 1.15:
            return "above_normal"
        return None
    if readout_family == "normality4":
        if ratio <= 0.5:
            return "far_below_normal"
        if 0.5 < ratio <= 0.85:
            return "below_normal"
        if 1.15 <= ratio < 2.0:
            return "above_normal"
        if ratio >= 2.0:
            return "far_above_normal"
        return None
    if readout_family not in {"adjective", "adjective_counts"}:
        raise ValueError(f"Unknown readout family {readout_family!r}; expected one of {sorted(READOUTS)}")
    if ratio <= 0.5:
        return "tiny"
    if 0.5 < ratio <= 0.85:
        return "small"
    if 1.15 <= ratio < 2.0:
        return "large"
    if ratio >= 2.0:
        return "huge"
    return None


def cm_phrase(value: int) -> str:
    return f"{int(value)} centimeter" if int(value) == 1 else f"{int(value)} centimeters"


def cm_modifier(value: int) -> str:
    return f"{int(value)}-centimeter"


def prompt_for(context: str, standard: int, value: int, *, variant: str, readout_family: str) -> str:
    standard_text = cm_phrase(standard)
    standard_modifier = cm_modifier(standard)
    value_text = cm_phrase(value)
    prompt_end = str(READOUTS[readout_family]["prompt_end"])
    if variant == "fictional_semantic":
        return (
            "In this measurement task, a dax is a manufactured rod-like object. "
            "Only overall length matters. "
            f"The normal overall length for {context} is {standard_text}. "
            f"Compared only to that {standard_modifier} length standard for {context}, "
            f"an overall length of {value_text} {prompt_end}"
        )
    if variant == "counter_natural":
        return (
            "In this fictional setting, ignore real-world sizes. "
            f"The normal body length for {context} is {standard_text}. "
            f"Compared only to that {standard_modifier} fictional baseline for {context}, "
            f"a body length of {value_text} {prompt_end}"
        )
    if variant == "iso_ratio":
        return (
            f"The normal overall length for {context} is {standard_text}. "
            f"Compared only to that {standard_modifier} length standard for {context}, "
            f"an overall length of {value_text} {prompt_end}"
        )
    if variant == "neutral":
        return (
            f"The normal overall length for {context} is {standard_text}. "
            f"Compared only to that {standard_modifier} length standard for {context}, "
            f"an overall length of {value_text} {prompt_end}"
        )
    return (
        f"The normal body length for {context} is {standard_text}. "
        f"Compared only to that {standard_modifier} baseline for {context}, "
        f"a body length of {value_text} {prompt_end}"
    )


def slug(raw: str) -> str:
    return str(raw).replace(" ", "_").replace("-", "_").replace("/", "_")


def stable_flip(key: str) -> bool:
    h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2 == 1


def oriented_pair(left: Dict[str, Any], right: Dict[str, Any], *, pair_key: str) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    if stable_flip(pair_key):
        return right, left, True
    return left, right, False


def version_for(variant: str, readout_family: str) -> str:
    suffix = "" if str(readout_family) == "adjective" else f"_{readout_family}"
    return f"v2_{variant}{suffix}"


def _build_sides(variant: str, *, readout_family: str) -> List[Dict[str, Any]]:
    cfg = VARIANTS[variant]
    sides: List[Dict[str, Any]] = []
    for value in cfg["values"]:
        for context, standard in cfg["contexts"]:
            ratio = float(value) / float(standard)
            label = label_for_ratio(ratio, readout_family=readout_family)
            if label is None:
                continue
            target = cm_phrase(int(value))
            prompt = prompt_for(str(context), int(standard), int(value), variant=variant, readout_family=readout_family)
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
                    "log_ratio_from_standard": float(math.log(ratio)),
                    "ratio_bucket": f"{ratio:.6g}",
                }
            )
    return sides


def _pair_sides(sides: Iterable[Dict[str, Any]], *, variant: str, readout_family: str) -> List[Dict[str, Any]]:
    by_value: Dict[int, List[Dict[str, Any]]] = {}
    for side in sides:
        by_value.setdefault(int(side["value"]), []).append(side)

    rows: List[Dict[str, Any]] = []
    for value, value_sides in sorted(by_value.items()):
        for i, left in enumerate(value_sides):
            for right in value_sides[i + 1 :]:
                if left["expected_label"] == right["expected_label"]:
                    continue
                value_tag = f"p{int(value):04d}"
                base_key = f"size-v2-{variant}-{readout_family}-{value_tag}-{slug(left['context'])}-vs-{slug(right['context'])}"
                a, b, flipped = oriented_pair(left, right, pair_key=base_key)
                pair_id = f"size-v2-{variant}-{value_tag}-{slug(a['context'])}-vs-{slug(b['context'])}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "target": str(a["target"]),
                        "target_occurrence": 0,
                        "a": {"prompt": str(a["prompt"]), "expected_label": str(a["expected_label"])},
                        "b": {"prompt": str(b["prompt"]), "expected_label": str(b["expected_label"])},
                        "choices": READOUTS[readout_family]["choices"],
                        "metadata": {
                            "type": "gradable_scalar",
                            "dimension": "size",
                            "unit": "centimeters",
                            "value": int(value),
                            "predicate_family": str(readout_family),
                            "readout_family": str(readout_family),
                            "label_order": ",".join(str(x) for x in READOUTS[readout_family]["label_order"]),
                            "comparison_a": str(a["context"]),
                            "comparison_b": str(b["context"]),
                            "standard_a": int(a["standard"]),
                            "standard_b": int(b["standard"]),
                            "ratio_a": float(a["ratio_from_standard"]),
                            "ratio_b": float(b["ratio_from_standard"]),
                            "log_ratio_a": float(a["log_ratio_from_standard"]),
                            "log_ratio_b": float(b["log_ratio_from_standard"]),
                            "ratio_bucket_a": str(a["ratio_bucket"]),
                            "ratio_bucket_b": str(b["ratio_bucket"]),
                            "pair_orientation_base_key": str(base_key),
                            "pair_flipped": bool(flipped),
                            "standard_type": "comparison_class_explicit",
                            "scale_type": "relative_open_size",
                            "control_type": "standard_explicit",
                            "context_type": str(VARIANTS[variant]["context_type"]),
                            "regime": str(variant),
                            "is_borderline": False,
                            "is_artificial_norm": bool(variant == "artificial"),
                            "is_counter_natural": bool(variant == "counter_natural"),
                            "has_semantic_scaffold": bool(variant in {"natural", "iso_ratio", "fictional_semantic", "counter_natural", "artificial"}),
                            "design": "size_v2_behavioral_lockin_same_value_different_standard",
                            "source": f"bierwisch_kennedy_size_v2_{variant}",
                        },
                    }
                )
    return rows


def build_items(variant: str, readout_family: str = "adjective") -> List[Dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    if readout_family not in READOUTS:
        raise ValueError(f"Unknown readout_family {readout_family!r}; expected one of {sorted(READOUTS)}")
    return _pair_sides(_build_sides(variant, readout_family=readout_family), variant=variant, readout_family=readout_family)


def write_outputs(variant: str, readout_family: str, rows: List[Dict[str, Any]]) -> None:
    root = ROOT / "results" / "manifold_groups_poc"
    version = version_for(variant, readout_family)
    out_jsonl = root / f"gradable_size_disamb_pairs_{version}.jsonl"
    out_csv = root / f"gradable_size_grid_{version}.csv"
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
    print(f"[size/{variant}] Wrote {len(rows)} pairs: {out_jsonl}")
    print(f"[size/{variant}] Wrote grid CSV: {out_csv}")
    print(f"[size/{variant}] Flipped pairs: {n_flipped}/{len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate size-only v2 behavioral-lock-in datasets.")
    p.add_argument("--variant", choices=sorted([*VARIANTS.keys(), "all"]), default="all")
    p.add_argument("--readout_family", choices=sorted(READOUTS.keys()), default="adjective")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = sorted(VARIANTS) if str(args.variant) == "all" else [str(args.variant)]
    for variant in variants:
        write_outputs(variant, str(args.readout_family), build_items(variant, readout_family=str(args.readout_family)))


if __name__ == "__main__":
    main()
