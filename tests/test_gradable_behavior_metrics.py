from __future__ import annotations

import csv
import importlib
import json
from collections import Counter
from pathlib import Path

from scripts.recompute_gradable_behavior_metrics import recompute_domain


def test_v1_2_generators_balance_expected_shift_orientation() -> None:
    cases = [
        ("scripts.generate_gradable_temperature_v1_2", ["cold", "cool", "warm", "hot"]),
        ("scripts.generate_gradable_size_v1_2", ["tiny", "small", "large", "huge"]),
        ("scripts.generate_gradable_age_v1_2", ["young", "youthful", "mature", "old"]),
    ]
    for module_name, label_order in cases:
        module = importlib.import_module(module_name)
        rows = module.build_items()
        order = {label: i for i, label in enumerate(label_order)}
        signs: Counter[str] = Counter()
        bad_target_counts = 0
        for row in rows:
            shift = order[row["b"]["expected_label"]] - order[row["a"]["expected_label"]]
            signs["positive" if shift > 0 else "negative" if shift < 0 else "zero"] += 1
            target = str(row["target"])
            bad_target_counts += int(row["a"]["prompt"].count(target) != 1)
            bad_target_counts += int(row["b"]["prompt"].count(target) != 1)
        assert signs["positive"] > 0
        assert signs["negative"] > 0
        assert signs["zero"] == 0
        assert abs(signs["positive"] - signs["negative"]) <= 5
        assert bad_target_counts == 0


def test_size_v2_generator_variants_are_balanced_and_valid() -> None:
    module = importlib.import_module("scripts.generate_gradable_size_v2")
    order = {label: i for i, label in enumerate(("tiny", "small", "large", "huge"))}
    for variant in ("natural", "neutral", "iso_ratio", "artificial", "fictional_semantic", "counter_natural"):
        rows = module.build_items(variant)
        assert rows
        signs: Counter[str] = Counter()
        context_types: set[str] = set()
        for row in rows:
            shift = order[row["b"]["expected_label"]] - order[row["a"]["expected_label"]]
            signs["positive" if shift > 0 else "negative" if shift < 0 else "zero"] += 1
            assert signs["zero"] == 0
            target = str(row["target"])
            assert row["a"]["prompt"].count(target) == 1
            assert row["b"]["prompt"].count(target) == 1
            md = row["metadata"]
            assert md["regime"] == variant
            assert float(md["ratio_a"]) > 0.0
            assert float(md["ratio_b"]) > 0.0
            context_types.add(str(md["context_type"]))
        assert signs["positive"] > 0
        assert signs["negative"] > 0
        assert len(context_types) == 1


def test_size_v2_generator_supports_normality_readouts() -> None:
    module = importlib.import_module("scripts.generate_gradable_size_v2")
    cases = [
        ("adjective_counts", ("tiny", "small", "large", "huge")),
        ("normality4", ("far_below_normal", "below_normal", "above_normal", "far_above_normal")),
        ("binary", ("below_normal", "above_normal")),
        (
            "comparative4",
            (
                "much_shorter_than_normal",
                "slightly_shorter_than_normal",
                "slightly_longer_than_normal",
                "much_longer_than_normal",
            ),
        ),
        ("comparative2", ("shorter_than_normal", "longer_than_normal")),
    ]
    for readout_family, labels in cases:
        rows = module.build_items("fictional_semantic", readout_family=readout_family)
        assert rows
        seen = {str(row["a"]["expected_label"]) for row in rows} | {str(row["b"]["expected_label"]) for row in rows}
        assert seen.issubset(set(labels))
        assert len(seen) >= 2
        first = rows[0]
        assert first["metadata"]["readout_family"] == readout_family
        assert set(first["choices"]) == set(labels)
        assert "overall length" in first["a"]["prompt"]
        assert "body length" not in first["a"]["prompt"]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def test_recompute_domain_rebuilds_ordered_scores_from_prob_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "size.jsonl"
    sides_path = tmp_path / "size.sides.csv"
    pairs_path = tmp_path / "size.pairs.csv"
    item = {
        "pair_id": "p1",
        "target": "10 centimeters",
        "target_occurrence": 0,
        "a": {"prompt": "A", "expected_label": "tiny"},
        "b": {"prompt": "B", "expected_label": "huge"},
        "choices": {"tiny": [" tiny"], "small": [" small"], "large": [" large"], "huge": [" huge"]},
        "metadata": {
            "dimension": "size",
            "value": 10,
            "comparison_a": "small baseline",
            "comparison_b": "large baseline",
            "standard_a": 100,
            "standard_b": 2,
            "ratio_a": 0.1,
            "ratio_b": 5.0,
        },
    }
    data_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    _write_csv(
        sides_path,
        [
            {
                "pair_id": "p1",
                "side": "a",
                "target": "10 centimeters",
                "prompt": "A",
                "expected_label": "tiny",
                "pred_label": "small",
                "expected_label_index": "nan",
                "ordered_label_score": "nan",
                "entropy": "0.0",
                "prob_tiny": 0.8,
                "prob_small": 0.1,
                "prob_large": 0.05,
                "prob_huge": 0.05,
            },
            {
                "pair_id": "p1",
                "side": "b",
                "target": "10 centimeters",
                "prompt": "B",
                "expected_label": "huge",
                "pred_label": "small",
                "expected_label_index": "nan",
                "ordered_label_score": "nan",
                "entropy": "0.0",
                "prob_tiny": 0.05,
                "prob_small": 0.05,
                "prob_large": 0.1,
                "prob_huge": 0.8,
            },
        ],
    )
    _write_csv(pairs_path, [{"pair_id": "p1", "expected_shift": "nan", "observed_ordered_shift": "nan"}])

    summary, side_rows, pair_rows = recompute_domain(
        domain="size",
        data_path=data_path,
        sides_path=sides_path,
        pairs_path=pairs_path,
        label_order=("tiny", "small", "large", "huge"),
        bootstrap_b=20,
        permutation_b=20,
        seed=0,
    )

    assert summary["counts"]["n_unique_sides"] == 2
    assert summary["consistency_checks"]["expected_labels_not_in_label_order"] == 0
    assert summary["shift_direction_match"]["count"] == 1
    assert pair_rows[0]["expected_shift"] == 3.0
    assert pair_rows[0]["observed_ordered_shift"] > 0.0
    assert side_rows[0]["ordered_score"] < side_rows[1]["ordered_score"]

