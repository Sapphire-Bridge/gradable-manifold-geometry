from __future__ import annotations

import math

import numpy as np

from scripts.steer_gradable_size import build_steering_directions_for_layer, summarize_rows


def _train_rows() -> list[dict[str, object]]:
    return [
        {"pair_id": "p1", "rho": -1.0, "ordered_score": 0.5, "signed_score": -0.5, "value": 10.0, "standard": 30.0},
        {"pair_id": "p1", "rho": 1.0, "ordered_score": 2.5, "signed_score": 0.5, "value": 10.0, "standard": 3.0},
        {"pair_id": "p2", "rho": -0.8, "ordered_score": 0.7, "signed_score": -0.4, "value": 20.0, "standard": 60.0},
        {"pair_id": "p2", "rho": 0.8, "ordered_score": 2.3, "signed_score": 0.4, "value": 20.0, "standard": 6.0},
    ]


def test_pca_delta_mean_direction_is_oriented_high_rho_minus_low_rho() -> None:
    X = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [2.0, 0.2, 0.0],
            [0.2, 1.0, 0.0],
            [2.2, 1.1, 0.0],
        ],
        dtype=np.float32,
    )
    directions = build_steering_directions_for_layer(
        X_train=X,
        train_rows=_train_rows(),
        layer=20,
        methods=["pca_delta_mean", "sham"],
        rank=1,
        ridge_alpha=1.0,
        seed=0,
        random_repeats=0,
    )

    primary = next(row for row in directions if row["method"] == "pca_delta_mean")
    unit = np.asarray(primary["unit_direction"])
    assert float(np.dot(unit, np.asarray([1.0, 0.0, 0.0]))) > 0.99
    assert math.isclose(float(primary["direction_unit_norm"]), 1.0, rel_tol=1e-6)
    assert float(primary["steer_scale"]) > 0.0

    sham = next(row for row in directions if row["method"] == "sham")
    assert float(sham["steer_scale"]) == 0.0


def test_steering_summary_reports_alpha_slope_and_signed_effect() -> None:
    rows = []
    for pair_id in ("p1", "p2"):
        for side in ("a", "b"):
            for alpha in (-1.0, 0.0, 1.0):
                effect = 0.2 * alpha
                rows.append(
                    {
                        "method": "pca_delta_mean",
                        "layer": 20,
                        "rank": 5,
                        "repeat": 0,
                        "pair_id": pair_id,
                        "side": side,
                        "alpha": alpha,
                        "effect": effect,
                        "signed_effect": effect * (1.0 if alpha > 0 else -1.0 if alpha < 0 else 0.0),
                        "effect_per_alpha": effect / alpha if alpha else float("nan"),
                        "base_ordered_score": 1.0,
                        "patched_ordered_score": 1.0 + effect,
                        "direction_match": 1.0 if alpha else float("nan"),
                        "steer_l2_norm": abs(alpha),
                        "steer_scale": 1.0,
                    }
                )

    summary = summarize_rows(rows, bootstrap_b=20, seed=0)
    curve = summary["curve_groups"]["L20_pca_delta_mean_rank5"]
    assert abs(float(curve["score_slope_per_alpha"]["mean"]) - 0.2) < 1e-9
    assert float(curve["positive_slope_rate"]) == 1.0

    alpha_group = summary["alpha_groups"]["L20_pca_delta_mean_rank5_alpha-1"]
    assert abs(float(alpha_group["signed_effect"]["mean"]) - 0.2) < 1e-9
    assert float(alpha_group["direction_match_rate"]) == 1.0
