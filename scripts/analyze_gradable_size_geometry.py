from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"


def _parse_csv_or_space(raw: str) -> List[str]:
    return [part.strip() for part in re.split(r"[,\s]+", str(raw)) if part.strip()]


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x) for x in _parse_csv_or_space(raw)]
    if not vals:
        raise ValueError("Expected at least one layer")
    return vals


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _to_float(raw: Any, default: float = float("nan")) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite(float(x)) and _is_finite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / den) if den > 0.0 else float("nan")


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    pairs = [(float(y), float(p)) for y, p in zip(y_true, y_pred) if _is_finite(float(y)) and _is_finite(float(p))]
    if len(pairs) < 3:
        return float("nan")
    y = np.asarray([p[0] for p in pairs], dtype=np.float64)
    pred = np.asarray([p[1] for p in pairs], dtype=np.float64)
    den = float(np.sum((y - y.mean()) ** 2))
    if den <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / den)


def _standardize_train_test(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train = X[train_idx].astype(np.float64)
    test = X[test_idx].astype(np.float64)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def _ridge_dual_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    Xtr, Xte = _standardize_train_test(X, train_idx, test_idx)
    ytr = y[train_idx].astype(np.float64)
    y_mean = float(np.mean(ytr))
    yc = ytr - y_mean
    K = Xtr @ Xtr.T
    coefs = np.linalg.solve(K + float(alpha) * np.eye(K.shape[0], dtype=np.float64), yc)
    return y_mean + Xte @ Xtr.T @ coefs


def _pca_ridge_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> np.ndarray:
    Xtr, Xte = _standardize_train_test(X, train_idx, test_idx)
    ytr = y[train_idx].astype(np.float64)
    y_mean = float(np.mean(ytr))
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    actual_rank = int(min(int(rank), Vt.shape[0]))
    comps = Vt[:actual_rank].T
    Ztr = Xtr @ comps
    Zte = Xte @ comps
    A = Ztr.T @ Ztr + float(alpha) * np.eye(actual_rank, dtype=np.float64)
    w = np.linalg.solve(A, Ztr.T @ (ytr - y_mean))
    return y_mean + Zte @ w


def _metric_row(name: str, target: str, y: np.ndarray, pred: np.ndarray, test_idx: np.ndarray, *, layer: int, model: str) -> Dict[str, Any]:
    return {
        "layer": int(layer),
        "split": str(name),
        "target": str(target),
        "model": str(model),
        "n_test": int(len(test_idx)),
        "r": _pearson(y[test_idx].tolist(), pred.tolist()),
        "r2": _r2(y[test_idx].tolist(), pred.tolist()),
    }


def _valid_target_indices(y: np.ndarray) -> np.ndarray:
    return np.asarray([i for i, val in enumerate(y.tolist()) if _is_finite(float(val))], dtype=np.int64)


def _crossfit_group_split(
    X: np.ndarray,
    y: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    row_filter: np.ndarray,
    layer: int,
    target: str,
    alpha: float,
    pca_ranks: Sequence[int],
) -> List[Dict[str, Any]]:
    candidate_idx = np.asarray([i for i in np.where(row_filter)[0] if _is_finite(float(y[i]))], dtype=np.int64)
    groups = sorted({str(rows[int(i)].get(group_key, "")) for i in candidate_idx})
    if len(candidate_idx) < 8 or len(groups) < 2:
        return []
    pred_full = np.full((len(rows),), np.nan, dtype=np.float64)
    pred_by_rank: Dict[int, np.ndarray] = {int(k): np.full((len(rows),), np.nan, dtype=np.float64) for k in pca_ranks}
    test_all: List[int] = []
    for group in groups:
        test_idx = np.asarray([int(i) for i in candidate_idx if str(rows[int(i)].get(group_key, "")) == group], dtype=np.int64)
        train_idx = np.asarray([int(i) for i in candidate_idx if str(rows[int(i)].get(group_key, "")) != group], dtype=np.int64)
        if len(train_idx) < 4 or len(test_idx) < 1:
            continue
        pred_full[test_idx] = _ridge_dual_predict(X, y, train_idx, test_idx, alpha=alpha)
        for rank in pca_ranks:
            if int(rank) <= min(len(train_idx), X.shape[1]):
                pred_by_rank[int(rank)][test_idx] = _pca_ridge_predict(X, y, train_idx, test_idx, rank=int(rank), alpha=alpha)
        test_all.extend(int(i) for i in test_idx.tolist())
    test_idx = np.asarray(sorted(set(test_all)), dtype=np.int64)
    if len(test_idx) < 3:
        return []
    rows_out = [_metric_row(f"leave_{group_key}_out", target, y, pred_full[test_idx], test_idx, layer=layer, model="ridge_full")]
    for rank in pca_ranks:
        pred = pred_by_rank[int(rank)][test_idx]
        rows_out.append(_metric_row(f"leave_{group_key}_out", target, y, pred, test_idx, layer=layer, model=f"pca{int(rank)}_ridge"))
    return rows_out


def _cross_variant_split(
    X: np.ndarray,
    y: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_variant: str,
    test_variant: str,
    layer: int,
    target: str,
    alpha: float,
    pca_ranks: Sequence[int],
) -> List[Dict[str, Any]]:
    train_idx = np.asarray(
        [i for i, row in enumerate(rows) if str(row.get("variant", "")) == str(train_variant) and _is_finite(float(y[i]))],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [i for i, row in enumerate(rows) if str(row.get("variant", "")) == str(test_variant) and _is_finite(float(y[i]))],
        dtype=np.int64,
    )
    if len(train_idx) < 4 or len(test_idx) < 3:
        return []
    name = f"{train_variant}_to_{test_variant}"
    pred_full = _ridge_dual_predict(X, y, train_idx, test_idx, alpha=alpha)
    out = [_metric_row(name, target, y, pred_full, test_idx, layer=layer, model="ridge_full")]
    for rank in pca_ranks:
        if int(rank) <= min(len(train_idx), X.shape[1]):
            pred = _pca_ridge_predict(X, y, train_idx, test_idx, rank=int(rank), alpha=alpha)
            out.append(_metric_row(name, target, y, pred, test_idx, layer=layer, model=f"pca{int(rank)}_ridge"))
    return out


def _standardize_full(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def _distance_diagnostics(X: np.ndarray, rows: Sequence[Mapping[str, Any]], *, layer: int) -> Dict[str, Any]:
    Z = _standardize_full(X)
    dists: List[float] = []
    drho: List[float] = []
    dvalue: List[float] = []
    dstandard: List[float] = []
    same_context: List[float] = []
    same_variant: List[float] = []
    same_rho_dists: List[float] = []
    same_value_dists: List[float] = []
    same_standard_dists: List[float] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dist = float(np.linalg.norm(Z[i] - Z[j]))
            rho_i = _to_float(rows[i].get("rho"))
            rho_j = _to_float(rows[j].get("rho"))
            value_i = max(_to_float(rows[i].get("value")), 1e-9)
            value_j = max(_to_float(rows[j].get("value")), 1e-9)
            standard_i = max(_to_float(rows[i].get("standard")), 1e-9)
            standard_j = max(_to_float(rows[j].get("standard")), 1e-9)
            dists.append(dist)
            drho.append(abs(rho_i - rho_j))
            dvalue.append(abs(math.log(value_i) - math.log(value_j)))
            dstandard.append(abs(math.log(standard_i) - math.log(standard_j)))
            same_context.append(float(str(rows[i].get("comparison", "")) == str(rows[j].get("comparison", ""))))
            same_variant.append(float(str(rows[i].get("variant", "")) == str(rows[j].get("variant", ""))))
            if abs(rho_i - rho_j) < 1e-6:
                same_rho_dists.append(dist)
            if abs(value_i - value_j) < 1e-6:
                same_value_dists.append(dist)
            if abs(standard_i - standard_j) < 1e-6:
                same_standard_dists.append(dist)
    y = np.asarray(dists, dtype=np.float64)
    continuous = [
        np.asarray(drho, dtype=np.float64),
        np.asarray(dvalue, dtype=np.float64),
        np.asarray(dstandard, dtype=np.float64),
        np.asarray(same_context, dtype=np.float64),
        np.asarray(same_variant, dtype=np.float64),
    ]
    Xreg_cols: List[np.ndarray] = [np.ones_like(y)]
    for col in continuous:
        sd = float(col.std())
        Xreg_cols.append((col - float(col.mean())) / (sd if sd > 1e-9 else 1.0))
    Xreg = np.stack(Xreg_cols, axis=1)
    y_std = (y - float(y.mean())) / (float(y.std()) if float(y.std()) > 1e-9 else 1.0)
    beta = np.linalg.lstsq(Xreg, y_std, rcond=None)[0]
    return {
        "layer": int(layer),
        "n_pairs": int(len(y)),
        "corr_distance_abs_delta_rho": _pearson(dists, drho),
        "corr_distance_abs_delta_log_value": _pearson(dists, dvalue),
        "corr_distance_abs_delta_log_standard": _pearson(dists, dstandard),
        "mean_distance_same_rho": float(np.mean(same_rho_dists)) if same_rho_dists else float("nan"),
        "mean_distance_same_value": float(np.mean(same_value_dists)) if same_value_dists else float("nan"),
        "mean_distance_same_standard": float(np.mean(same_standard_dists)) if same_standard_dists else float("nan"),
        "mean_distance_all": float(np.mean(y)) if len(y) else float("nan"),
        "std_beta_abs_delta_rho": float(beta[1]),
        "std_beta_abs_delta_log_value": float(beta[2]),
        "std_beta_abs_delta_log_standard": float(beta[3]),
        "std_beta_same_context": float(beta[4]),
        "std_beta_same_variant": float(beta[5]),
    }


def _delta_alignment(X: np.ndarray, rows: Sequence[Mapping[str, Any]], *, layer: int) -> Dict[str, Any]:
    Z = _standardize_full(X)
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_pair[str(row.get("pair_id", ""))].append(i)
    deltas: List[np.ndarray] = []
    for idxs in by_pair.values():
        if len(idxs) != 2:
            continue
        i, j = idxs
        rho_i = _to_float(rows[i].get("rho"))
        rho_j = _to_float(rows[j].get("rho"))
        if not (_is_finite(rho_i) and _is_finite(rho_j)) or rho_i == rho_j:
            continue
        high, low = (i, j) if rho_i > rho_j else (j, i)
        delta = Z[high] - Z[low]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            deltas.append(delta / norm)
    if len(deltas) < 2:
        return {
            "layer": int(layer),
            "n_deltas": int(len(deltas)),
            "mean_pairwise_delta_cosine": float("nan"),
            "projection_corr_with_rho": float("nan"),
        }
    D = np.stack(deltas, axis=0)
    sims = D @ D.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    mean_dir = D.mean(axis=0)
    norm = float(np.linalg.norm(mean_dir))
    if norm > 1e-9:
        mean_dir = mean_dir / norm
        proj = Z @ mean_dir
        rho = [_to_float(row.get("rho")) for row in rows]
        proj_corr = _pearson(proj.tolist(), rho)
    else:
        proj_corr = float("nan")
    return {
        "layer": int(layer),
        "n_deltas": int(len(deltas)),
        "mean_pairwise_delta_cosine": float(np.mean(tri)),
        "projection_corr_with_rho": proj_corr,
    }


def analyze_layer(
    X: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    alpha: float,
    pca_ranks: Sequence[int],
) -> Dict[str, Any]:
    targets = {
        "rho": np.asarray([_to_float(row.get("rho")) for row in rows], dtype=np.float64),
        "ordered_score": np.asarray([_to_float(row.get("ordered_score")) for row in rows], dtype=np.float64),
        "signed_score": np.asarray([_to_float(row.get("signed_score")) for row in rows], dtype=np.float64),
    }
    variants = sorted({str(row.get("variant", "")) for row in rows})
    fictional_variant = next((v for v in variants if v.startswith("fictional_semantic")), "")
    iso_variant = next((v for v in variants if v.startswith("iso_ratio")), "")
    metrics: List[Dict[str, Any]] = []
    fictional_filter = np.asarray([str(row.get("variant", "")) == fictional_variant for row in rows], dtype=bool)
    for target, y in targets.items():
        valid_idx = _valid_target_indices(y)
        if len(valid_idx) >= 8:
            train_idx = valid_idx[::2]
            test_idx = np.asarray([i for i in valid_idx if i not in set(train_idx.tolist())], dtype=np.int64)
            if len(train_idx) >= 4 and len(test_idx) >= 3:
                pred = _ridge_dual_predict(X, y, train_idx, test_idx, alpha=alpha)
                metrics.append(_metric_row("hash_even_odd", target, y, pred, test_idx, layer=layer, model="ridge_full"))
                for rank in pca_ranks:
                    pred_rank = _pca_ridge_predict(X, y, train_idx, test_idx, rank=int(rank), alpha=alpha)
                    metrics.append(
                        _metric_row(
                            "hash_even_odd",
                            target,
                            y,
                            pred_rank,
                            test_idx,
                            layer=layer,
                            model=f"pca{int(rank)}_ridge",
                        )
                    )
        if fictional_variant:
            for group_key in ("comparison", "value", "standard"):
                metrics.extend(
                    _crossfit_group_split(
                        X,
                        y,
                        rows,
                        group_key=group_key,
                        row_filter=fictional_filter,
                        layer=layer,
                        target=target,
                        alpha=alpha,
                        pca_ranks=pca_ranks,
                    )
                )
        if fictional_variant and iso_variant:
            metrics.extend(
                _cross_variant_split(
                    X,
                    y,
                    rows,
                    train_variant=fictional_variant,
                    test_variant=iso_variant,
                    layer=layer,
                    target=target,
                    alpha=alpha,
                    pca_ranks=pca_ranks,
                )
            )
            metrics.extend(
                _cross_variant_split(
                    X,
                    y,
                    rows,
                    train_variant=iso_variant,
                    test_variant=fictional_variant,
                    layer=layer,
                    target=target,
                    alpha=alpha,
                    pca_ranks=pca_ranks,
                )
            )
    return {
        "layer": int(layer),
        "metrics": metrics,
        "distance": _distance_diagnostics(X, rows, layer=layer),
        "delta_alignment": _delta_alignment(X, rows, layer=layer),
    }


def _format_float(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Gradable Size Activation Geometry",
        "",
        f"- Activation NPZ: `{summary['inputs']['npz']}`",
        f"- Metadata CSV: `{summary['inputs']['metadata_csv']}`",
        f"- Rows: `{summary['counts']['n_rows']}`",
        f"- Layers: `{', '.join(str(x) for x in summary['config']['layers'])}`",
        "",
        "## Cross-Variant Ridge",
        "",
        "| layer | split | target | model | n | r | r2 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for layer in summary["layers"].values():
        for row in layer["metrics"]:
            if "_to_" not in str(row["split"]):
                continue
            if str(row["target"]) not in {"rho", "ordered_score", "signed_score"}:
                continue
            if str(row["model"]) not in {"ridge_full", "pca1_ridge", "pca2_ridge", "pca5_ridge"}:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["layer"]),
                        str(row["split"]),
                        str(row["target"]),
                        str(row["model"]),
                        str(row["n_test"]),
                        _format_float(row["r"]),
                        _format_float(row["r2"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Distance Diagnostics",
            "",
            "| layer | corr dist~|delta rho| | beta rho | beta value | beta standard | same-rho dist | all dist |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in summary["layers"].values():
        row = layer["distance"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer"]),
                    _format_float(row["corr_distance_abs_delta_rho"]),
                    _format_float(row["std_beta_abs_delta_rho"]),
                    _format_float(row["std_beta_abs_delta_log_value"]),
                    _format_float(row["std_beta_abs_delta_log_standard"]),
                    _format_float(row["mean_distance_same_rho"]),
                    _format_float(row["mean_distance_all"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Delta Alignment",
            "",
            "| layer | n deltas | mean pairwise cosine | projection r with rho |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in summary["layers"].values():
        row = layer["delta_alignment"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["layer"]),
                    str(row["n_deltas"]),
                    _format_float(row["mean_pairwise_delta_cosine"]),
                    _format_float(row["projection_corr_with_rho"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation discipline:",
            "- This is an activation-geometry gate, not an SAE/CLT group result.",
            "- Treat L20/L24 as the primary mechanistic layers; L32/L33 are readout-near sanity endpoints.",
            "- A manifold-style claim requires held-out low-rank structure plus causal low-rank patching, not only a positive PCA/Ridge table.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze gradable-size activation geometry.")
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--layers", type=str, default="20,24,28,32,33")
    p.add_argument("--pca_ranks", type=str, default="1,2,3,4,5")
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--out_json", type=str, default=str(DEFAULT_ROOT / "gradable_size_geometry_late_final_token_gemma3.analysis.json"))
    p.add_argument("--out_md", type=str, default=str(DEFAULT_ROOT / "gradable_size_geometry_late_final_token_gemma3.analysis.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_csv(Path(str(args.metadata_csv)))
    layers = _parse_int_list(str(args.layers))
    pca_ranks = _parse_int_list(str(args.pca_ranks))
    arrs = np.load(Path(str(args.npz)))
    layer_summaries: Dict[str, Any] = {}
    for layer in layers:
        key = f"X_layer_{int(layer)}"
        if key not in arrs:
            raise KeyError(f"Missing {key} in {args.npz}")
        X = np.asarray(arrs[key], dtype=np.float32)
        if X.shape[0] != len(rows):
            raise ValueError(f"{key} has {X.shape[0]} rows, metadata has {len(rows)} rows")
        layer_summaries[str(layer)] = analyze_layer(
            X,
            rows,
            layer=int(layer),
            alpha=float(args.ridge_alpha),
            pca_ranks=pca_ranks,
        )
    summary = {
        "schema_version": "gradable_size_geometry_analysis_v1",
        "inputs": {"npz": str(args.npz), "metadata_csv": str(args.metadata_csv)},
        "counts": {
            "n_rows": int(len(rows)),
            "variants": sorted({str(row.get("variant", "")) for row in rows}),
        },
        "config": {
            "layers": [int(x) for x in layers],
            "pca_ranks": [int(x) for x in pca_ranks],
            "ridge_alpha": float(args.ridge_alpha),
        },
        "layers": layer_summaries,
    }
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Wrote geometry analysis JSON: {out_json}")
    print(f"Wrote geometry analysis Markdown: {out_md}")


if __name__ == "__main__":
    main()
