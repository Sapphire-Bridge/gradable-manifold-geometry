from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"

FICTIONAL = "fictional_semantic_adjective_counts"
ISO = "iso_ratio_adjective_counts"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def _to_float(raw: Any, default: float = float("nan")) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if _is_finite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(xs: Iterable[float]) -> float:
    vals = sorted(float(x) for x in xs if _is_finite(float(x)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2.0)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite(float(x)) and _is_finite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0.0 else float("nan")


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, None]
    keep: List[np.ndarray] = []
    for j in range(mat.shape[1]):
        col = mat[:, j].copy()
        for prev in keep:
            col = col - prev * float(np.dot(prev, col))
        norm = float(np.linalg.norm(col))
        if norm > 1e-9:
            keep.append(col / norm)
    if not keep:
        raise ValueError("Could not build nonzero orthonormal basis")
    return np.stack(keep, axis=1)


def _pca_basis_and_scores(X: np.ndarray, *, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = _orthonormalize(vt[: int(rank)].T)
    scores = Xc @ basis
    denom = max(float(np.sum(s**2)), 1e-12)
    explained = (s[: int(rank)] ** 2) / denom
    return basis, scores, explained


def _subspace_alignment(U: np.ndarray, V: np.ndarray) -> Dict[str, Any]:
    s = np.linalg.svd(np.asarray(U, dtype=np.float64).T @ np.asarray(V, dtype=np.float64), compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = [float(math.degrees(math.acos(float(x)))) for x in s]
    denom = max(int(min(U.shape[1], V.shape[1])), 1)
    return {
        "mean_squared_cosine": float(np.sum(s**2) / float(denom)),
        "singular_values": [float(x) for x in s.tolist()],
        "principal_angles_deg": angles,
        "max_principal_angle_deg": max(angles) if angles else float("nan"),
        "mean_principal_angle_deg": _mean(angles),
    }


def _variant_indices(rows: Sequence[Mapping[str, Any]], variant: str) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(rows) if str(row.get("variant", "")) == str(variant)], dtype=np.int64)


def compute_subspace_overlap(
    *,
    arrs: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    rank: int,
) -> List[Dict[str, Any]]:
    idx_a = _variant_indices(rows, FICTIONAL)
    idx_b = _variant_indices(rows, ISO)
    out: List[Dict[str, Any]] = []
    for layer in layers:
        X = np.asarray(arrs[f"X_layer_{int(layer)}"], dtype=np.float32)
        Ua, _, exp_a = _pca_basis_and_scores(X[idx_a], rank=int(rank))
        Ub, _, exp_b = _pca_basis_and_scores(X[idx_b], rank=int(rank))
        align = _subspace_alignment(Ua, Ub)
        row: Dict[str, Any] = {
            "layer": int(layer),
            "rank": int(rank),
            "variant_a": FICTIONAL,
            "variant_b": ISO,
            "n_a": int(len(idx_a)),
            "n_b": int(len(idx_b)),
            "mean_squared_cosine": float(align["mean_squared_cosine"]),
            "mean_principal_angle_deg": float(align["mean_principal_angle_deg"]),
            "max_principal_angle_deg": float(align["max_principal_angle_deg"]),
            "explained_a_rank_sum": float(np.sum(exp_a)),
            "explained_b_rank_sum": float(np.sum(exp_b)),
            "random_overlap_expectation": float(int(rank) / float(X.shape[1])),
        }
        for i, angle in enumerate(align["principal_angles_deg"], start=1):
            row[f"angle_{i}_deg"] = float(angle)
        for i, sv in enumerate(align["singular_values"], start=1):
            row[f"singular_value_{i}"] = float(sv)
        out.append(row)
    return out


def compute_projection(
    *,
    arrs: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    rank: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    X = np.asarray(arrs[f"X_layer_{int(layer)}"], dtype=np.float32)
    _, scores, explained = _pca_basis_and_scores(X, rank=int(rank))
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = {
            "row_idx": int(i),
            "layer": int(layer),
            "variant": str(row.get("variant", "")),
            "pair_id": str(row.get("pair_id", "")),
            "side": str(row.get("side", "")),
            "rho": _to_float(row.get("rho")),
            "value": _to_float(row.get("value")),
            "standard": _to_float(row.get("standard")),
            "ordered_score": _to_float(row.get("ordered_score")),
        }
        for j in range(int(rank)):
            item[f"pc{j + 1}"] = float(scores[i, j])
        out.append(item)
    summary = {
        "layer": int(layer),
        "rank": int(rank),
        "explained_variance": [float(x) for x in explained.tolist()],
        "pc1_rho_r": _pearson([r["pc1"] for r in out], [r["rho"] for r in out]),
        "pc2_rho_r": _pearson([r["pc2"] for r in out], [r["rho"] for r in out]),
        "pc3_rho_r": _pearson([r["pc3"] for r in out], [r["rho"] for r in out]),
    }
    return out, summary


def _patch_rows(path: Path, *, direction_label: str) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("method", "")) != "pca":
            continue
        if int(float(row.get("rank", 0))) != 5:
            continue
        if int(float(row.get("layer", 0))) != 20:
            continue
        if abs(_to_float(row.get("alpha")) - 1.0) > 1e-9:
            continue
        if int(float(row.get("repeat", 0))) != 0:
            continue
        item = dict(row)
        item["transfer_direction"] = direction_label
        out.append(item)
    return out


def compute_transfer_by_delta(
    *,
    patch_csvs: Sequence[Tuple[Path, str]],
    n_bins: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for path, label in patch_csvs:
        rows.extend(_patch_rows(path, direction_label=label))
    vals = [abs(_to_float(row.get("predictor_delta"))) for row in rows if _is_finite(abs(_to_float(row.get("predictor_delta"))))]
    if not vals:
        return [], []
    qs = np.quantile(np.asarray(vals, dtype=np.float64), np.linspace(0.0, 1.0, int(n_bins) + 1))
    detailed: List[Dict[str, Any]] = []
    for row in rows:
        abs_delta = abs(_to_float(row.get("predictor_delta")))
        bin_idx = int(np.searchsorted(qs[1:-1], abs_delta, side="right"))
        detailed.append(
            {
                "transfer_direction": str(row.get("transfer_direction", "")),
                "pair_id": str(row.get("pair_id", "")),
                "direction": str(row.get("direction", "")),
                "abs_delta_rho": float(abs_delta),
                "bin": int(bin_idx + 1),
                "bin_lo": float(qs[bin_idx]),
                "bin_hi": float(qs[bin_idx + 1]),
                "aligned_effect": _to_float(row.get("aligned_effect")),
                "recovery_vs_full": _to_float(row.get("recovery_vs_full")),
                "full_aligned_effect": _to_float(row.get("full_aligned_effect")),
                "patch_fraction": _to_float(row.get("patch_fraction")),
                "direction_match": int(float(row.get("direction_match", 0))),
            }
        )
    summary: List[Dict[str, Any]] = []
    for bin_idx in range(1, int(n_bins) + 1):
        bin_rows = [r for r in detailed if int(r["bin"]) == bin_idx]
        summary.append(
            {
                "bin": int(bin_idx),
                "n": int(len(bin_rows)),
                "bin_lo": float(qs[bin_idx - 1]),
                "bin_hi": float(qs[bin_idx]),
                "mean_abs_delta_rho": _mean(r["abs_delta_rho"] for r in bin_rows),
                "mean_aligned_effect": _mean(r["aligned_effect"] for r in bin_rows),
                "mean_recovery_vs_full": _mean(r["recovery_vs_full"] for r in bin_rows),
                "median_recovery_vs_full": _median(r["recovery_vs_full"] for r in bin_rows),
                "mean_patch_fraction": _mean(r["patch_fraction"] for r in bin_rows),
                "direction_match_rate": _mean(r["direction_match"] for r in bin_rows),
            }
        )
    return detailed, summary


def compute_layer_trajectory(
    *,
    arrs: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    patch_csvs: Sequence[Tuple[Path, str]],
) -> List[Dict[str, Any]]:
    rho = [_to_float(row.get("rho")) for row in rows]
    by_layer: Dict[int, Dict[str, Any]] = {}
    for layer in layers:
        X = np.asarray(arrs[f"X_layer_{int(layer)}"], dtype=np.float32)
        _, scores, explained = _pca_basis_and_scores(X, rank=5)
        by_layer[int(layer)] = {
            "layer": int(layer),
            "pc1_rho_r": _pearson(scores[:, 0].tolist(), rho),
            "pc2_rho_r": _pearson(scores[:, 1].tolist(), rho),
            "pc3_rho_r": _pearson(scores[:, 2].tolist(), rho),
            "max_abs_pc1_to_pc5_rho_r": max(abs(_pearson(scores[:, j].tolist(), rho)) for j in range(5)),
            "rank5_explained_variance": float(np.sum(explained)),
            "pca5_recovery_fictional_to_iso": float("nan"),
            "pca5_recovery_iso_to_fictional": float("nan"),
            "pca5_aligned_fictional_to_iso": float("nan"),
            "pca5_aligned_iso_to_fictional": float("nan"),
        }
    patch_values: Dict[Tuple[int, str], Dict[str, List[float]]] = {}
    for path, label in patch_csvs:
        for row in _read_csv(path):
            if str(row.get("method", "")) != "pca" or int(float(row.get("rank", 0))) != 5:
                continue
            if abs(_to_float(row.get("alpha")) - 1.0) > 1e-9:
                continue
            if int(float(row.get("repeat", 0))) != 0:
                continue
            layer = int(float(row.get("layer", 0)))
            if layer not in by_layer:
                continue
            vals = patch_values.setdefault((layer, label), {"recovery": [], "aligned": []})
            vals["recovery"].append(_to_float(row.get("recovery_vs_full")))
            vals["aligned"].append(_to_float(row.get("aligned_effect")))
    for (layer, label), vals in patch_values.items():
        if label == "fictional_to_iso":
            by_layer[layer]["pca5_recovery_fictional_to_iso"] = _mean(vals["recovery"])
            by_layer[layer]["pca5_aligned_fictional_to_iso"] = _mean(vals["aligned"])
        elif label == "iso_to_fictional":
            by_layer[layer]["pca5_recovery_iso_to_fictional"] = _mean(vals["recovery"])
            by_layer[layer]["pca5_aligned_iso_to_fictional"] = _mean(vals["aligned"])
    for item in by_layer.values():
        item["pca5_recovery_mean_available"] = _mean(
            [item["pca5_recovery_fictional_to_iso"], item["pca5_recovery_iso_to_fictional"]]
        )
        item["pca5_aligned_mean_available"] = _mean(
            [item["pca5_aligned_fictional_to_iso"], item["pca5_aligned_iso_to_fictional"]]
        )
    return [by_layer[int(layer)] for layer in layers]


def plot_subspace_overlap(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    layers = [int(r["layer"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for i in range(1, 6):
        axes[0].plot(layers, [float(r[f"angle_{i}_deg"]) for r in rows], marker="o", label=f"angle {i}")
    axes[0].set_title("Cross-variant PCA basis angles")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Principal angle (deg)")
    axes[0].set_ylim(0, 92)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].axvline(20, color="#64748b", linestyle="--", linewidth=1)
    axes[1].plot(layers, [float(r["mean_squared_cosine"]) for r in rows], marker="o", color="#2563eb")
    axes[1].plot(layers, [float(r["random_overlap_expectation"]) for r in rows], linestyle="--", color="#94a3b8", label="random expectation")
    axes[1].set_title("Projection overlap")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("mean squared cosine")
    axes[1].set_ylim(0, max(0.45, max(float(r["mean_squared_cosine"]) for r in rows) * 1.15))
    axes[1].axvline(20, color="#64748b", linestyle="--", linewidth=1)
    axes[1].legend(frameon=False)
    fig.suptitle("Analysis 1: independently fitted size bases are compatible but not identical", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    _strip_trailing_whitespace(path)


def plot_projection(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    variants = [(FICTIONAL, "o", "fictional"), (ISO, "^", "iso")]
    all_rho = np.asarray([float(r["rho"]) for r in rows], dtype=np.float64)
    vmin, vmax = float(np.nanmin(all_rho)), float(np.nanmax(all_rho))
    scatter = None
    for variant, marker, label in variants:
        subset = [r for r in rows if str(r["variant"]) == variant]
        scatter = ax.scatter(
            [float(r["pc1"]) for r in subset],
            [float(r["pc2"]) for r in subset],
            c=[float(r["rho"]) for r in subset],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            s=54,
            label=label,
        )
    sorted_rows = sorted(rows, key=lambda r: float(r["rho"]))
    bins = np.array_split(sorted_rows, 12)
    curve_x = [_median(float(r["pc1"]) for r in b) for b in bins if len(b)]
    curve_y = [_median(float(r["pc2"]) for r in b) for b in bins if len(b)]
    ax.plot(curve_x, curve_y, color="#111827", linewidth=1.8, alpha=0.75, label="rho-binned median path")
    ax.set_title("Analysis 2: L20 PCA projection colored by log(value/standard)", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("rho = log(value/standard)")
    ax.text(
        0.02,
        0.02,
        f"r(PC1,rho)={float(summary['pc1_rho_r']):.3f}; r(PC2,rho)={float(summary['pc2_rho_r']):.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    _strip_trailing_whitespace(path)


def plot_transfer_by_delta(detailed: Sequence[Mapping[str, Any]], summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {"fictional_to_iso": "#2563eb", "iso_to_fictional": "#dc2626"}
    for label, color in colors.items():
        subset = [r for r in detailed if str(r["transfer_direction"]) == label]
        axes[0].scatter(
            [float(r["abs_delta_rho"]) for r in subset],
            [float(r["recovery_vs_full"]) for r in subset],
            s=34,
            alpha=0.72,
            label=label,
            color=color,
        )
    axes[0].axhline(0.0, color="#94a3b8", linewidth=1)
    axes[0].set_xlabel("|delta rho|")
    axes[0].set_ylabel("pca-k5 recovery/full")
    axes[0].set_title("Pair-level transfer efficiency")
    axes[0].legend(frameon=False)
    axes[1].plot(
        [float(r["mean_abs_delta_rho"]) for r in summary],
        [float(r["mean_recovery_vs_full"]) for r in summary],
        marker="o",
        color="#111827",
        label="mean recovery",
    )
    axes[1].plot(
        [float(r["mean_abs_delta_rho"]) for r in summary],
        [float(r["mean_aligned_effect"]) for r in summary],
        marker="s",
        color="#047857",
        label="mean aligned effect",
    )
    axes[1].axhline(0.0, color="#94a3b8", linewidth=1)
    axes[1].set_xlabel("mean |delta rho| bin")
    axes[1].set_title("Binned by |delta rho|")
    axes[1].legend(frameon=False)
    fig.suptitle("Analysis 3: transfer as a function of scalar distance", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    _strip_trailing_whitespace(path)


def plot_layer_trajectory(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    layers = [int(r["layer"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(layers, [abs(float(r["pc1_rho_r"])) for r in rows], marker="o", color="#2563eb", label="|r(PC1,rho)|")
    axes[0].plot(
        layers,
        [float(r["max_abs_pc1_to_pc5_rho_r"]) for r in rows],
        marker="s",
        color="#64748b",
        label="max |r(PC1..PC5,rho)|",
    )
    axes[0].set_ylabel("rho alignment")
    axes[0].set_title("Activation geometry")
    axes[0].legend(frameon=False)
    axes[0].axvline(20, color="#64748b", linestyle="--", linewidth=1)
    available = [r for r in rows if _is_finite(float(r["pca5_aligned_mean_available"]))]
    axes[1].plot(
        [int(r["layer"]) for r in available],
        [float(r["pca5_aligned_fictional_to_iso"]) for r in available],
        marker="o",
        color="#2563eb",
        label="fictional -> iso aligned effect",
    )
    axes[1].plot(
        [int(r["layer"]) for r in available],
        [float(r["pca5_aligned_iso_to_fictional"]) for r in available],
        marker="o",
        color="#dc2626",
        label="iso -> fictional aligned effect",
    )
    axes[1].axhline(0.0, color="#94a3b8", linewidth=1)
    axes[1].axvline(20, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("pca-k5 aligned effect")
    axes[1].set_title("Causal patching profile (available low-rank patch layers only)")
    axes[1].legend(frameon=False)
    fig.suptitle("Analysis 4: layer trajectory of geometry and causal structure", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    _strip_trailing_whitespace(path)


def write_markdown(
    *,
    out_path: Path,
    overlap_rows: Sequence[Mapping[str, Any]],
    projection_summary: Mapping[str, Any],
    transfer_summary: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    figure_paths: Mapping[str, Path],
) -> None:
    l20 = next(r for r in overlap_rows if int(r["layer"]) == 20)
    l20_angles = [float(l20[f"angle_{i}_deg"]) for i in range(1, 6)]
    best_overlap = max(overlap_rows, key=lambda r: float(r["mean_squared_cosine"]))
    l20_layer = next(r for r in layer_rows if int(r["layer"]) == 20)
    lines = [
        "# Gradable Manifold-Claim Diagnostics",
        "",
        "These are visual and tabular diagnostics for the current Manifold-Groups paper claim.",
        "They use existing activation caches and existing low-rank patch CSVs; no new model patching run is performed.",
        "",
        "## Headline",
        "",
        "- Cross-variant PCA bases are not literally the same L20 subspace: L20 principal angles are "
        + ", ".join(f"{x:.1f} deg" for x in l20_angles)
        + f", with overlap {float(l20['mean_squared_cosine']):.3f}.",
        f"- The strongest cross-variant basis overlap is at L{int(best_overlap['layer'])}, overlap {float(best_overlap['mean_squared_cosine']):.3f}; overlap falls by L20.",
        f"- L20 combined PCA has r(PC1,rho)={float(projection_summary['pc1_rho_r']):.3f}, r(PC2,rho)={float(projection_summary['pc2_rho_r']):.3f}, r(PC3,rho)={float(projection_summary['pc3_rho_r']):.3f}.",
        f"- L20 remains the causal peak among available low-rank patch layers: mean bidirectional pca-k5 aligned effect {float(l20_layer['pca5_aligned_mean_available']):.3f}.",
        "",
        "Interpretation: the diagnostics support a behaviorally causal, low-rank size-calibration geometry, but not the stronger claim that independently fitted fictional and iso bases are literally the same rank-5 subspace at L20.",
        "",
        "## Figures",
        "",
    ]
    for label, path in figure_paths.items():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        lines.append(f"- {label}: `{display_path}`")
    lines.extend(
        [
            "",
            "## Analysis 1: Cross-variant subspace overlap",
            "",
            "| layer | overlap | angles deg | random overlap |",
            "| ---: | ---: | --- | ---: |",
        ]
    )
    for row in overlap_rows:
        angles = ", ".join(f"{float(row[f'angle_{i}_deg']):.1f}" for i in range(1, 6))
        lines.append(
            f"| {int(row['layer'])} | {float(row['mean_squared_cosine']):.3f} | {angles} | {float(row['random_overlap_expectation']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Analysis 3: Transfer efficiency by |delta rho|",
            "",
            "| bin | n | mean abs delta rho | mean recovery | mean aligned effect | dir match |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in transfer_summary:
        lines.append(
            f"| {int(row['bin'])} | {int(row['n'])} | {float(row['mean_abs_delta_rho']):.3f} | "
            f"{float(row['mean_recovery_vs_full']):.3f} | {float(row['mean_aligned_effect']):.3f} | {float(row['direction_match_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Analysis 4: Layer trajectory",
            "",
            "| layer | abs r(PC1,rho) | max abs r(PC1..5,rho) | pca5 aligned f->i | pca5 aligned i->f | mean aligned |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in layer_rows:
        lines.append(
            f"| {int(row['layer'])} | {abs(float(row['pc1_rho_r'])):.3f} | {float(row['max_abs_pc1_to_pc5_rho_r']):.3f} | "
            f"{float(row['pca5_aligned_fictional_to_iso']):.3f} | {float(row['pca5_aligned_iso_to_fictional']):.3f} | "
            f"{float(row['pca5_aligned_mean_available']):.3f} |"
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manifold-claim diagnostics from gradable size artifacts.")
    p.add_argument("--activations_npz", type=str, default=str(DEFAULT_ROOT / "gradable_size_geometry_broad_final_token_gemma3.npz"))
    p.add_argument(
        "--metadata_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_geometry_broad_final_token_gemma3.metadata.csv"),
    )
    p.add_argument(
        "--forward_patch_csv",
        type=str,
        default=str(
            DEFAULT_ROOT
            / "gradable_size_low_rank_patch_train_fictional_semantic_adjective_counts_eval_iso_ratio_adjective_counts_l162024_normmatched_r20_gemma3.csv"
        ),
    )
    p.add_argument(
        "--reverse_patch_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_low_rank_patch_train_iso_ratio_adjective_counts_eval_fictional_semantic_adjective_counts_l162024_gemma3.csv"),
    )
    p.add_argument("--layers", type=str, default="8,12,16,20,24,28,32,33")
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--projection_layer", type=int, default=20)
    p.add_argument("--out_prefix", type=str, default=str(DEFAULT_ROOT / "gradable_manifold_claim_diagnostics"))
    p.add_argument("--fig_dir", type=str, default=str(ROOT / "figures" / "manifold_groups"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    rows = _read_csv(Path(str(args.metadata_csv)))
    arrs = np.load(Path(str(args.activations_npz)))
    patch_csvs = [
        (Path(str(args.forward_patch_csv)), "fictional_to_iso"),
        (Path(str(args.reverse_patch_csv)), "iso_to_fictional"),
    ]
    out_prefix = Path(str(args.out_prefix))
    fig_dir = Path(str(args.fig_dir))

    overlap_rows = compute_subspace_overlap(arrs=arrs, rows=rows, layers=layers, rank=int(args.rank))
    projection_rows, projection_summary = compute_projection(
        arrs=arrs,
        rows=rows,
        layer=int(args.projection_layer),
        rank=3,
    )
    transfer_detailed, transfer_summary = compute_transfer_by_delta(patch_csvs=patch_csvs, n_bins=4)
    layer_rows = compute_layer_trajectory(arrs=arrs, rows=rows, layers=layers, patch_csvs=patch_csvs)

    _write_csv(overlap_rows, out_prefix.with_suffix(".subspace_overlap.csv"))
    _write_csv(projection_rows, out_prefix.with_suffix(".l20_pca_projection.csv"))
    _write_csv(transfer_detailed, out_prefix.with_suffix(".transfer_by_delta_rho.rows.csv"))
    _write_csv(transfer_summary, out_prefix.with_suffix(".transfer_by_delta_rho.summary.csv"))
    _write_csv(layer_rows, out_prefix.with_suffix(".layer_trajectory.csv"))
    out_prefix.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "activations_npz": str(args.activations_npz),
                "metadata_csv": str(args.metadata_csv),
                "forward_patch_csv": str(args.forward_patch_csv),
                "reverse_patch_csv": str(args.reverse_patch_csv),
                "layers": layers,
                "rank": int(args.rank),
                "projection_layer": int(args.projection_layer),
                "projection_summary": projection_summary,
                "n_projection_rows": int(len(projection_rows)),
                "n_transfer_rows": int(len(transfer_detailed)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    figure_paths = {
        "cross-variant subspace overlap": fig_dir / "cross_variant_subspace_overlap.svg",
        "L20 PCA rho projection": fig_dir / "l20_pca_rho_projection.svg",
        "transfer efficiency by delta rho": fig_dir / "transfer_efficiency_by_delta_rho.svg",
        "layer trajectory": fig_dir / "layer_trajectory.svg",
    }
    plot_subspace_overlap(overlap_rows, figure_paths["cross-variant subspace overlap"])
    plot_projection(projection_rows, projection_summary, figure_paths["L20 PCA rho projection"])
    plot_transfer_by_delta(transfer_detailed, transfer_summary, figure_paths["transfer efficiency by delta rho"])
    plot_layer_trajectory(layer_rows, figure_paths["layer trajectory"])
    write_markdown(
        out_path=out_prefix.with_suffix(".md"),
        overlap_rows=overlap_rows,
        projection_summary=projection_summary,
        transfer_summary=transfer_summary,
        layer_rows=layer_rows,
        figure_paths=figure_paths,
    )
    print(f"Wrote manifold diagnostics prefix: {out_prefix}")
    print(f"Wrote figures under: {fig_dir}")


if __name__ == "__main__":
    main()
