from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.data.schemas import DisambPair, PromptSide
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.metrics.disamb import _encode_prompt, score_labels_next_continuations, score_labels_next_continuations_patched
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed
from scripts.patch_gradable_size_low_rank import (
    DEFAULT_ROOT,
    LABEL_ORDER,
    _assert_eval_alignment,
    _cluster_bootstrap_mean_ci,
    _delta_mean_direction,
    _is_finite,
    _mean,
    _ordered_score,
    _orthonormalize,
    _parse_csv_or_space,
    _parse_float_list,
    _parse_int_list,
    _pca_basis,
    _quantile,
    _random_basis,
    _read_csv,
    _softmax,
    _supervised_direction,
    _target_vector,
    _to_float,
    _write_csv,
)


def _sign(x: float) -> float:
    if float(x) > 0.0:
        return 1.0
    if float(x) < 0.0:
        return -1.0
    return 0.0


def _side(item: DisambPair, side_name: str) -> PromptSide:
    if str(side_name) == "a":
        return item.a
    if str(side_name) == "b":
        return item.b
    raise ValueError(f"Unknown side {side_name!r}")


def _ratio_for_side(item: DisambPair, side_name: str) -> float:
    md = dict(item.metadata or {})
    raw = md.get(f"ratio_{side_name}")
    if raw is None:
        raise ValueError(f"{item.pair_id}: missing ratio_{side_name} metadata")
    return float(raw)


def _standard_for_side(item: DisambPair, side_name: str) -> float:
    md = dict(item.metadata or {})
    raw = md.get(f"standard_{side_name}")
    if raw is None:
        return float("nan")
    return float(raw)


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize near-zero steering direction")
    return (out / norm).astype(np.float32)


def _project_np(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    U = np.asarray(basis, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    return (U @ (U.T @ v)).astype(np.float32)


def _pair_deltas_high_minus_low(X: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> List[np.ndarray]:
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_pair[str(row.get("pair_id", ""))].append(int(idx))
    deltas: List[np.ndarray] = []
    for idxs in by_pair.values():
        if len(idxs) != 2:
            continue
        i, j = idxs
        rho_i = _to_float(rows[i].get("rho"))
        rho_j = _to_float(rows[j].get("rho"))
        if not (_is_finite(rho_i) and _is_finite(rho_j)) or abs(rho_i - rho_j) <= 1e-12:
            continue
        high, low = (i, j) if rho_i > rho_j else (j, i)
        delta = np.asarray(X[high] - X[low], dtype=np.float64)
        if float(np.linalg.norm(delta)) > 1e-9:
            deltas.append(delta.astype(np.float32))
    if not deltas:
        raise ValueError("No valid high-minus-low training pair deltas")
    return deltas


def _median_positive(xs: Iterable[float]) -> float:
    vals = sorted(float(x) for x in xs if _is_finite(float(x)) and float(x) > 1e-9)
    if not vals:
        return float("nan")
    return _quantile(vals, 0.5)


def _typical_pair_delta_scale(X: np.ndarray, rows: Sequence[Mapping[str, Any]], unit: np.ndarray) -> float:
    deltas = _pair_deltas_high_minus_low(X, rows)
    u = _unit_vector(unit)
    projected = _median_positive(abs(float(np.dot(delta, u))) for delta in deltas)
    if _is_finite(projected) and projected > 1e-9:
        return float(projected)
    fallback = _median_positive(float(np.linalg.norm(delta)) for delta in deltas)
    return float(fallback if _is_finite(fallback) and fallback > 1e-9 else 1.0)


def _direction_record(
    *,
    method: str,
    unit: np.ndarray,
    X_train: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    rank: int,
    repeat: int,
    layer: int,
    control_family: str,
    direction_source: str,
    basis_projection_norm: float,
    steer_scale: float | None = None,
) -> Dict[str, Any]:
    unit_vec = _unit_vector(unit)
    scale = (
        float(steer_scale)
        if steer_scale is not None
        else _typical_pair_delta_scale(X_train, train_rows, unit_vec)
    )
    return {
        "method": str(method),
        "layer": int(layer),
        "rank": int(rank),
        "repeat": int(repeat),
        "unit_direction": unit_vec,
        "direction_unit_norm": float(np.linalg.norm(unit_vec)),
        "steer_scale": float(scale),
        "control_family": str(control_family),
        "direction_source": str(direction_source),
        "basis_projection_norm": float(basis_projection_norm),
    }


def build_steering_directions_for_layer(
    *,
    X_train: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    layer: int,
    methods: Sequence[str],
    rank: int,
    ridge_alpha: float,
    seed: int,
    random_repeats: int,
) -> List[Dict[str, Any]]:
    """Build fixed steering directions from train activations only.

    Positive alpha is oriented toward higher rho: high standard-relative size
    minus low standard-relative size.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    delta_unit = _unit_vector(_delta_mean_direction(X_train, train_rows)[:, 0])
    pca_basis = _pca_basis(X_train, int(rank))
    pca_projected = _project_np(delta_unit, pca_basis)
    pca_unit = _unit_vector(pca_projected)
    primary_scale = _typical_pair_delta_scale(X_train, train_rows, pca_unit)

    out: List[Dict[str, Any]] = []
    for method_raw in methods:
        method = str(method_raw)
        if method == "pca_delta_mean":
            out.append(
                _direction_record(
                    method=method,
                    unit=pca_unit,
                    X_train=X_train,
                    train_rows=train_rows,
                    rank=int(rank),
                    repeat=0,
                    layer=int(layer),
                    control_family="primary",
                    direction_source="delta_mean_projected_into_train_pca",
                    basis_projection_norm=float(np.linalg.norm(pca_projected)),
                    steer_scale=primary_scale,
                )
            )
        elif method == "delta_mean":
            out.append(
                _direction_record(
                    method=method,
                    unit=delta_unit,
                    X_train=X_train,
                    train_rows=train_rows,
                    rank=1,
                    repeat=0,
                    layer=int(layer),
                    control_family="primary_1d",
                    direction_source="mean_high_rho_minus_low_rho_pair_delta",
                    basis_projection_norm=1.0,
                    steer_scale=primary_scale,
                )
            )
        elif method in {"rho", "ordered_score", "signed_score", "value", "standard"}:
            target = {"value": "log_value", "standard": "log_standard"}.get(method, method)
            unit = _supervised_direction(
                X_train,
                _target_vector(train_rows, target),
                ridge_alpha=float(ridge_alpha),
            )[:, 0]
            out.append(
                _direction_record(
                    method=method,
                    unit=unit,
                    X_train=X_train,
                    train_rows=train_rows,
                    rank=1,
                    repeat=0,
                    layer=int(layer),
                    control_family="semantic_control" if method in {"value", "standard"} else "supervised_control",
                    direction_source=f"ridge_direction_for_{target}",
                    basis_projection_norm=1.0,
                    steer_scale=primary_scale,
                )
            )
        elif method == "value_standard_2d":
            value_basis = _supervised_direction(
                X_train,
                _target_vector(train_rows, "log_value"),
                ridge_alpha=float(ridge_alpha),
            )
            standard_basis = _supervised_direction(
                X_train,
                _target_vector(train_rows, "log_standard"),
                ridge_alpha=float(ridge_alpha),
            )
            basis = _orthonormalize(np.concatenate([value_basis, standard_basis], axis=1))
            projected = _project_np(delta_unit, basis)
            out.append(
                _direction_record(
                    method=method,
                    unit=projected,
                    X_train=X_train,
                    train_rows=train_rows,
                    rank=int(basis.shape[1]),
                    repeat=0,
                    layer=int(layer),
                    control_family="semantic_control",
                    direction_source="delta_mean_projected_into_value_standard_plane",
                    basis_projection_norm=float(np.linalg.norm(projected)),
                    steer_scale=primary_scale,
                )
            )
        elif method == "random_norm_matched":
            for repeat in range(int(random_repeats)):
                basis = _random_basis(
                    int(X_train.shape[1]),
                    int(rank),
                    seed=int(seed) + int(layer) * 7919 + int(repeat) * 10007,
                )
                projected = _project_np(delta_unit, basis)
                out.append(
                    _direction_record(
                        method=method,
                        unit=projected,
                        X_train=X_train,
                        train_rows=train_rows,
                        rank=int(rank),
                        repeat=int(repeat),
                        layer=int(layer),
                        control_family="random_control",
                        direction_source="delta_mean_projected_into_random_basis",
                        basis_projection_norm=float(np.linalg.norm(projected)),
                        steer_scale=primary_scale,
                    )
                )
        elif method in {"reverse", "pca_delta_mean_reverse"}:
            out.append(
                _direction_record(
                    method="pca_delta_mean_reverse" if method == "reverse" else method,
                    unit=-pca_unit,
                    X_train=X_train,
                    train_rows=train_rows,
                    rank=int(rank),
                    repeat=0,
                    layer=int(layer),
                    control_family="orientation_control",
                    direction_source="negative_delta_mean_projected_into_train_pca",
                    basis_projection_norm=float(np.linalg.norm(pca_projected)),
                    steer_scale=primary_scale,
                )
            )
        elif method == "sham":
            out.append(
                {
                    "method": "sham",
                    "layer": int(layer),
                    "rank": 0,
                    "repeat": 0,
                    "unit_direction": np.zeros(int(X_train.shape[1]), dtype=np.float32),
                    "direction_unit_norm": 0.0,
                    "steer_scale": 0.0,
                    "control_family": "sham",
                    "direction_source": "zero_vector",
                    "basis_projection_norm": 0.0,
                }
            )
        else:
            raise ValueError(f"Unknown steering method {method!r}")
    return out


def _training_slice(
    *,
    arrs: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    train_variant: str,
    layer: int,
) -> Tuple[np.ndarray, List[Mapping[str, Any]]]:
    key = f"X_layer_{int(layer)}"
    if key not in arrs:
        raise KeyError(f"Missing {key} in activation NPZ")
    X = np.asarray(arrs[key], dtype=np.float32)
    if X.shape[0] != len(metadata_rows):
        raise ValueError(f"{key} has {X.shape[0]} rows; metadata has {len(metadata_rows)}")
    train_idx = [i for i, row in enumerate(metadata_rows) if str(row.get("variant", "")) == str(train_variant)]
    if len(train_idx) < 4:
        raise ValueError(f"Not enough train rows for variant {train_variant!r}")
    return X[np.asarray(train_idx, dtype=np.int64)], [metadata_rows[i] for i in train_idx]


def validate_steering_inputs(
    *,
    items: Sequence[DisambPair],
    metadata_rows: Sequence[Mapping[str, Any]],
    arrs: Mapping[str, Any],
    train_variant: str,
    eval_variant: str,
    layers: Sequence[int],
    methods: Sequence[str],
    rank: int,
    ridge_alpha: float,
    seed: int,
    random_repeats: int,
) -> Dict[str, Any]:
    alignment = _assert_eval_alignment(items=items, metadata_rows=metadata_rows, eval_variant=eval_variant)
    train_count = sum(1 for row in metadata_rows if str(row.get("variant", "")) == str(train_variant))
    eval_count = sum(1 for row in metadata_rows if str(row.get("variant", "")) == str(eval_variant))
    direction_summaries: Dict[str, Any] = {}
    for layer in layers:
        X_train, train_rows = _training_slice(
            arrs=arrs,
            metadata_rows=metadata_rows,
            train_variant=str(train_variant),
            layer=int(layer),
        )
        directions = build_steering_directions_for_layer(
            X_train=X_train,
            train_rows=train_rows,
            layer=int(layer),
            methods=methods,
            rank=int(rank),
            ridge_alpha=float(ridge_alpha),
            seed=int(seed),
            random_repeats=int(random_repeats),
        )
        direction_summaries[f"L{int(layer)}"] = [
            {
                "method": str(info["method"]),
                "rank": int(info["rank"]),
                "repeat": int(info["repeat"]),
                "control_family": str(info["control_family"]),
                "direction_source": str(info["direction_source"]),
                "steer_scale": float(info["steer_scale"]),
                "basis_projection_norm": float(info["basis_projection_norm"]),
                "direction_unit_norm": float(info["direction_unit_norm"]),
            }
            for info in directions
        ]
    return {
        "n_items": int(len(items)),
        "n_eval_sides": int(2 * len(items)),
        "n_unique_eval_prompts": int(alignment["eval_item_prompt_count"]),
        "n_metadata_rows": int(len(metadata_rows)),
        "n_train_rows": int(train_count),
        "n_eval_rows": int(eval_count),
        "train_variant": str(train_variant),
        "eval_variant": str(eval_variant),
        "layers": [int(x) for x in layers],
        "methods": [str(x) for x in methods],
        "rank": int(rank),
        "alignment": alignment,
        "directions": direction_summaries,
    }


class ActivationCache:
    def __init__(self, *, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._cache: Dict[Tuple[str, int], Tuple[torch.Tensor, int, int]] = {}

    @torch.no_grad()
    def get(self, prompt: str, *, layer: int) -> Tuple[torch.Tensor, int, int]:
        key = (str(prompt), int(layer))
        if key not in self._cache:
            ids = _encode_prompt(self.tokenizer, str(prompt), device=self.device)
            token_idx = int(ids.shape[-1]) - 1
            blocks = get_block_outputs(self.model, ids, layers=[int(layer)])
            vec = blocks[int(layer)][0, token_idx, :].detach()
            self._cache[key] = (vec, int(token_idx), int(ids.shape[-1]))
        return self._cache[key]


class ScoreCache:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        normalize_by_length: bool,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.normalize_by_length = bool(normalize_by_length)
        self._cache: Dict[str, Tuple[Any, float, Dict[str, float]]] = {}

    @torch.no_grad()
    def get(self, prompt: str, choices: Mapping[str, List[str]]) -> Tuple[Any, float, Dict[str, float]]:
        key = str(prompt)
        if key not in self._cache:
            scores = score_labels_next_continuations(
                self.model,
                self.tokenizer,
                key,
                choices,
                self.device,
                normalize_by_length=self.normalize_by_length,
            )
            self._cache[key] = (scores, _ordered_score(scores), _softmax(scores.by_label))
        return self._cache[key]


def _label_prob_columns(prefix: str, probs: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_prob_{label}": float(probs.get(str(label), 0.0)) for label in LABEL_ORDER}


@torch.no_grad()
def run_size_steering(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: Sequence[DisambPair],
    npz_path: Path,
    metadata_csv: Path,
    train_variant: str,
    eval_variant: str,
    layers: Sequence[int],
    methods: Sequence[str],
    rank: int,
    alphas: Sequence[float],
    device: torch.device,
    normalize_by_length: bool,
    ridge_alpha: float,
    seed: int,
    random_repeats: int,
) -> List[Dict[str, Any]]:
    metadata_rows = _read_csv(metadata_csv)
    _assert_eval_alignment(items=items, metadata_rows=metadata_rows, eval_variant=eval_variant)
    arrs = np.load(npz_path)
    activation_cache = ActivationCache(model=model, tokenizer=tokenizer, device=device)
    score_cache = ScoreCache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        normalize_by_length=normalize_by_length,
    )
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        X_train, train_rows = _training_slice(
            arrs=arrs,
            metadata_rows=metadata_rows,
            train_variant=str(train_variant),
            layer=int(layer),
        )
        directions = build_steering_directions_for_layer(
            X_train=X_train,
            train_rows=train_rows,
            layer=int(layer),
            methods=methods,
            rank=int(rank),
            ridge_alpha=float(ridge_alpha),
            seed=int(seed),
            random_repeats=int(random_repeats),
        )
        for item in items:
            md = dict(item.metadata or {})
            for side_name in ("a", "b"):
                side = _side(item, side_name)
                base_scores, base_ordered, base_probs = score_cache.get(side.prompt, item.choices)
                recv_vec, token_idx, prompt_length = activation_cache.get(side.prompt, layer=int(layer))
                ratio = _ratio_for_side(item, side_name)
                rho = float(math.log(float(ratio)))
                value = _to_float(md.get("value"))
                standard = _standard_for_side(item, side_name)
                for direction in directions:
                    unit = torch.tensor(
                        np.asarray(direction["unit_direction"], dtype=np.float32),
                        dtype=recv_vec.dtype,
                        device=device,
                    )
                    steer_scale = float(direction["steer_scale"])
                    for alpha in alphas:
                        alpha = float(alpha)
                        replacement = recv_vec + alpha * steer_scale * unit
                        patched_scores = score_labels_next_continuations_patched(
                            model,
                            tokenizer,
                            side.prompt,
                            item.choices,
                            device,
                            patch_site=PatchSpanSite(layer=int(layer), token_indices=(int(token_idx),)),
                            replacement=replacement,
                            normalize_by_length=normalize_by_length,
                        )
                        patched_ordered = _ordered_score(patched_scores)
                        patched_probs = _softmax(patched_scores.by_label)
                        effect = float(patched_ordered - base_ordered)
                        alpha_sign = _sign(alpha)
                        signed_effect = float(effect * alpha_sign) if alpha_sign else 0.0
                        direction_match = int(signed_effect > 0.0) if alpha_sign else float("nan")
                        row: Dict[str, Any] = {
                            "pair_id": str(item.pair_id),
                            "side": str(side_name),
                            "prompt": str(side.prompt),
                            "target": str(item.target),
                            "expected_label": str(side.expected_label),
                            "base_pred_label": str(base_scores.argmax_label()),
                            "patched_pred_label": str(patched_scores.argmax_label()),
                            "train_variant": str(train_variant),
                            "eval_variant": str(eval_variant),
                            "layer": int(layer),
                            "patch_site": "final_prompt_token",
                            "token_idx": int(token_idx),
                            "prompt_length": int(prompt_length),
                            "method": str(direction["method"]),
                            "rank": int(direction["rank"]),
                            "repeat": int(direction["repeat"]),
                            "control_family": str(direction["control_family"]),
                            "direction_source": str(direction["direction_source"]),
                            "alpha": float(alpha),
                            "alpha_sign": float(alpha_sign),
                            "rho": float(rho),
                            "ratio": float(ratio),
                            "value": float(value) if _is_finite(value) else "",
                            "standard": float(standard) if _is_finite(standard) else "",
                            "base_ordered_score": float(base_ordered),
                            "patched_ordered_score": float(patched_ordered),
                            "effect": float(effect),
                            "signed_effect": float(signed_effect),
                            "effect_per_alpha": (
                                float(effect / alpha) if abs(float(alpha)) > 1e-12 else float("nan")
                            ),
                            "direction_match": direction_match,
                            "steer_scale": float(steer_scale),
                            "steer_l2_norm": float(abs(alpha) * steer_scale),
                            "direction_unit_norm": float(direction["direction_unit_norm"]),
                            "basis_projection_norm": float(direction["basis_projection_norm"]),
                        }
                        row.update(_label_prob_columns("base", base_probs))
                        row.update(_label_prob_columns("patched", patched_probs))
                        rows.append(row)
    return rows


def _curve_slope(points: Sequence[Mapping[str, Any]]) -> float:
    xs = [_to_float(row.get("alpha")) for row in points]
    ys = [_to_float(row.get("effect")) for row in points]
    pairs = [(x, y) for x, y in zip(xs, ys) if _is_finite(x) and _is_finite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx = _mean(x for x, _ in pairs)
    my = _mean(y for _, y in pairs)
    denom = sum((x - mx) ** 2 for x, _ in pairs)
    if denom <= 1e-12:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in pairs) / denom)


def _curve_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int, int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["method"]),
                int(row["layer"]),
                int(row["rank"]),
                int(row["repeat"]),
                str(row["pair_id"]),
                str(row["side"]),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (method, layer, rank, repeat, pair_id, side), vals in sorted(grouped.items()):
        slope = _curve_slope(vals)
        out.append(
            {
                "method": str(method),
                "layer": int(layer),
                "rank": int(rank),
                "repeat": int(repeat),
                "pair_id": str(pair_id),
                "side": str(side),
                "curve_cluster": f"{pair_id}:{side}:r{repeat}",
                "n_alpha": int(len(vals)),
                "score_slope_per_alpha": float(slope),
                "positive_slope": int(float(slope) > 0.0) if _is_finite(slope) else float("nan"),
            }
        )
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, bootstrap_b: int, seed: int) -> Dict[str, Any]:
    alpha_groups: Dict[Tuple[str, int, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        alpha_groups[
            (
                str(row["method"]),
                int(row["layer"]),
                int(row["rank"]),
                float(row["alpha"]),
            )
        ].append(row)

    alpha_out: Dict[str, Any] = {}
    for (method, layer, rank, alpha), vals in sorted(alpha_groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2], kv[0][3])):
        matches = [_to_float(row.get("direction_match")) for row in vals]
        key = f"L{layer}_{method}_rank{rank}_alpha{alpha:g}"
        alpha_out[key] = {
            "method": str(method),
            "layer": int(layer),
            "rank": int(rank),
            "alpha": float(alpha),
            "n": int(len(vals)),
            "effect": _cluster_bootstrap_mean_ci(
                vals,
                value_key="effect",
                cluster_key="pair_id",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 11 + int(rank),
            ),
            "signed_effect": _cluster_bootstrap_mean_ci(
                vals,
                value_key="signed_effect",
                cluster_key="pair_id",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 17 + int(rank),
            ),
            "effect_per_alpha": _cluster_bootstrap_mean_ci(
                vals,
                value_key="effect_per_alpha",
                cluster_key="pair_id",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 19 + int(rank),
            ),
            "base_ordered_score": _cluster_bootstrap_mean_ci(
                vals,
                value_key="base_ordered_score",
                cluster_key="pair_id",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 23 + int(rank),
            ),
            "patched_ordered_score": _cluster_bootstrap_mean_ci(
                vals,
                value_key="patched_ordered_score",
                cluster_key="pair_id",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 29 + int(rank),
            ),
            "direction_match_rate": _mean(matches),
            "steer_l2_norm_mean": _mean(_to_float(row.get("steer_l2_norm")) for row in vals),
            "steer_scale_mean": _mean(_to_float(row.get("steer_scale")) for row in vals),
        }

    curves = _curve_rows(rows)
    curve_groups: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in curves:
        curve_groups[(str(row["method"]), int(row["layer"]), int(row["rank"]))].append(row)
    curve_out: Dict[str, Any] = {}
    for (method, layer, rank), vals in sorted(curve_groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        slopes = [_to_float(row.get("score_slope_per_alpha")) for row in vals]
        positives = [_to_float(row.get("positive_slope")) for row in vals]
        key = f"L{layer}_{method}_rank{rank}"
        curve_out[key] = {
            "method": str(method),
            "layer": int(layer),
            "rank": int(rank),
            "n_curves": int(len(vals)),
            "score_slope_per_alpha": _cluster_bootstrap_mean_ci(
                vals,
                value_key="score_slope_per_alpha",
                cluster_key="curve_cluster",
                b=int(bootstrap_b),
                seed=int(seed) + int(layer) * 31 + int(rank),
            ),
            "score_slope_median": _quantile(slopes, 0.5),
            "positive_slope_rate": _mean(positives),
        }
    return {
        "schema_version": "gradable_size_semantics_steering_v1",
        "n_rows": int(len(rows)),
        "n_curves": int(len(curves)),
        "alpha_groups": alpha_out,
        "curve_groups": curve_out,
    }


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Gradable Size Semantics-Derived Steering",
        "",
        f"- Train variant: `{summary['run']['train_variant']}`",
        f"- Eval variant: `{summary['run']['eval_variant']}`",
        f"- Layers: `{summary['run']['layers']}`",
        f"- Alpha sweep: `{summary['run']['alphas']}`",
        "",
        "This is fixed-vector activation steering, not donor-conditioned patching:",
        "`h' = h + alpha * s * d`, where `d` is derived from train-set",
        "high-rho minus low-rho size contrasts and `s` is the primary",
        "pca-delta-mean train-pair delta scale shared by non-sham directions.",
        "Positive alpha is oriented toward larger standard-relative size",
        "judgments.",
        "",
        "## Curve-level steering slope",
        "",
        "| layer | method | rank | curves | score slope/alpha | positive slope rate |",
        "| ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for group in summary["curve_groups"].values():
        slope = group["score_slope_per_alpha"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["layer"]),
                    str(group["method"]),
                    str(group["rank"]),
                    str(group["n_curves"]),
                    f"{float(slope['mean']):.3f} [{float(slope['lo']):.3f}, {float(slope['hi']):.3f}]",
                    f"{float(group['positive_slope_rate']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Alpha-level effects",
            "",
            "| layer | method | rank | alpha | n | effect | signed effect | direction match | steer norm |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for group in summary["alpha_groups"].values():
        effect = group["effect"]
        signed = group["signed_effect"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["layer"]),
                    str(group["method"]),
                    str(group["rank"]),
                    f"{float(group['alpha']):g}",
                    str(group["n"]),
                    f"{float(effect['mean']):.3f} [{float(effect['lo']):.3f}, {float(effect['hi']):.3f}]",
                    f"{float(signed['mean']):.3f} [{float(signed['lo']):.3f}, {float(signed['hi']):.3f}]",
                    f"{float(group['direction_match_rate']):.3f}",
                    f"{float(group['steer_l2_norm_mean']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation discipline:",
            "- `pca_delta_mean` is the primary semantics-derived steering direction.",
            "- `value`, `standard`, and `value_standard_2d` are explicit-variable controls.",
            "- `random_norm_matched` should stay weaker than the primary direction.",
            "- `sham` should remain near zero.",
            "- This supports linear residual-stream steering if the alpha curve is monotone and control directions are weaker; it is not a geodesic manifold-steering claim.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-vector steering for Bierwisch-style gradable size judgments.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl"),
    )
    p.add_argument(
        "--activations_npz",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_geometry_broad_final_token_gemma3.npz"),
    )
    p.add_argument(
        "--metadata_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_geometry_broad_final_token_gemma3.metadata.csv"),
    )
    p.add_argument("--train_variant", type=str, default="fictional_semantic_adjective_counts")
    p.add_argument("--eval_variant", type=str, default="iso_ratio_adjective_counts")
    p.add_argument("--layers", type=str, default="20")
    p.add_argument(
        "--methods",
        type=str,
        default="pca_delta_mean,value,standard,value_standard_2d,random_norm_matched,sham",
    )
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--alphas", type=str, default="-2,-1,-0.5,0,0.5,1,2")
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--random_repeats", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap_B", type=int, default=1000)
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument("--strict_finite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--out_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_semantic_steering_l20_r5_gemma3.csv"),
    )
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_semantic_steering_l20_r5_gemma3.summary.json"),
    )
    p.add_argument(
        "--out_md",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_semantic_steering_l20_r5_gemma3.md"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Validate inputs and directions without loading the model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    set_seed(int(args.seed))
    configure_logprob_computation(
        logprobs_dtype=getattr(torch, str(args.logprobs_dtype)),
        strict_finite=bool(args.strict_finite),
    )
    layers = _parse_int_list(str(args.layers))
    methods = _parse_csv_or_space(str(args.methods))
    alphas = _parse_float_list(str(args.alphas))
    items = load_disamb_pairs(str(args.data_path))
    metadata_rows = _read_csv(Path(str(args.metadata_csv)))
    arrs = np.load(Path(str(args.activations_npz)))
    validation = validate_steering_inputs(
        items=items,
        metadata_rows=metadata_rows,
        arrs=arrs,
        train_variant=str(args.train_variant),
        eval_variant=str(args.eval_variant),
        layers=layers,
        methods=methods,
        rank=int(args.rank),
        ridge_alpha=float(args.ridge_alpha),
        seed=int(args.seed),
        random_repeats=int(args.random_repeats),
    )
    if bool(args.dry_run):
        print(json.dumps({"dry_run": True, "validation": validation}, indent=2, sort_keys=True))
        return

    out_csv = Path(str(args.out_csv))
    out_summary = Path(str(args.out_summary))
    out_md = Path(str(args.out_md))
    if (out_csv.exists() or out_summary.exists() or out_md.exists()) and not bool(args.overwrite):
        raise FileExistsError("Output exists. Use --overwrite to replace steering outputs.")

    device = get_best_device() if str(args.device) == "auto" else torch.device(str(args.device))
    model_torch_dtype = str(args.torch_dtype) if args.torch_dtype else None
    if model_torch_dtype is None and str(getattr(device, "type", "")).lower() == "mps":
        model_torch_dtype = "float32"

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=model_torch_dtype,
        revision=str(args.revision) if args.revision else None,
        tokenizer_revision=str(args.tokenizer_revision) if args.tokenizer_revision else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        device_map=None,
    )
    rows = run_size_steering(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        npz_path=Path(str(args.activations_npz)),
        metadata_csv=Path(str(args.metadata_csv)),
        train_variant=str(args.train_variant),
        eval_variant=str(args.eval_variant),
        layers=layers,
        methods=methods,
        rank=int(args.rank),
        alphas=alphas,
        device=device,
        normalize_by_length=not bool(args.no_length_norm),
        ridge_alpha=float(args.ridge_alpha),
        seed=int(args.seed),
        random_repeats=int(args.random_repeats),
    )
    summary = summarize_rows(rows, bootstrap_b=int(args.bootstrap_B), seed=int(args.seed))
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "data_path": str(args.data_path),
        "activations_npz": str(args.activations_npz),
        "metadata_csv": str(args.metadata_csv),
        "train_variant": str(args.train_variant),
        "eval_variant": str(args.eval_variant),
        "layers": str(args.layers),
        "methods": str(args.methods),
        "rank": int(args.rank),
        "alphas": str(args.alphas),
        "random_repeats": int(args.random_repeats),
        "patch_site": "final_prompt_token",
        "device": str(device),
        "torch_dtype": str(model_torch_dtype or ""),
        "normalize_by_length": not bool(args.no_length_norm),
        "steering_rule": "h_prime = h + alpha * steer_scale * unit_direction",
        "git_commit": str(get_git_commit_hash(repo_root=ROOT, required=False)),
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": float(time.perf_counter() - t0),
        "versions": collect_versions(),
    }
    summary["validation"] = validation
    _write_csv(rows, out_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Wrote steering CSV: {out_csv}")
    print(f"Wrote steering summary: {out_summary}")
    print(f"Wrote steering Markdown: {out_md}")


if __name__ == "__main__":
    main()
