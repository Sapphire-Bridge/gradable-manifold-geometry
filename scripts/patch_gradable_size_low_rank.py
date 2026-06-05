from __future__ import annotations

import argparse
import csv
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
from aom.metrics.disamb import LabelScores, _encode_prompt, score_labels_next_continuations, score_labels_next_continuations_patched
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed


DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"
LABEL_ORDER = ("tiny", "small", "large", "huge")


def _parse_csv_or_space(raw: str) -> List[str]:
    return [part.strip() for part in re.split(r"[,\s]+", str(raw)) if part.strip()]


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x) for x in _parse_csv_or_space(raw)]
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x) for x in _parse_csv_or_space(raw)]
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


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


def _quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if _is_finite(float(x)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _bootstrap_mean_ci(xs: Sequence[float], *, b: int, seed: int) -> Dict[str, Any]:
    vals = [float(x) for x in xs if _is_finite(float(x))]
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(int(seed))
    n = len(vals)
    boot: List[float] = []
    for _ in range(int(b)):
        boot.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "mean": _mean(vals),
        "lo": _quantile(boot, 0.025),
        "hi": _quantile(boot, 0.975),
        "n": int(n),
    }


def _cluster_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    b: int,
    seed: int,
) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        val = _to_float(row.get(value_key))
        if _is_finite(val):
            grouped[str(row.get(cluster_key, ""))].append(float(val))
    keys = [k for k, vals in grouped.items() if vals]
    vals_all = [v for vals in grouped.values() for v in vals]
    if not keys or not vals_all:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_clusters": 0}
    rng = random.Random(int(seed))
    boot: List[float] = []
    for _ in range(int(b)):
        sample: List[float] = []
        for _ in keys:
            sample.extend(grouped[rng.choice(keys)])
        boot.append(_mean(sample))
    return {
        "mean": _mean(vals_all),
        "lo": _quantile(boot, 0.025),
        "hi": _quantile(boot, 0.975),
        "n": int(len(vals_all)),
        "n_clusters": int(len(keys)),
    }


def _softmax(scores: Mapping[str, float]) -> Dict[str, float]:
    vals = {str(k): float(v) for k, v in scores.items()}
    m = max(vals.values())
    exps = {k: math.exp(v - m) for k, v in vals.items()}
    denom = sum(exps.values())
    return {k: float(v / denom) for k, v in exps.items()}


def _kl_div(p: Mapping[str, float], q: Mapping[str, float], *, label_order: Sequence[str] = LABEL_ORDER) -> float:
    eps = 1e-12
    total = 0.0
    for label in label_order:
        pk = max(float(p.get(str(label), 0.0)), eps)
        qk = max(float(q.get(str(label), 0.0)), eps)
        total += pk * math.log(pk / qk)
    return float(total)


def _ordered_score(scores: LabelScores, label_order: Sequence[str] = LABEL_ORDER) -> float:
    probs = _softmax(scores.by_label)
    total = 0.0
    weighted = 0.0
    for i, label in enumerate(label_order):
        p = float(probs.get(str(label), 0.0))
        total += p
        weighted += float(i) * p
    return float(weighted / total) if total > 0.0 else float("nan")


def _ratio_for_side(item: DisambPair, side_name: str) -> float:
    md = dict(item.metadata or {})
    raw = md.get(f"ratio_{side_name}")
    if raw is None:
        raise ValueError(f"{item.pair_id}: missing ratio_{side_name} metadata")
    return float(raw)


def _metadata_by_variant(rows: Sequence[Mapping[str, Any]], variant: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("variant", "")) == str(variant)]


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        mat = mat[:, None]
    mat = np.asarray(mat, dtype=np.float64)
    keep = []
    for j in range(mat.shape[1]):
        col = mat[:, j].copy()
        for prev in keep:
            col = col - prev * float(np.dot(prev, col))
        norm = float(np.linalg.norm(col))
        if norm > 1e-9:
            keep.append(col / norm)
    if not keep:
        raise ValueError("Could not form nonzero orthonormal basis")
    return np.stack(keep, axis=1).astype(np.float32)


def _pca_basis(X: np.ndarray, rank: int) -> np.ndarray:
    Xc = np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return _orthonormalize(Vt[: int(rank)].T)


def _supervised_direction(X: np.ndarray, y: np.ndarray, *, ridge_alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray([i for i, val in enumerate(y.tolist()) if _is_finite(float(val))], dtype=np.int64)
    if len(valid) < 3:
        raise ValueError("Not enough finite targets for supervised direction")
    Xv = X[valid]
    yv = y[valid]
    Xc = Xv - Xv.mean(axis=0, keepdims=True)
    yc = yv - float(yv.mean())
    K = Xc @ Xc.T
    coef = np.linalg.solve(K + float(ridge_alpha) * np.eye(K.shape[0], dtype=np.float64), yc)
    w = Xc.T @ coef
    return _orthonormalize(w)


def _delta_mean_direction(X: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
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
        delta = np.asarray(X[high] - X[low], dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            deltas.append(delta / norm)
    if not deltas:
        raise ValueError("No valid pair deltas for delta-mean direction")
    return _orthonormalize(np.mean(np.stack(deltas, axis=0), axis=0))


def _random_basis(dim: int, rank: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return _orthonormalize(rng.normal(size=(int(dim), int(rank))))


def _target_vector(rows: Sequence[Mapping[str, Any]], target: str) -> np.ndarray:
    if target == "rho":
        return np.asarray([_to_float(row.get("rho")) for row in rows], dtype=np.float64)
    if target == "signed_score":
        return np.asarray([_to_float(row.get("signed_score")) for row in rows], dtype=np.float64)
    if target == "ordered_score":
        return np.asarray([_to_float(row.get("ordered_score")) for row in rows], dtype=np.float64)
    if target == "log_value":
        return np.asarray([math.log(max(_to_float(row.get("value")), 1e-9)) for row in rows], dtype=np.float64)
    if target == "log_standard":
        return np.asarray([math.log(max(_to_float(row.get("standard")), 1e-9)) for row in rows], dtype=np.float64)
    raise ValueError(f"Unknown supervised target {target!r}")


def _build_bases_for_layer(
    *,
    X_train: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    ranks: Sequence[int],
    ridge_alpha: float,
    seed: int,
    random_repeats: int,
) -> List[Dict[str, Any]]:
    bases: List[Dict[str, Any]] = []
    dim = int(X_train.shape[1])
    for method in methods:
        method = str(method)
        if method == "pca":
            for rank in ranks:
                basis = _pca_basis(X_train, int(rank))
                bases.append({"method": "pca", "rank": int(basis.shape[1]), "repeat": 0, "basis": basis})
        elif method in {"rho", "signed_score", "ordered_score", "value", "standard"}:
            target = {"value": "log_value", "standard": "log_standard"}.get(method, method)
            basis = _supervised_direction(X_train, _target_vector(train_rows, target), ridge_alpha=ridge_alpha)
            bases.append({"method": method, "rank": int(basis.shape[1]), "repeat": 0, "basis": basis})
        elif method == "value_standard_2d":
            value_basis = _supervised_direction(X_train, _target_vector(train_rows, "log_value"), ridge_alpha=ridge_alpha)
            standard_basis = _supervised_direction(
                X_train,
                _target_vector(train_rows, "log_standard"),
                ridge_alpha=ridge_alpha,
            )
            basis = _orthonormalize(np.concatenate([value_basis, standard_basis], axis=1))
            bases.append({"method": "value_standard_2d", "rank": int(basis.shape[1]), "repeat": 0, "basis": basis})
        elif method == "delta_mean":
            basis = _delta_mean_direction(X_train, train_rows)
            bases.append({"method": "delta_mean", "rank": int(basis.shape[1]), "repeat": 0, "basis": basis})
        elif method in {"random", "random_norm_matched"}:
            for rank in ranks:
                for repeat in range(int(random_repeats)):
                    basis = _random_basis(
                        dim,
                        int(rank),
                        seed=int(seed) + int(rank) * 101 + int(repeat) * 10007 + (7919 if method == "random_norm_matched" else 0),
                    )
                    bases.append(
                        {
                            "method": method,
                            "rank": int(basis.shape[1]),
                            "repeat": int(repeat),
                            "basis": basis,
                        }
                    )
        else:
            raise ValueError(f"Unknown method {method!r}")
    return bases


def _project_delta(delta: torch.Tensor, basis: np.ndarray, *, device: torch.device) -> torch.Tensor:
    U = torch.tensor(basis, dtype=delta.dtype, device=device)
    return U @ (U.T @ delta)


def _assert_eval_alignment(
    *,
    items: Sequence[DisambPair],
    metadata_rows: Sequence[Mapping[str, Any]],
    eval_variant: str,
) -> Dict[str, Any]:
    item_prompts = {str(item.a.prompt) for item in items} | {str(item.b.prompt) for item in items}
    meta_prompts = {str(row.get("prompt", "")) for row in metadata_rows if str(row.get("variant", "")) == str(eval_variant)}
    if not item_prompts:
        raise ValueError("No eval item prompts loaded")
    if not meta_prompts:
        raise ValueError(f"No metadata rows found for eval_variant={eval_variant!r}")
    missing_from_metadata = sorted(item_prompts - meta_prompts)
    missing_from_items = sorted(meta_prompts - item_prompts)
    if missing_from_metadata or missing_from_items:
        raise ValueError(
            "JSONL eval items and geometry metadata are out of sync: "
            f"missing_from_metadata={len(missing_from_metadata)}, missing_from_items={len(missing_from_items)}"
        )
    regimes = sorted({str((item.metadata or {}).get("regime", "")) for item in items})
    return {
        "eval_item_prompt_count": int(len(item_prompts)),
        "metadata_eval_prompt_count": int(len(meta_prompts)),
        "eval_item_regimes": regimes,
    }


@torch.no_grad()
def run_low_rank_patch(
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
    ranks: Sequence[int],
    alphas: Sequence[float],
    device: torch.device,
    normalize_by_length: bool,
    ridge_alpha: float,
    seed: int,
    random_repeats: int,
) -> List[Dict[str, Any]]:
    metadata_rows = _read_csv(metadata_csv)
    _ = _assert_eval_alignment(items=items, metadata_rows=metadata_rows, eval_variant=eval_variant)
    arrs = np.load(npz_path)
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        key = f"X_layer_{int(layer)}"
        if key not in arrs:
            raise KeyError(f"Missing {key} in {npz_path}")
        X = np.asarray(arrs[key], dtype=np.float32)
        if X.shape[0] != len(metadata_rows):
            raise ValueError(f"{key} has {X.shape[0]} rows; metadata has {len(metadata_rows)}")
        train_idx = [i for i, row in enumerate(metadata_rows) if str(row.get("variant", "")) == str(train_variant)]
        if len(train_idx) < 4:
            raise ValueError(f"Not enough train rows for variant {train_variant!r}")
        train_rows = [metadata_rows[i] for i in train_idx]
        X_train = X[np.asarray(train_idx, dtype=np.int64)]
        bases = _build_bases_for_layer(
            X_train=X_train,
            train_rows=train_rows,
            methods=methods,
            ranks=ranks,
            ridge_alpha=float(ridge_alpha),
            seed=int(seed) + int(layer) * 997,
            random_repeats=int(random_repeats),
        )

        for item in items:
            sides = (("a", item.a), ("b", item.b))
            for donor_name, donor in sides:
                for recv_name, recv in sides:
                    if donor_name == recv_name:
                        continue
                    donor_ratio = _ratio_for_side(item, donor_name)
                    recv_ratio = _ratio_for_side(item, recv_name)
                    predictor_delta = float(math.log(donor_ratio) - math.log(recv_ratio))
                    if predictor_delta == 0.0:
                        continue
                    expected_sign = 1.0 if predictor_delta > 0.0 else -1.0

                    donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
                    recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
                    donor_token_idx = int(donor_ids.shape[-1]) - 1
                    recv_token_idx = int(recv_ids.shape[-1]) - 1

                    base_scores = score_labels_next_continuations(
                        model,
                        tokenizer,
                        recv.prompt,
                        item.choices,
                        device,
                        normalize_by_length=normalize_by_length,
                    )
                    donor_scores = score_labels_next_continuations(
                        model,
                        tokenizer,
                        donor.prompt,
                        item.choices,
                        device,
                        normalize_by_length=normalize_by_length,
                    )
                    base_ordered = _ordered_score(base_scores)
                    donor_ordered = _ordered_score(donor_scores)
                    base_probs = _softmax(base_scores.by_label)
                    donor_probs = _softmax(donor_scores.by_label)
                    raw_gap = float(donor_ordered - base_ordered)
                    aligned_raw_gap = float(raw_gap * expected_sign)

                    donor_blocks = get_block_outputs(model, donor_ids, layers=[int(layer)])
                    recv_blocks = get_block_outputs(model, recv_ids, layers=[int(layer)])
                    donor_vec = donor_blocks[int(layer)][0, donor_token_idx, :].detach()
                    recv_vec = recv_blocks[int(layer)][0, recv_token_idx, :].detach()
                    delta = donor_vec - recv_vec

                    delta_norm = float(torch.linalg.vector_norm(delta).item())
                    candidate_norm_by_rank: Dict[int, float] = defaultdict(float)
                    projected_by_basis: List[Dict[str, Any]] = []
                    for basis_info in bases:
                        basis = np.asarray(basis_info["basis"], dtype=np.float32)
                        projected_delta = _project_delta(delta, basis, device=device)
                        projected_norm = float(torch.linalg.vector_norm(projected_delta).item())
                        if str(basis_info["method"]) not in {"random", "random_norm_matched"}:
                            candidate_norm_by_rank[int(basis_info["rank"])] = max(
                                float(candidate_norm_by_rank[int(basis_info["rank"])]),
                                projected_norm,
                            )
                        projected_by_basis.append(
                            {
                                **basis_info,
                                "projected_delta": projected_delta,
                                "projected_delta_norm": projected_norm,
                            }
                        )

                    method_replacements: List[Dict[str, Any]] = [
                        {
                            "method": "full",
                            "rank": int(delta.numel()),
                            "repeat": 0,
                            "alpha": 1.0,
                            "replacement": donor_vec,
                            "projected_delta_norm": delta_norm,
                        },
                        {
                            "method": "sham",
                            "rank": 0,
                            "repeat": 0,
                            "alpha": 1.0,
                            "replacement": recv_vec,
                            "projected_delta_norm": 0.0,
                        },
                    ]
                    for basis_info in projected_by_basis:
                        projected_delta = basis_info["projected_delta"]
                        projected_norm = float(basis_info["projected_delta_norm"])
                        if str(basis_info["method"]) == "random_norm_matched":
                            target_norm = float(candidate_norm_by_rank.get(int(basis_info["rank"]), 0.0))
                            if projected_norm > 1e-9 and target_norm > 0.0:
                                projected_delta = projected_delta * float(target_norm / projected_norm)
                                projected_norm = float(target_norm)
                        for alpha in alphas:
                            method_replacements.append(
                                {
                                    "method": str(basis_info["method"]),
                                    "rank": int(basis_info["rank"]),
                                    "repeat": int(basis_info.get("repeat", 0)),
                                    "alpha": float(alpha),
                                    "replacement": recv_vec + float(alpha) * projected_delta,
                                    "projected_delta_norm": projected_norm,
                                }
                            )

                    local_rows: List[Dict[str, Any]] = []
                    for repl in method_replacements:
                        patched_scores = score_labels_next_continuations_patched(
                            model,
                            tokenizer,
                            recv.prompt,
                            item.choices,
                            device,
                            patch_site=PatchSpanSite(layer=int(layer), token_indices=(int(recv_token_idx),)),
                            replacement=repl["replacement"],
                            normalize_by_length=normalize_by_length,
                        )
                        patched_ordered = _ordered_score(patched_scores)
                        patched_probs = _softmax(patched_scores.by_label)
                        effect = float(patched_ordered - base_ordered)
                        aligned_effect = float(effect * expected_sign)
                        projected_delta_norm = float(repl["projected_delta_norm"])
                        local_rows.append(
                            {
                                "pair_id": str(item.pair_id),
                                "direction": f"{donor_name}_to_{recv_name}",
                                "train_variant": str(train_variant),
                                "eval_variant": str(eval_variant),
                                "layer": int(layer),
                                "patch_site": "final_prompt_token",
                                "method": str(repl["method"]),
                                "rank": int(repl["rank"]),
                                "repeat": int(repl.get("repeat", 0)),
                                "alpha": float(repl["alpha"]),
                                "donor_side": str(donor_name),
                                "recv_side": str(recv_name),
                                "donor_expected_label": str(donor.expected_label),
                                "recv_expected_label": str(recv.expected_label),
                                "donor_ratio": donor_ratio,
                                "recv_ratio": recv_ratio,
                                "predictor_delta": predictor_delta,
                                "expected_sign": expected_sign,
                                "base_ordered_score": base_ordered,
                                "donor_ordered_score": donor_ordered,
                                "patched_ordered_score": patched_ordered,
                                "raw_gap": raw_gap,
                                "aligned_raw_gap": aligned_raw_gap,
                                "effect": effect,
                                "aligned_effect": aligned_effect,
                                "patch_fraction": float(aligned_effect / max(abs(aligned_raw_gap), 1e-9)),
                                "delta_norm": delta_norm,
                                "projected_delta_norm": projected_delta_norm,
                                "projected_norm_fraction": float(projected_delta_norm / max(delta_norm, 1e-9)),
                                "direction_match": int(aligned_effect > 0.0),
                                "kl_base_to_donor": _kl_div(base_probs, donor_probs),
                                "kl_patched_to_donor": _kl_div(patched_probs, donor_probs),
                                "kl_patched_to_receiver": _kl_div(patched_probs, base_probs),
                                "kl_patched_to_donor_delta_vs_base": float(_kl_div(patched_probs, donor_probs) - _kl_div(base_probs, donor_probs)),
                            }
                        )
                    full_effect = next(
                        (
                            float(row["aligned_effect"])
                            for row in local_rows
                            if str(row["method"]) == "full" and float(row["alpha"]) == 1.0
                        ),
                        float("nan"),
                    )
                    for row in local_rows:
                        row["full_aligned_effect"] = full_effect
                        row["recovery_vs_full"] = (
                            float(row["aligned_effect"]) / float(full_effect)
                            if _is_finite(full_effect) and abs(float(full_effect)) > 1e-9
                            else float("nan")
                        )
                    rows.extend(local_rows)
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, bootstrap_b: int, seed: int) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int, str, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method"]),
                int(row["layer"]),
                str(row["eval_variant"]),
                int(row["rank"]),
                float(row["alpha"]),
            )
        ].append(row)
    out_groups: Dict[str, Any] = {}
    for (method, layer, eval_variant, rank, alpha), vals in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][3], kv[0][4])):
        aligned = [float(r["aligned_effect"]) for r in vals]
        patch_fraction = [float(r["patch_fraction"]) for r in vals]
        recovery_vs_full = [float(r["recovery_vs_full"]) for r in vals]
        matches = [float(r["direction_match"]) for r in vals]
        kl_delta = [float(r["kl_patched_to_donor_delta_vs_base"]) for r in vals]
        norm_fraction = [float(r["projected_norm_fraction"]) for r in vals]
        valid_gap_005 = [r for r in vals if abs(float(r["aligned_raw_gap"])) >= 0.05]
        valid_gap_010 = [r for r in vals if abs(float(r["aligned_raw_gap"])) >= 0.10]
        key = f"L{layer}_{method}_rank{rank}_alpha{alpha:g}_{eval_variant}"
        out_groups[key] = {
            "method": method,
            "layer": int(layer),
            "eval_variant": eval_variant,
            "rank": int(rank),
            "alpha": float(alpha),
            "n": int(len(vals)),
            "n_gap_ge_0_05": int(len(valid_gap_005)),
            "n_gap_ge_0_10": int(len(valid_gap_010)),
            "aligned_effect": _cluster_bootstrap_mean_ci(
                vals,
                value_key="aligned_effect",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 11 + int(rank),
            ),
            "patch_fraction": _cluster_bootstrap_mean_ci(
                vals,
                value_key="patch_fraction",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 17 + int(rank),
            ),
            "patch_fraction_median": _quantile(patch_fraction, 0.5),
            "patch_fraction_gap_ge_0_10": _cluster_bootstrap_mean_ci(
                valid_gap_010,
                value_key="patch_fraction",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 23 + int(rank),
            ),
            "recovery_vs_full": _cluster_bootstrap_mean_ci(
                vals,
                value_key="recovery_vs_full",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 29 + int(rank),
            ),
            "recovery_vs_full_median": _quantile(recovery_vs_full, 0.5),
            "recovery_vs_full_gap_ge_0_10": _cluster_bootstrap_mean_ci(
                valid_gap_010,
                value_key="recovery_vs_full",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 31 + int(rank),
            ),
            "projected_norm_fraction": _cluster_bootstrap_mean_ci(
                vals,
                value_key="projected_norm_fraction",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 37 + int(rank),
            ),
            "projected_norm_fraction_median": _quantile(norm_fraction, 0.5),
            "direction_match_rate": _mean(matches),
            "kl_patched_to_donor_delta_vs_base": _cluster_bootstrap_mean_ci(
                vals,
                value_key="kl_patched_to_donor_delta_vs_base",
                cluster_key="pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 19 + int(rank),
            ),
        }
    return {
        "schema_version": "gradable_size_low_rank_patch_v1",
        "n_rows": int(len(rows)),
        "groups": out_groups,
    }


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Gradable Size Low-Rank Causal Subspace Patching",
        "",
        f"- Train variant: `{summary['run']['train_variant']}`",
        f"- Eval variant: `{summary['run']['eval_variant']}`",
        f"- Layers: `{summary['run']['layers']}`",
        "",
        "| layer | method | rank | alpha | n | aligned effect | recovery/full | patch frac | norm frac | dir match | KL delta vs base |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for group in summary["groups"].values():
        ae = group["aligned_effect"]
        rvf = group["recovery_vs_full"]
        pf = group["patch_fraction"]
        nf = group["projected_norm_fraction"]
        kd = group["kl_patched_to_donor_delta_vs_base"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["layer"]),
                    str(group["method"]),
                    str(group["rank"]),
                    f"{float(group['alpha']):g}",
                    str(group["n"]),
                    f"{float(ae['mean']):.3f} [{float(ae['lo']):.3f}, {float(ae['hi']):.3f}]",
                    f"{float(rvf['mean']):.3f} [{float(rvf['lo']):.3f}, {float(rvf['hi']):.3f}]",
                    f"{float(pf['mean']):.3f} [{float(pf['lo']):.3f}, {float(pf['hi']):.3f}]",
                    f"{float(nf['mean']):.3f}",
                    f"{float(group['direction_match_rate']):.3f}",
                    f"{float(kd['mean']):.3f} [{float(kd['lo']):.3f}, {float(kd['hi']):.3f}]",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation discipline:",
            "- `full` is the full-vector raw-patch upper bound.",
            "- `recovery/full` is the primary low-rank recovery metric.",
            "- `sham` should remain near zero.",
            "- Low-rank methods should be interpreted only if they beat `random`, `random_norm_matched`, `value`, `standard`, and `value_standard_2d` controls.",
            "- Passing this gate still precedes SAE/CLT feature-group discovery.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-rank causal subspace patching for gradable size calibration.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--activations_npz", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--train_variant", type=str, required=True)
    p.add_argument("--eval_variant", type=str, required=True)
    p.add_argument("--layers", type=str, default="16,20,24")
    p.add_argument("--methods", type=str, default="pca,rho,ordered_score,signed_score,delta_mean,random,random_norm_matched,value,standard,value_standard_2d")
    p.add_argument("--ranks", type=str, default="1,2,5")
    p.add_argument("--alphas", type=str, default="1.0")
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
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_summary", type=str, required=True)
    p.add_argument("--out_md", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
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
    device = get_best_device() if str(args.device) == "auto" else torch.device(str(args.device))
    model_torch_dtype = str(args.torch_dtype) if args.torch_dtype else None
    if model_torch_dtype is None and str(getattr(device, "type", "")).lower() == "mps":
        model_torch_dtype = "float32"
    out_csv = Path(str(args.out_csv))
    out_summary = Path(str(args.out_summary))
    out_md = Path(str(args.out_md))
    if (out_csv.exists() or out_summary.exists() or out_md.exists()) and not bool(args.overwrite):
        raise FileExistsError("Output exists. Use --overwrite to replace low-rank patch outputs.")

    items = load_disamb_pairs(str(args.data_path))
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
    rows = run_low_rank_patch(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        npz_path=Path(str(args.activations_npz)),
        metadata_csv=Path(str(args.metadata_csv)),
        train_variant=str(args.train_variant),
        eval_variant=str(args.eval_variant),
        layers=_parse_int_list(str(args.layers)),
        methods=_parse_csv_or_space(str(args.methods)),
        ranks=_parse_int_list(str(args.ranks)),
        alphas=_parse_float_list(str(args.alphas)),
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
        "ranks": str(args.ranks),
        "alphas": str(args.alphas),
        "random_repeats": int(args.random_repeats),
        "patch_site": "final_prompt_token",
        "device": str(device),
        "torch_dtype": str(model_torch_dtype or ""),
        "normalize_by_length": not bool(args.no_length_norm),
        "replacement_source": "decoder_block_output_hook",
        "git_commit": str(get_git_commit_hash(repo_root=ROOT, required=False)),
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": float(time.perf_counter() - t0),
        "versions": collect_versions(),
    }
    _write_csv(rows, out_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Wrote low-rank patch CSV: {out_csv}")
    print(f"Wrote low-rank patch summary: {out_summary}")
    print(f"Wrote low-rank patch Markdown: {out_md}")


if __name__ == "__main__":
    main()
