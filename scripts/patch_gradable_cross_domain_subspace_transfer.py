from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from aom.metrics.disamb import score_labels_next_continuations
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed
from scripts.make_gradable_cross_domain_delta_rho_controls import _metadata_coordinate
from scripts.patch_gradable_cross_domain_low_rank_control import ActivationCache, _score_replacement
from scripts.patch_gradable_size_low_rank import (
    _build_bases_for_layer,
    _cluster_bootstrap_mean_ci,
    _is_finite,
    _mean,
    _ordered_score,
    _parse_float_list,
    _parse_int_list,
    _project_delta,
    _quantile,
    _random_basis,
    _read_csv,
    _to_float,
    _write_csv,
)


DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"


def _parse_csv_or_space(raw: str) -> List[str]:
    return [part.strip() for part in re.split(r"[,\s]+", str(raw)) if part.strip()]


def _stable_int(raw: str) -> int:
    h = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _side(item: DisambPair, side_name: str) -> PromptSide:
    if str(side_name) == "a":
        return item.a
    if str(side_name) == "b":
        return item.b
    raise ValueError(f"Unknown side {side_name!r}")


def _size_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(row["size_pair_id"]), str(row["size_donor_side"]), str(row["size_recv_side"]))


def _load_control_rows(path: Path, *, source_domains: Sequence[str]) -> List[Dict[str, str]]:
    rows = _read_csv(path)
    wanted = {str(x) for x in source_domains}
    if wanted:
        rows = [row for row in rows if str(row.get("source_domain", "")) in wanted]
    if not rows:
        raise ValueError(f"No control rows loaded from {path}")
    return rows


def _unique_source_paths(control_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Path]:
    by_domain: Dict[str, set[str]] = defaultdict(set)
    for row in control_rows:
        by_domain[str(row["source_domain"])].add(str(row["source_data_path"]))
    paths: Dict[str, Path] = {}
    for domain, raw_paths in sorted(by_domain.items()):
        if len(raw_paths) != 1:
            raise ValueError(f"{domain}: expected one source_data_path, got {sorted(raw_paths)}")
        paths[str(domain)] = Path(next(iter(raw_paths)))
    return paths


def _unique_prompt_rows(items: Sequence[DisambPair], *, domain: str, data_path: Path) -> List[Dict[str, Any]]:
    by_prompt: Dict[str, Dict[str, Any]] = {}
    for item in items:
        for side_name in ("a", "b"):
            side = _side(item, side_name)
            by_prompt.setdefault(
                str(side.prompt),
                {
                    "domain": str(domain),
                    "data_path": str(data_path),
                    "pair_id": str(item.pair_id),
                    "side": str(side_name),
                    "prompt": str(side.prompt),
                    "expected_label": str(side.expected_label),
                },
            )
    return [by_prompt[key] for key in sorted(by_prompt)]


def _basis_from_activations(X: np.ndarray, *, rank: int, center: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if int(X.shape[0]) < int(rank):
        raise ValueError(f"Need at least rank rows for PCA; got {X.shape[0]} rows for rank={rank}")
    Xc = np.asarray(X, dtype=np.float64)
    if bool(center):
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = np.asarray(Vt[: int(rank)].T, dtype=np.float64)
    keep: List[np.ndarray] = []
    for j in range(basis.shape[1]):
        col = basis[:, j].copy()
        for prev in keep:
            col = col - prev * float(np.dot(prev, col))
        norm = float(np.linalg.norm(col))
        if norm > 1e-9:
            keep.append(col / norm)
    if len(keep) < int(rank):
        raise ValueError(f"Only {len(keep)} nonzero PCA directions available for rank={rank}")
    return np.stack(keep[: int(rank)], axis=1).astype(np.float32)


def _source_basis_for_layer(
    *,
    cache: ActivationCache,
    prompt_rows: Sequence[Mapping[str, Any]],
    layer: int,
    rank: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    vecs: List[np.ndarray] = []
    for row in prompt_rows:
        vec, _, _ = cache.get(str(row["prompt"]), layer=int(layer))
        vecs.append(vec.detach().to("cpu", dtype=torch.float32).numpy())
    X = np.stack(vecs, axis=0).astype(np.float32)
    basis = _basis_from_activations(X, rank=int(rank), center=True)
    return basis, {
        "basis_train_kind": "state_prompts",
        "n_train_prompts": int(len(prompt_rows)),
        "n_train_items": int(len(prompt_rows)),
        "n_train_pairs": int(len({str(row["pair_id"]) for row in prompt_rows})),
        "activation_shape": list(X.shape),
    }


def _source_delta_rows(items: Sequence[DisambPair], *, domain: str, data_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        coord_a, coord_kind_a = _metadata_coordinate(item, "a")
        coord_b, coord_kind_b = _metadata_coordinate(item, "b")
        if not (_is_finite(coord_a) and _is_finite(coord_b)) or abs(float(coord_a) - float(coord_b)) <= 1e-12:
            continue
        high_side, low_side = ("a", "b") if float(coord_a) > float(coord_b) else ("b", "a")
        high = _side(item, high_side)
        low = _side(item, low_side)
        high_coord = float(coord_a if high_side == "a" else coord_b)
        low_coord = float(coord_b if high_side == "a" else coord_a)
        rows.append(
            {
                "domain": str(domain),
                "data_path": str(data_path),
                "pair_id": str(item.pair_id),
                "high_side": str(high_side),
                "low_side": str(low_side),
                "high_prompt": str(high.prompt),
                "low_prompt": str(low.prompt),
                "high_expected_label": str(high.expected_label),
                "low_expected_label": str(low.expected_label),
                "coordinate_kind": str(coord_kind_a if coord_kind_a == coord_kind_b else f"{coord_kind_a}/{coord_kind_b}"),
                "high_coordinate": float(high_coord),
                "low_coordinate": float(low_coord),
                "delta_coordinate": float(high_coord - low_coord),
            }
        )
    return rows


def _source_delta_basis_for_layer(
    *,
    cache: ActivationCache,
    delta_rows: Sequence[Mapping[str, Any]],
    layer: int,
    rank: int,
    normalize_deltas: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    deltas: List[np.ndarray] = []
    raw_norms: List[float] = []
    for row in delta_rows:
        high_vec, _, _ = cache.get(str(row["high_prompt"]), layer=int(layer))
        low_vec, _, _ = cache.get(str(row["low_prompt"]), layer=int(layer))
        delta = (high_vec - low_vec).detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32)
        norm = float(np.linalg.norm(delta))
        raw_norms.append(norm)
        if bool(normalize_deltas):
            if norm <= 1e-9:
                continue
            delta = delta / float(norm)
        deltas.append(delta)
    if len(deltas) < int(rank):
        raise ValueError(f"Only {len(deltas)} usable source deltas; need rank={int(rank)}")
    X = np.stack(deltas, axis=0).astype(np.float32)
    basis = _basis_from_activations(X, rank=int(rank), center=False)
    return basis, {
        "basis_train_kind": "pair_deltas_unit" if bool(normalize_deltas) else "pair_deltas_raw",
        "n_train_prompts": 0,
        "n_train_items": int(len(deltas)),
        "n_train_pairs": int(len({str(row["pair_id"]) for row in delta_rows})),
        "activation_shape": list(X.shape),
        "normalize_source_deltas": bool(normalize_deltas),
        "mean_raw_delta_norm": _mean(raw_norms),
        "min_raw_delta_norm": min(raw_norms) if raw_norms else float("nan"),
        "max_raw_delta_norm": max(raw_norms) if raw_norms else float("nan"),
    }


def _size_basis_for_layer(
    *,
    arrs: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    size_train_variant: str,
    layer: int,
    rank: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    key = f"X_layer_{int(layer)}"
    X = np.asarray(arrs[key], dtype=np.float32)
    train_idx = [i for i, row in enumerate(metadata_rows) if str(row.get("variant", "")) == str(size_train_variant)]
    train_rows = [metadata_rows[i] for i in train_idx]
    X_train = X[np.asarray(train_idx, dtype=np.int64)]
    bases = _build_bases_for_layer(
        X_train=X_train,
        train_rows=train_rows,
        methods=["pca"],
        ranks=[int(rank)],
        ridge_alpha=10.0,
        seed=int(seed),
        random_repeats=0,
    )
    for basis_info in bases:
        if str(basis_info["method"]) == "pca" and int(basis_info["rank"]) == int(rank):
            return np.asarray(basis_info["basis"], dtype=np.float32), {
                "basis_train_kind": "state_prompts",
                "n_train_prompts": int(len(train_rows)),
                "n_train_items": int(len(train_rows)),
                "n_train_pairs": int(len({str(row.get("pair_id", "")) for row in train_rows})),
                "activation_shape": list(X_train.shape),
            }
    raise ValueError(f"Could not build size pca rank {rank} for layer {layer}")


def _subspace_alignment(U: np.ndarray, V: np.ndarray) -> Dict[str, Any]:
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = [float(math.degrees(math.acos(float(x)))) for x in s]
    denom = max(min(U.shape[1], V.shape[1]), 1)
    return {
        "mean_squared_cosine": float(np.sum(s**2) / float(denom)),
        "singular_values": [float(x) for x in s.tolist()],
        "principal_angles_deg": angles,
        "max_principal_angle_deg": max(angles) if angles else float("nan"),
        "mean_principal_angle_deg": _mean(angles),
    }


def _validate_inputs(
    *,
    control_rows: Sequence[Mapping[str, Any]],
    size_items: Sequence[DisambPair],
    source_prompt_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_delta_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_basis_mode: str,
    metadata_rows: Sequence[Mapping[str, Any]],
    arrs: Mapping[str, Any],
    size_train_variant: str,
    layers: Sequence[int],
    rank: int,
    min_unique_source_pairs: int,
    min_size_pairs: int,
) -> Dict[str, Any]:
    item_ids = {str(item.pair_id) for item in size_items}
    control_item_ids = {str(row["size_pair_id"]) for row in control_rows}
    missing = sorted(control_item_ids - item_ids)
    if missing:
        raise ValueError(f"Control table references size pairs missing from size_data_path: {missing[:5]}")
    train_idx = [i for i, row in enumerate(metadata_rows) if str(row.get("variant", "")) == str(size_train_variant)]
    if len(train_idx) < int(rank):
        raise ValueError(f"Not enough size train rows for variant {size_train_variant!r}: {len(train_idx)}")
    for layer in layers:
        key = f"X_layer_{int(layer)}"
        if key not in arrs:
            raise KeyError(f"Missing {key} in activation NPZ")
        arr = np.asarray(arrs[key])
        if arr.shape[0] != len(metadata_rows):
            raise ValueError(f"{key} has {arr.shape[0]} rows but metadata has {len(metadata_rows)}")

    by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in control_rows:
        by_source[str(row["source_domain"])].append(row)
    source_summary: Dict[str, Any] = {}
    for domain, rows in sorted(by_source.items()):
        unique_source_pairs = {str(row["source_pair_id"]) for row in rows}
        unique_size_pairs = {str(row["size_pair_id"]) for row in rows}
        train_prompts = list(source_prompt_rows.get(str(domain), []))
        if len(unique_source_pairs) < int(min_unique_source_pairs):
            raise ValueError(
                f"{domain}: only {len(unique_source_pairs)} unique matched source pairs; "
                f"minimum is {int(min_unique_source_pairs)}"
            )
        if len(unique_size_pairs) < int(min_size_pairs):
            raise ValueError(f"{domain}: only {len(unique_size_pairs)} size pairs; minimum is {int(min_size_pairs)}")
        train_deltas = list(source_delta_rows.get(str(domain), []))
        if str(source_basis_mode) == "states":
            train_n = len(train_prompts)
            train_kind = "state_prompts"
            if train_n < int(rank):
                raise ValueError(f"{domain}: only {train_n} source train prompts; need rank={int(rank)}")
        elif str(source_basis_mode) == "deltas":
            train_n = len(train_deltas)
            train_kind = "pair_deltas"
            if train_n < int(rank):
                raise ValueError(f"{domain}: only {train_n} source train deltas; need rank={int(rank)}")
        else:
            raise ValueError(f"Unknown source_basis_mode {source_basis_mode!r}")
        source_summary[domain] = {
            "n_rows": int(len(rows)),
            "n_unique_matched_source_pairs": int(len(unique_source_pairs)),
            "n_unique_size_pairs": int(len(unique_size_pairs)),
            "n_source_train_prompts": int(len(train_prompts)),
            "n_source_train_deltas": int(len(train_deltas)),
            "n_source_train_pairs": int(len({str(row["pair_id"]) for row in train_prompts})),
            "source_basis_mode": str(source_basis_mode),
            "source_train_kind": str(train_kind),
            "source_train_n_items": int(train_n),
            "mean_abs_match_error": _mean(_to_float(row.get("match_abs_error")) for row in rows),
            "passes_min_unique_source_pairs": True,
            "passes_min_size_pairs": True,
        }
    return {
        "n_control_rows": int(len(control_rows)),
        "n_size_items": int(len(size_items)),
        "n_metadata_rows": int(len(metadata_rows)),
        "n_size_train_rows": int(len(train_idx)),
        "source_summary": source_summary,
    }


def _result_row(
    *,
    row: Mapping[str, Any],
    basis_domain: str,
    method: str,
    layer: int,
    rank: int,
    repeat: int,
    alpha: float,
    expected_sign: float,
    base_ordered: float,
    patched_ordered: float,
    effect: float,
    aligned_effect: float,
    direction_match: int,
    patched_entropy: float,
    delta_norm: float,
    projected_delta_norm: float,
    full_aligned_effect: float,
    basis_train_kind: str,
    basis_train_n_items: int,
    basis_train_n_prompts: int,
    basis_train_n_pairs: int,
    subspace_overlap_with_size: float,
    max_principal_angle_deg: float,
) -> Dict[str, Any]:
    return {
        "size_pair_id": str(row["size_pair_id"]),
        "size_direction": str(row["size_direction"]),
        "size_donor_side": str(row["size_donor_side"]),
        "size_recv_side": str(row["size_recv_side"]),
        "basis_domain": str(basis_domain),
        "delta_domain": "size",
        "method": str(method),
        "layer": int(layer),
        "patch_site": "final_prompt_token",
        "rank": int(rank),
        "repeat": int(repeat),
        "alpha": float(alpha),
        "expected_sign": float(expected_sign),
        "base_ordered_score": float(base_ordered),
        "patched_ordered_score": float(patched_ordered),
        "effect": float(effect),
        "aligned_effect": float(aligned_effect),
        "direction_match": int(direction_match),
        "patched_entropy": float(patched_entropy),
        "delta_norm": float(delta_norm),
        "projected_delta_norm": float(projected_delta_norm),
        "projected_norm_fraction": float(projected_delta_norm / max(delta_norm, 1e-9)),
        "size_delta_raw": _to_float(row.get("size_delta_raw")),
        "size_delta_z": _to_float(row.get("size_delta_z")),
        "source_match_domain": str(row.get("source_domain", "")),
        "source_match_pair_id": str(row.get("source_pair_id", "")),
        "source_match_direction": str(row.get("source_direction", "")),
        "source_match_delta_raw": _to_float(row.get("source_delta_raw")),
        "source_match_delta_z": _to_float(row.get("source_delta_z")),
        "match_abs_error": _to_float(row.get("match_abs_error")),
        "match_mode": str(row.get("match_mode", "")),
        "full_size_aligned_effect": float(full_aligned_effect),
        "recovery_vs_size_full": (
            float(aligned_effect) / float(full_aligned_effect)
            if _is_finite(full_aligned_effect) and abs(float(full_aligned_effect)) > 1e-9
            else float("nan")
        ),
        "basis_train_kind": str(basis_train_kind),
        "basis_train_n_items": int(basis_train_n_items),
        "basis_train_n_prompts": int(basis_train_n_prompts),
        "basis_train_n_pairs": int(basis_train_n_pairs),
        "subspace_overlap_with_size": float(subspace_overlap_with_size),
        "max_principal_angle_deg": float(max_principal_angle_deg),
    }


@torch.no_grad()
def run_subspace_transfer(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    size_items: Sequence[DisambPair],
    control_rows: Sequence[Mapping[str, Any]],
    source_prompt_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_delta_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_basis_mode: str,
    normalize_source_deltas: bool,
    npz_path: Path,
    metadata_csv: Path,
    size_train_variant: str,
    layers: Sequence[int],
    rank: int,
    alphas: Sequence[float],
    random_repeats: int,
    device: torch.device,
    normalize_by_length: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata_rows = _read_csv(metadata_csv)
    arrs = np.load(npz_path)
    item_by_id = {str(item.pair_id): item for item in size_items}
    controls_by_size_key: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for row in control_rows:
        controls_by_size_key.setdefault(_size_key(row), row)

    cache = ActivationCache(model=model, tokenizer=tokenizer, device=device)
    rows_out: List[Dict[str, Any]] = []
    basis_summary: Dict[str, Any] = {}

    for layer in layers:
        size_basis, size_basis_meta = _size_basis_for_layer(
            arrs=arrs,
            metadata_rows=metadata_rows,
            size_train_variant=size_train_variant,
            layer=int(layer),
            rank=int(rank),
            seed=int(seed) + int(layer) * 997,
        )
        source_bases: Dict[str, np.ndarray] = {}
        source_basis_meta: Dict[str, Dict[str, Any]] = {}
        source_alignments: Dict[str, Dict[str, Any]] = {}
        random_bases: Dict[str, List[np.ndarray]] = {}
        for source_domain, prompts in sorted(source_prompt_rows.items()):
            if str(source_basis_mode) == "states":
                source_basis, meta = _source_basis_for_layer(
                    cache=cache,
                    prompt_rows=prompts,
                    layer=int(layer),
                    rank=int(rank),
                )
            elif str(source_basis_mode) == "deltas":
                source_basis, meta = _source_delta_basis_for_layer(
                    cache=cache,
                    delta_rows=source_delta_rows[str(source_domain)],
                    layer=int(layer),
                    rank=int(rank),
                    normalize_deltas=bool(normalize_source_deltas),
                )
            else:
                raise ValueError(f"Unknown source_basis_mode {source_basis_mode!r}")
            source_bases[str(source_domain)] = source_basis
            source_basis_meta[str(source_domain)] = meta
            source_alignments[str(source_domain)] = _subspace_alignment(size_basis, source_basis)
            random_bases[str(source_domain)] = [
                _random_basis(
                    int(source_basis.shape[0]),
                    int(rank),
                    seed=int(seed) + int(layer) * 7919 + repeat * 10007 + _stable_int(str(source_domain)) % 997,
                )
                for repeat in range(int(random_repeats))
            ]
        basis_summary[f"L{int(layer)}"] = {
            "size": size_basis_meta,
            "sources": {
                str(domain): {
                    **source_basis_meta[str(domain)],
                    "alignment_with_size": source_alignments[str(domain)],
                }
                for domain in sorted(source_bases)
            },
        }

        full_effect_by_size_key: Dict[Tuple[str, str, str], float] = {}
        base_ordered_by_size_key: Dict[Tuple[str, str, str], float] = {}
        size_delta_by_key: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor, int, float]] = {}

        for size_key, row in sorted(controls_by_size_key.items()):
            pair_id, donor_side, recv_side = size_key
            item = item_by_id[str(pair_id)]
            donor = _side(item, donor_side)
            recv = _side(item, recv_side)
            expected_sign = float(row.get("expected_sign", 1.0))
            donor_vec, _, _ = cache.get(donor.prompt, layer=int(layer))
            recv_vec, recv_token_idx, _ = cache.get(recv.prompt, layer=int(layer))
            delta_size = donor_vec - recv_vec
            delta_size_norm = float(torch.linalg.vector_norm(delta_size).item())
            size_delta_by_key[size_key] = (delta_size, recv_vec, int(recv_token_idx), float(delta_size_norm))
            base_scores = score_labels_next_continuations(
                model,
                tokenizer,
                recv.prompt,
                item.choices,
                device,
                normalize_by_length=normalize_by_length,
            )
            base_ordered = _ordered_score(base_scores)
            base_ordered_by_size_key[size_key] = float(base_ordered)

            full_patched, full_effect, full_aligned, full_match, full_entropy = _score_replacement(
                model=model,
                tokenizer=tokenizer,
                item=item,
                recv=recv,
                recv_token_idx=int(recv_token_idx),
                layer=int(layer),
                replacement=donor_vec,
                base_ordered=base_ordered,
                expected_sign=expected_sign,
                device=device,
                normalize_by_length=normalize_by_length,
            )
            full_effect_by_size_key[size_key] = float(full_aligned)
            rows_out.append(
                _result_row(
                    row=row,
                    basis_domain="size_in_domain",
                    method="full_size",
                    layer=int(layer),
                    rank=int(delta_size.numel()),
                    repeat=0,
                    alpha=1.0,
                    expected_sign=expected_sign,
                    base_ordered=base_ordered,
                    patched_ordered=full_patched,
                    effect=full_effect,
                    aligned_effect=full_aligned,
                    direction_match=full_match,
                    patched_entropy=full_entropy,
                    delta_norm=delta_size_norm,
                    projected_delta_norm=delta_size_norm,
                    full_aligned_effect=full_aligned,
                    basis_train_kind=str(size_basis_meta["basis_train_kind"]),
                    basis_train_n_items=int(size_basis_meta["n_train_items"]),
                    basis_train_n_prompts=int(size_basis_meta["n_train_prompts"]),
                    basis_train_n_pairs=int(size_basis_meta["n_train_pairs"]),
                    subspace_overlap_with_size=1.0,
                    max_principal_angle_deg=0.0,
                )
            )
            sham_patched, sham_effect, sham_aligned, sham_match, sham_entropy = _score_replacement(
                model=model,
                tokenizer=tokenizer,
                item=item,
                recv=recv,
                recv_token_idx=int(recv_token_idx),
                layer=int(layer),
                replacement=recv_vec,
                base_ordered=base_ordered,
                expected_sign=expected_sign,
                device=device,
                normalize_by_length=normalize_by_length,
            )
            rows_out.append(
                _result_row(
                    row=row,
                    basis_domain="size_in_domain",
                    method="sham",
                    layer=int(layer),
                    rank=0,
                    repeat=0,
                    alpha=1.0,
                    expected_sign=expected_sign,
                    base_ordered=base_ordered,
                    patched_ordered=sham_patched,
                    effect=sham_effect,
                    aligned_effect=sham_aligned,
                    direction_match=sham_match,
                    patched_entropy=sham_entropy,
                    delta_norm=0.0,
                    projected_delta_norm=0.0,
                    full_aligned_effect=full_aligned,
                    basis_train_kind=str(size_basis_meta["basis_train_kind"]),
                    basis_train_n_items=int(size_basis_meta["n_train_items"]),
                    basis_train_n_prompts=int(size_basis_meta["n_train_prompts"]),
                    basis_train_n_pairs=int(size_basis_meta["n_train_pairs"]),
                    subspace_overlap_with_size=1.0,
                    max_principal_angle_deg=0.0,
                )
            )
            projected_size = _project_delta(delta_size, size_basis, device=device)
            projected_size_norm = float(torch.linalg.vector_norm(projected_size).item())
            for alpha in alphas:
                patched, effect, aligned, match, entropy = _score_replacement(
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    recv=recv,
                    recv_token_idx=int(recv_token_idx),
                    layer=int(layer),
                    replacement=recv_vec + float(alpha) * projected_size,
                    base_ordered=base_ordered,
                    expected_sign=expected_sign,
                    device=device,
                    normalize_by_length=normalize_by_length,
                )
                rows_out.append(
                    _result_row(
                        row=row,
                        basis_domain="size_in_domain",
                        method="size_basis_pca",
                        layer=int(layer),
                        rank=int(rank),
                        repeat=0,
                        alpha=float(alpha),
                        expected_sign=expected_sign,
                        base_ordered=base_ordered,
                        patched_ordered=patched,
                        effect=effect,
                        aligned_effect=aligned,
                        direction_match=match,
                        patched_entropy=entropy,
                        delta_norm=delta_size_norm,
                        projected_delta_norm=projected_size_norm,
                        full_aligned_effect=full_aligned,
                        basis_train_kind=str(size_basis_meta["basis_train_kind"]),
                        basis_train_n_items=int(size_basis_meta["n_train_items"]),
                        basis_train_n_prompts=int(size_basis_meta["n_train_prompts"]),
                        basis_train_n_pairs=int(size_basis_meta["n_train_pairs"]),
                        subspace_overlap_with_size=1.0,
                        max_principal_angle_deg=0.0,
                    )
                )

        for row in control_rows:
            size_key = _size_key(row)
            pair_id, _, recv_side = size_key
            item = item_by_id[str(pair_id)]
            recv = _side(item, recv_side)
            delta_size, recv_vec, recv_token_idx, delta_size_norm = size_delta_by_key[size_key]
            expected_sign = float(row.get("expected_sign", 1.0))
            base_ordered = float(base_ordered_by_size_key[size_key])
            full_aligned = float(full_effect_by_size_key[size_key])
            domain = str(row["source_domain"])
            source_basis = source_bases[domain]
            source_meta = source_basis_meta[domain]
            align = source_alignments[domain]
            projected = _project_delta(delta_size, source_basis, device=device)
            projected_norm = float(torch.linalg.vector_norm(projected).item())
            for alpha in alphas:
                patched, effect, aligned, match, entropy = _score_replacement(
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    recv=recv,
                    recv_token_idx=int(recv_token_idx),
                    layer=int(layer),
                    replacement=recv_vec + float(alpha) * projected,
                    base_ordered=base_ordered,
                    expected_sign=expected_sign,
                    device=device,
                    normalize_by_length=normalize_by_length,
                )
                rows_out.append(
                    _result_row(
                        row=row,
                        basis_domain=domain,
                        method="source_basis_pca",
                        layer=int(layer),
                        rank=int(rank),
                        repeat=0,
                        alpha=float(alpha),
                        expected_sign=expected_sign,
                        base_ordered=base_ordered,
                        patched_ordered=patched,
                        effect=effect,
                        aligned_effect=aligned,
                        direction_match=match,
                        patched_entropy=entropy,
                        delta_norm=delta_size_norm,
                        projected_delta_norm=projected_norm,
                        full_aligned_effect=full_aligned,
                        basis_train_kind=str(source_meta["basis_train_kind"]),
                        basis_train_n_items=int(source_meta["n_train_items"]),
                        basis_train_n_prompts=int(source_meta["n_train_prompts"]),
                        basis_train_n_pairs=int(source_meta["n_train_pairs"]),
                        subspace_overlap_with_size=float(align["mean_squared_cosine"]),
                        max_principal_angle_deg=float(align["max_principal_angle_deg"]),
                    )
                )
                for repeat, random_basis in enumerate(random_bases[domain]):
                    projected_random = _project_delta(delta_size, random_basis, device=device)
                    random_norm = float(torch.linalg.vector_norm(projected_random).item())
                    if random_norm > 1e-9 and projected_norm > 0.0:
                        projected_random = projected_random * float(projected_norm / random_norm)
                        random_norm = projected_norm
                    patched_r, effect_r, aligned_r, match_r, entropy_r = _score_replacement(
                        model=model,
                        tokenizer=tokenizer,
                        item=item,
                        recv=recv,
                        recv_token_idx=int(recv_token_idx),
                        layer=int(layer),
                        replacement=recv_vec + float(alpha) * projected_random,
                        base_ordered=base_ordered,
                        expected_sign=expected_sign,
                        device=device,
                        normalize_by_length=normalize_by_length,
                    )
                    rows_out.append(
                        _result_row(
                            row=row,
                            basis_domain=domain,
                            method="source_basis_random_norm_matched",
                            layer=int(layer),
                            rank=int(rank),
                            repeat=int(repeat),
                            alpha=float(alpha),
                            expected_sign=expected_sign,
                            base_ordered=base_ordered,
                            patched_ordered=patched_r,
                            effect=effect_r,
                            aligned_effect=aligned_r,
                            direction_match=match_r,
                            patched_entropy=entropy_r,
                            delta_norm=delta_size_norm,
                            projected_delta_norm=random_norm,
                            full_aligned_effect=full_aligned,
                            basis_train_kind=str(source_meta["basis_train_kind"]),
                            basis_train_n_items=int(source_meta["n_train_items"]),
                            basis_train_n_prompts=int(source_meta["n_train_prompts"]),
                            basis_train_n_pairs=int(source_meta["n_train_pairs"]),
                            subspace_overlap_with_size=float("nan"),
                            max_principal_angle_deg=float("nan"),
                        )
                    )
    return rows_out, basis_summary


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_b: int,
    seed: int,
    min_unique_source_pairs: int,
    min_size_pairs: int,
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, int, int, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["basis_domain"]),
                str(row["method"]),
                int(row["layer"]),
                int(row["rank"]),
                float(row["alpha"]),
            )
        ].append(row)
    out_groups: Dict[str, Any] = {}
    for (basis_domain, method, layer, rank, alpha), vals in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1], kv[0][3], kv[0][4])):
        source_pairs = {str(row["source_match_pair_id"]) for row in vals if str(row["basis_domain"]) != "size_in_domain"}
        size_pairs = {str(row["size_pair_id"]) for row in vals}
        match_errors = [_to_float(row.get("match_abs_error")) for row in vals if _is_finite(_to_float(row.get("match_abs_error")))]
        overlaps = [_to_float(row.get("subspace_overlap_with_size")) for row in vals if _is_finite(_to_float(row.get("subspace_overlap_with_size")))]
        max_angles = [_to_float(row.get("max_principal_angle_deg")) for row in vals if _is_finite(_to_float(row.get("max_principal_angle_deg")))]
        key = f"L{layer}_{basis_domain}_{method}_rank{rank}_alpha{alpha:g}"
        out_groups[key] = {
            "basis_domain": basis_domain,
            "method": method,
            "layer": int(layer),
            "rank": int(rank),
            "alpha": float(alpha),
            "n": int(len(vals)),
            "n_size_pairs": int(len(size_pairs)),
            "n_unique_source_pairs": int(len(source_pairs)),
            "basis_train_kind": str(next((str(row.get("basis_train_kind", "")) for row in vals if str(row.get("basis_train_kind", ""))), "")),
            "basis_train_n_items": int(max(int(row.get("basis_train_n_items", 0)) for row in vals)),
            "basis_train_n_prompts": int(max(int(row.get("basis_train_n_prompts", 0)) for row in vals)),
            "basis_train_n_pairs": int(max(int(row.get("basis_train_n_pairs", 0)) for row in vals)),
            "mean_abs_match_error": _mean(match_errors),
            "subspace_overlap_with_size": _mean(overlaps),
            "max_principal_angle_deg": _mean(max_angles),
            "passes_min_n": bool(
                len(size_pairs) >= int(min_size_pairs)
                and (basis_domain == "size_in_domain" or len(source_pairs) >= int(min_unique_source_pairs))
            ),
            "aligned_effect": _cluster_bootstrap_mean_ci(
                vals,
                value_key="aligned_effect",
                cluster_key="size_pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 11 + int(rank),
            ),
            "recovery_vs_size_full": _cluster_bootstrap_mean_ci(
                vals,
                value_key="recovery_vs_size_full",
                cluster_key="size_pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 17 + int(rank),
            ),
            "recovery_vs_size_full_median": _quantile([float(row["recovery_vs_size_full"]) for row in vals], 0.5),
            "projected_norm_fraction": _cluster_bootstrap_mean_ci(
                vals,
                value_key="projected_norm_fraction",
                cluster_key="size_pair_id",
                b=bootstrap_b,
                seed=seed + int(layer) * 23 + int(rank),
            ),
            "projected_norm_fraction_median": _quantile([float(row["projected_norm_fraction"]) for row in vals], 0.5),
            "direction_match_rate": _mean(float(row["direction_match"]) for row in vals),
        }
    return {
        "schema_version": "gradable_cross_domain_subspace_transfer_v1",
        "n_rows": int(len(rows)),
        "groups": out_groups,
    }


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    source_basis_mode = str(summary["run"].get("source_basis_mode", "states"))
    title = (
        "# Gradable Cross-Domain Source-Delta-Basis Transfer"
        if source_basis_mode == "deltas"
        else "# Gradable Cross-Domain Source-Basis Subspace Transfer"
    )
    lines = [
        title,
        "",
        f"- Size train variant for positive control: `{summary['run']['size_train_variant']}`",
        f"- Size eval data: `{summary['run']['size_data_path']}`",
        f"- Controls: `{summary['run']['controls_csv']}`",
        f"- Source domains: `{summary['run']['source_domains']}`",
        f"- Source basis mode: `{source_basis_mode}`",
        f"- Layers: `{summary['run']['layers']}`",
        f"- Rank: `{summary['run']['rank']}`",
        "",
        "| layer | basis | method | rank | alpha | n | size pairs | source pairs | train kind | train items | overlap | max angle | aligned effect | recovery/size-full | norm frac | dir match | min-n |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for group in summary["groups"].values():
        ae = group["aligned_effect"]
        rvf = group["recovery_vs_size_full"]
        nf = group["projected_norm_fraction"]
        overlap = group["subspace_overlap_with_size"]
        max_angle = group["max_principal_angle_deg"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["layer"]),
                    str(group["basis_domain"]),
                    str(group["method"]),
                    str(group["rank"]),
                    f"{float(group['alpha']):g}",
                    str(group["n"]),
                    str(group["n_size_pairs"]),
                    str(group["n_unique_source_pairs"]),
                    str(group["basis_train_kind"]),
                    str(group["basis_train_n_items"]),
                    f"{float(overlap):.3f}" if _is_finite(float(overlap)) else "NA",
                    f"{float(max_angle):.1f}" if _is_finite(float(max_angle)) else "NA",
                    f"{float(ae['mean']):.3f} [{float(ae['lo']):.3f}, {float(ae['hi']):.3f}]",
                    f"{float(rvf['mean']):.3f} [{float(rvf['lo']):.3f}, {float(rvf['hi']):.3f}]",
                    f"{float(nf['mean']):.3f}",
                    f"{float(group['direction_match_rate']):.3f}",
                    "yes" if bool(group["passes_min_n"]) else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Decision discipline:",
            "- `size_in_domain/full_size` is the raw size full-vector upper bound for the same size receiver directions.",
            "- `size_in_domain/size_basis_pca` is the positive L20 size-basis control.",
            "- `temperature/source_basis_pca` and `age/source_basis_pca` fit PCA bases on source-domain states or source-domain pair deltas, then project in-domain size deltas through those source-trained bases.",
            "- A shared gradable-calibration subspace predicts positive source-basis transfer that beats norm-matched random controls.",
            "- In `deltas` mode, source pair deltas are oriented from higher scalar coordinate to lower scalar coordinate and SVD is uncentered so the mean contrast direction is retained.",
            "- Subspace overlap is diagnostic only; the causal patching metric is the gate.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit source-domain PCA bases and evaluate them on size causal deltas.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument(
        "--size_data_path",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl"),
    )
    p.add_argument(
        "--controls_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_matched_delta_rho_controls.csv"),
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
    p.add_argument("--size_train_variant", type=str, default="fictional_semantic_adjective_counts")
    p.add_argument("--source_domains", type=str, default="temperature,age")
    p.add_argument("--source_basis_mode", type=str, default="states", choices=["states", "deltas"])
    p.add_argument("--normalize_source_deltas", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--layers", type=str, default="20")
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--alphas", type=str, default="1.0")
    p.add_argument("--random_repeats", type=int, default=20)
    p.add_argument("--min_unique_source_pairs", type=int, default=20)
    p.add_argument("--min_size_pairs", type=int, default=20)
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
        default=str(DEFAULT_ROOT / "gradable_cross_domain_subspace_transfer_l20_r5_gemma3.csv"),
    )
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_subspace_transfer_l20_r5_gemma3.summary.json"),
    )
    p.add_argument(
        "--out_md",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_subspace_transfer_l20_r5_gemma3.md"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Validate inputs without loading the model.")
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
    alphas = _parse_float_list(str(args.alphas))
    source_domains = _parse_csv_or_space(str(args.source_domains))
    control_rows = _load_control_rows(Path(str(args.controls_csv)), source_domains=source_domains)
    size_items = load_disamb_pairs(str(args.size_data_path))
    source_paths = _unique_source_paths(control_rows)
    source_items = {domain: load_disamb_pairs(str(path)) for domain, path in source_paths.items()}
    source_prompt_rows = {
        domain: _unique_prompt_rows(items, domain=domain, data_path=source_paths[domain])
        for domain, items in source_items.items()
    }
    source_delta_rows = {
        domain: _source_delta_rows(items, domain=domain, data_path=source_paths[domain])
        for domain, items in source_items.items()
    }
    metadata_rows = _read_csv(Path(str(args.metadata_csv)))
    arrs = np.load(Path(str(args.activations_npz)))
    validation = _validate_inputs(
        control_rows=control_rows,
        size_items=size_items,
        source_prompt_rows=source_prompt_rows,
        source_delta_rows=source_delta_rows,
        source_basis_mode=str(args.source_basis_mode),
        metadata_rows=metadata_rows,
        arrs=arrs,
        size_train_variant=str(args.size_train_variant),
        layers=layers,
        rank=int(args.rank),
        min_unique_source_pairs=int(args.min_unique_source_pairs),
        min_size_pairs=int(args.min_size_pairs),
    )
    if bool(args.dry_run):
        print(json.dumps({"dry_run": True, "validation": validation}, indent=2, sort_keys=True))
        return

    out_csv = Path(str(args.out_csv))
    out_summary = Path(str(args.out_summary))
    out_md = Path(str(args.out_md))
    if (out_csv.exists() or out_summary.exists() or out_md.exists()) and not bool(args.overwrite):
        raise FileExistsError("Output exists. Use --overwrite to replace source-basis transfer outputs.")

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
    rows, basis_summary = run_subspace_transfer(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        size_items=size_items,
        control_rows=control_rows,
        source_prompt_rows=source_prompt_rows,
        source_delta_rows=source_delta_rows,
        source_basis_mode=str(args.source_basis_mode),
        normalize_source_deltas=bool(args.normalize_source_deltas),
        npz_path=Path(str(args.activations_npz)),
        metadata_csv=Path(str(args.metadata_csv)),
        size_train_variant=str(args.size_train_variant),
        layers=layers,
        rank=int(args.rank),
        alphas=alphas,
        random_repeats=int(args.random_repeats),
        device=device,
        normalize_by_length=not bool(args.no_length_norm),
        seed=int(args.seed),
    )
    summary = summarize_rows(
        rows,
        bootstrap_b=int(args.bootstrap_B),
        seed=int(args.seed),
        min_unique_source_pairs=int(args.min_unique_source_pairs),
        min_size_pairs=int(args.min_size_pairs),
    )
    summary["validation"] = validation
    summary["basis_summary"] = basis_summary
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "size_data_path": str(args.size_data_path),
        "controls_csv": str(args.controls_csv),
        "source_paths": {str(k): str(v) for k, v in source_paths.items()},
        "activations_npz": str(args.activations_npz),
        "metadata_csv": str(args.metadata_csv),
        "size_train_variant": str(args.size_train_variant),
        "source_domains": str(args.source_domains),
        "source_basis_mode": str(args.source_basis_mode),
        "normalize_source_deltas": bool(args.normalize_source_deltas),
        "layers": str(args.layers),
        "rank": int(args.rank),
        "alphas": str(args.alphas),
        "random_repeats": int(args.random_repeats),
        "min_unique_source_pairs": int(args.min_unique_source_pairs),
        "min_size_pairs": int(args.min_size_pairs),
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
    print(f"Wrote source-basis transfer CSV: {out_csv}")
    print(f"Wrote source-basis transfer summary: {out_summary}")
    print(f"Wrote source-basis transfer Markdown: {out_md}")


if __name__ == "__main__":
    main()
