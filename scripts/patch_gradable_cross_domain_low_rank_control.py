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
from aom.metrics.disamb import _encode_prompt, score_labels_next_continuations, score_labels_next_continuations_patched
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed
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
    _softmax,
    _to_float,
    _write_csv,
)


DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"


def _parse_csv_or_space(raw: str) -> List[str]:
    return [part.strip() for part in re.split(r"[,\s]+", str(raw)) if part.strip()]


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


def _validate_inputs(
    *,
    control_rows: Sequence[Mapping[str, Any]],
    size_items: Sequence[DisambPair],
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
        source_summary[domain] = {
            "n_rows": int(len(rows)),
            "n_unique_source_pairs": int(len(unique_source_pairs)),
            "n_unique_size_pairs": int(len(unique_size_pairs)),
            "mean_abs_match_error": _mean(_to_float(row.get("match_abs_error")) for row in rows),
            "passes_min_unique_source_pairs": bool(len(unique_source_pairs) >= int(min_unique_source_pairs)),
            "passes_min_size_pairs": bool(len(unique_size_pairs) >= int(min_size_pairs)),
        }
        if len(unique_source_pairs) < int(min_unique_source_pairs):
            raise ValueError(
                f"{domain}: only {len(unique_source_pairs)} unique source pairs; "
                f"minimum is {int(min_unique_source_pairs)}"
            )
        if len(unique_size_pairs) < int(min_size_pairs):
            raise ValueError(f"{domain}: only {len(unique_size_pairs)} size pairs; minimum is {int(min_size_pairs)}")
    return {
        "n_control_rows": int(len(control_rows)),
        "n_size_items": int(len(size_items)),
        "n_metadata_rows": int(len(metadata_rows)),
        "n_size_train_rows": int(len(train_idx)),
        "source_summary": source_summary,
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


def _basis_for_layer(
    *,
    arrs: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    size_train_variant: str,
    layer: int,
    rank: int,
    seed: int,
) -> np.ndarray:
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
            return np.asarray(basis_info["basis"], dtype=np.float32)
    raise ValueError(f"Could not build pca rank {rank} for layer {layer}")


@torch.no_grad()
def _score_replacement(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    item: DisambPair,
    recv: PromptSide,
    recv_token_idx: int,
    layer: int,
    replacement: torch.Tensor,
    base_ordered: float,
    expected_sign: float,
    device: torch.device,
    normalize_by_length: bool,
) -> Tuple[float, float, float, int, float]:
    patched_scores = score_labels_next_continuations_patched(
        model,
        tokenizer,
        recv.prompt,
        item.choices,
        device,
        patch_site=PatchSpanSite(layer=int(layer), token_indices=(int(recv_token_idx),)),
        replacement=replacement,
        normalize_by_length=normalize_by_length,
    )
    patched_ordered = _ordered_score(patched_scores)
    patched_probs = _softmax(patched_scores.by_label)
    effect = float(patched_ordered - base_ordered)
    aligned_effect = float(effect * float(expected_sign))
    entropy = 0.0
    for prob in patched_probs.values():
        p = max(float(prob), 1e-12)
        entropy -= p * math.log(p)
    return patched_ordered, effect, aligned_effect, int(aligned_effect > 0.0), float(entropy)


def _common_row(
    *,
    row: Mapping[str, Any],
    source_domain: str,
    source_pair_id: str,
    source_direction: str,
    source_donor_side: str,
    source_recv_side: str,
    source_delta_raw: float,
    source_delta_z: float,
    match_abs_error: float,
    layer: int,
    method: str,
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
) -> Dict[str, Any]:
    return {
        "control_id": str(row.get("control_id", "")),
        "size_pair_id": str(row["size_pair_id"]),
        "size_direction": str(row["size_direction"]),
        "size_donor_side": str(row["size_donor_side"]),
        "size_recv_side": str(row["size_recv_side"]),
        "source_domain": str(source_domain),
        "source_pair_id": str(source_pair_id),
        "source_direction": str(source_direction),
        "source_donor_side": str(source_donor_side),
        "source_recv_side": str(source_recv_side),
        "layer": int(layer),
        "patch_site": "final_prompt_token",
        "method": str(method),
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
        "source_delta_raw": float(source_delta_raw),
        "source_delta_z": float(source_delta_z),
        "match_abs_error": float(match_abs_error),
        "match_mode": str(row.get("match_mode", "")),
        "full_size_aligned_effect": float(full_aligned_effect),
        "recovery_vs_size_full": (
            float(aligned_effect) / float(full_aligned_effect)
            if _is_finite(full_aligned_effect) and abs(float(full_aligned_effect)) > 1e-9
            else float("nan")
        ),
    }


@torch.no_grad()
def run_cross_domain_patch(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    size_items: Sequence[DisambPair],
    control_rows: Sequence[Mapping[str, Any]],
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
) -> List[Dict[str, Any]]:
    metadata_rows = _read_csv(metadata_csv)
    arrs = np.load(npz_path)
    item_by_id = {str(item.pair_id): item for item in size_items}
    controls_by_size_key: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for row in control_rows:
        controls_by_size_key.setdefault(_size_key(row), row)

    cache = ActivationCache(model=model, tokenizer=tokenizer, device=device)
    rows_out: List[Dict[str, Any]] = []
    full_effect_by_size_key: Dict[Tuple[str, str, str, int], float] = {}
    base_ordered_by_size_key: Dict[Tuple[str, str, str], float] = {}
    recv_token_idx_by_size_key_layer: Dict[Tuple[str, str, str, int], int] = {}

    for layer in layers:
        basis = _basis_for_layer(
            arrs=arrs,
            metadata_rows=metadata_rows,
            size_train_variant=size_train_variant,
            layer=int(layer),
            rank=int(rank),
            seed=int(seed) + int(layer) * 997,
        )
        random_bases = [
            _random_basis(int(basis.shape[0]), int(rank), seed=int(seed) + int(layer) * 7919 + repeat * 10007)
            for repeat in range(int(random_repeats))
        ]

        for size_key, row in sorted(controls_by_size_key.items()):
            pair_id, donor_side, recv_side = size_key
            item = item_by_id[str(pair_id)]
            donor = _side(item, donor_side)
            recv = _side(item, recv_side)
            expected_sign = float(row.get("expected_sign", 1.0))
            donor_vec, _, _ = cache.get(donor.prompt, layer=int(layer))
            recv_vec, recv_token_idx, _ = cache.get(recv.prompt, layer=int(layer))
            recv_token_idx_by_size_key_layer[(pair_id, donor_side, recv_side, int(layer))] = int(recv_token_idx)
            delta_size = donor_vec - recv_vec
            delta_size_norm = float(torch.linalg.vector_norm(delta_size).item())
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
            full_effect_by_size_key[(pair_id, donor_side, recv_side, int(layer))] = float(full_aligned)
            rows_out.append(
                _common_row(
                    row=row,
                    source_domain="size_in_domain",
                    source_pair_id=str(pair_id),
                    source_direction=str(row["size_direction"]),
                    source_donor_side=str(donor_side),
                    source_recv_side=str(recv_side),
                    source_delta_raw=_to_float(row.get("size_delta_raw")),
                    source_delta_z=_to_float(row.get("size_delta_z")),
                    match_abs_error=0.0,
                    layer=int(layer),
                    method="full_size",
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
                _common_row(
                    row=row,
                    source_domain="size_in_domain",
                    source_pair_id=str(pair_id),
                    source_direction=str(row["size_direction"]),
                    source_donor_side=str(donor_side),
                    source_recv_side=str(recv_side),
                    source_delta_raw=_to_float(row.get("size_delta_raw")),
                    source_delta_z=_to_float(row.get("size_delta_z")),
                    match_abs_error=0.0,
                    layer=int(layer),
                    method="sham",
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
                )
            )

            projected_size = _project_delta(delta_size, basis, device=device)
            projected_size_norm = float(torch.linalg.vector_norm(projected_size).item())
            for alpha in alphas:
                repl = recv_vec + float(alpha) * projected_size
                patched, effect, aligned, match, entropy = _score_replacement(
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    recv=recv,
                    recv_token_idx=int(recv_token_idx),
                    layer=int(layer),
                    replacement=repl,
                    base_ordered=base_ordered,
                    expected_sign=expected_sign,
                    device=device,
                    normalize_by_length=normalize_by_length,
                )
                rows_out.append(
                    _common_row(
                        row=row,
                        source_domain="size_in_domain",
                        source_pair_id=str(pair_id),
                        source_direction=str(row["size_direction"]),
                        source_donor_side=str(donor_side),
                        source_recv_side=str(recv_side),
                        source_delta_raw=_to_float(row.get("size_delta_raw")),
                        source_delta_z=_to_float(row.get("size_delta_z")),
                        match_abs_error=0.0,
                        layer=int(layer),
                        method="size_pca",
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
                    )
                )

        for row in control_rows:
            size_key = _size_key(row)
            pair_id, donor_side, recv_side = size_key
            item = item_by_id[str(pair_id)]
            recv = _side(item, recv_side)
            recv_vec, recv_token_idx, _ = cache.get(recv.prompt, layer=int(layer))
            source_donor_vec, _, _ = cache.get(str(row["source_donor_prompt"]), layer=int(layer))
            source_recv_vec, _, _ = cache.get(str(row["source_recv_prompt"]), layer=int(layer))
            delta_source = source_donor_vec - source_recv_vec
            delta_source_norm = float(torch.linalg.vector_norm(delta_source).item())
            projected_source = _project_delta(delta_source, basis, device=device)
            projected_source_norm = float(torch.linalg.vector_norm(projected_source).item())
            expected_sign = float(row.get("expected_sign", 1.0))
            base_ordered = float(base_ordered_by_size_key[size_key])
            full_aligned = float(full_effect_by_size_key[(pair_id, donor_side, recv_side, int(layer))])

            for alpha in alphas:
                repl = recv_vec + float(alpha) * projected_source
                patched, effect, aligned, match, entropy = _score_replacement(
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    recv=recv,
                    recv_token_idx=int(recv_token_idx),
                    layer=int(layer),
                    replacement=repl,
                    base_ordered=base_ordered,
                    expected_sign=expected_sign,
                    device=device,
                    normalize_by_length=normalize_by_length,
                )
                rows_out.append(
                    _common_row(
                        row=row,
                        source_domain=str(row["source_domain"]),
                        source_pair_id=str(row["source_pair_id"]),
                        source_direction=str(row["source_direction"]),
                        source_donor_side=str(row["source_donor_side"]),
                        source_recv_side=str(row["source_recv_side"]),
                        source_delta_raw=_to_float(row.get("source_delta_raw")),
                        source_delta_z=_to_float(row.get("source_delta_z")),
                        match_abs_error=_to_float(row.get("match_abs_error")),
                        layer=int(layer),
                        method="source_pca",
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
                        delta_norm=delta_source_norm,
                        projected_delta_norm=projected_source_norm,
                        full_aligned_effect=full_aligned,
                    )
                )
                for repeat, random_basis in enumerate(random_bases):
                    projected_random = _project_delta(delta_source, random_basis, device=device)
                    random_norm = float(torch.linalg.vector_norm(projected_random).item())
                    if random_norm > 1e-9 and projected_source_norm > 0.0:
                        projected_random = projected_random * float(projected_source_norm / random_norm)
                        random_norm = projected_source_norm
                    repl_random = recv_vec + float(alpha) * projected_random
                    patched_r, effect_r, aligned_r, match_r, entropy_r = _score_replacement(
                        model=model,
                        tokenizer=tokenizer,
                        item=item,
                        recv=recv,
                        recv_token_idx=int(recv_token_idx),
                        layer=int(layer),
                        replacement=repl_random,
                        base_ordered=base_ordered,
                        expected_sign=expected_sign,
                        device=device,
                        normalize_by_length=normalize_by_length,
                    )
                    rows_out.append(
                        _common_row(
                            row=row,
                            source_domain=str(row["source_domain"]),
                            source_pair_id=str(row["source_pair_id"]),
                            source_direction=str(row["source_direction"]),
                            source_donor_side=str(row["source_donor_side"]),
                            source_recv_side=str(row["source_recv_side"]),
                            source_delta_raw=_to_float(row.get("source_delta_raw")),
                            source_delta_z=_to_float(row.get("source_delta_z")),
                            match_abs_error=_to_float(row.get("match_abs_error")),
                            layer=int(layer),
                            method="source_random_norm_matched",
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
                            delta_norm=delta_source_norm,
                            projected_delta_norm=random_norm,
                            full_aligned_effect=full_aligned,
                        )
                    )
    return rows_out


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
                str(row["source_domain"]),
                str(row["method"]),
                int(row["layer"]),
                int(row["rank"]),
                float(row["alpha"]),
            )
        ].append(row)
    out_groups: Dict[str, Any] = {}
    for (source_domain, method, layer, rank, alpha), vals in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1], kv[0][3], kv[0][4])):
        source_pairs = {str(row["source_pair_id"]) for row in vals if str(row["source_domain"]) != "size_in_domain"}
        size_pairs = {str(row["size_pair_id"]) for row in vals}
        match_errors = [_to_float(row.get("match_abs_error")) for row in vals if _is_finite(_to_float(row.get("match_abs_error")))]
        key = f"L{layer}_{source_domain}_{method}_rank{rank}_alpha{alpha:g}"
        out_groups[key] = {
            "source_domain": source_domain,
            "method": method,
            "layer": int(layer),
            "rank": int(rank),
            "alpha": float(alpha),
            "n": int(len(vals)),
            "n_size_pairs": int(len(size_pairs)),
            "n_unique_source_pairs": int(len(source_pairs)),
            "mean_abs_match_error": _mean(match_errors),
            "max_abs_match_error": max(match_errors) if match_errors else float("nan"),
            "passes_min_n": bool(
                len(size_pairs) >= int(min_size_pairs)
                and (source_domain == "size_in_domain" or len(source_pairs) >= int(min_unique_source_pairs))
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
        "schema_version": "gradable_cross_domain_low_rank_control_v1",
        "n_rows": int(len(rows)),
        "groups": out_groups,
    }


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Gradable Cross-Domain Matched-Delta Low-Rank Control",
        "",
        f"- Size train variant: `{summary['run']['size_train_variant']}`",
        f"- Size eval data: `{summary['run']['size_data_path']}`",
        f"- Controls: `{summary['run']['controls_csv']}`",
        f"- Layers: `{summary['run']['layers']}`",
        f"- Rank: `{summary['run']['rank']}`",
        "",
        "| layer | source | method | rank | alpha | n | size pairs | source pairs | match err | aligned effect | recovery/size-full | norm frac | dir match | min-n |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for group in summary["groups"].values():
        ae = group["aligned_effect"]
        rvf = group["recovery_vs_size_full"]
        nf = group["projected_norm_fraction"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["layer"]),
                    str(group["source_domain"]),
                    str(group["method"]),
                    str(group["rank"]),
                    f"{float(group['alpha']):g}",
                    str(group["n"]),
                    str(group["n_size_pairs"]),
                    str(group["n_unique_source_pairs"]),
                    f"{float(group['mean_abs_match_error']):.3f}" if _is_finite(float(group["mean_abs_match_error"])) else "NA",
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
            "- `size_in_domain/full_size` is the raw size full-vector upper bound for the same receiver directions.",
            "- `size_in_domain/size_pca` is the positive low-rank size control.",
            "- `temperature/source_pca` and `age/source_pca` test whether matched signed-delta cross-domain activation deltas carry the same size-causal signal after projection through the size PCA subspace.",
            "- A size-specific manifold candidate predicts strong `size_pca` and near-zero cross-domain `source_pca`, with min-n passing.",
            "- This is not an SAE/CLT group explanation; it is a domain-specificity control for the causal low-rank candidate.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain matched-delta low-rank control for gradable size calibration.")
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
        default=str(DEFAULT_ROOT / "gradable_cross_domain_low_rank_control_l20_gemma3.csv"),
    )
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_low_rank_control_l20_gemma3.summary.json"),
    )
    p.add_argument(
        "--out_md",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_low_rank_control_l20_gemma3.md"),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Validate inputs and basis availability without loading the model.")
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
    metadata_rows = _read_csv(Path(str(args.metadata_csv)))
    arrs = np.load(Path(str(args.activations_npz)))
    validation = _validate_inputs(
        control_rows=control_rows,
        size_items=size_items,
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
        raise FileExistsError("Output exists. Use --overwrite to replace cross-domain patch outputs.")

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
    rows = run_cross_domain_patch(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        size_items=size_items,
        control_rows=control_rows,
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
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "size_data_path": str(args.size_data_path),
        "controls_csv": str(args.controls_csv),
        "activations_npz": str(args.activations_npz),
        "metadata_csv": str(args.metadata_csv),
        "size_train_variant": str(args.size_train_variant),
        "source_domains": str(args.source_domains),
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
    print(f"Wrote cross-domain low-rank CSV: {out_csv}")
    print(f"Wrote cross-domain low-rank summary: {out_summary}")
    print(f"Wrote cross-domain low-rank Markdown: {out_md}")


if __name__ == "__main__":
    main()
