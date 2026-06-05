from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
from aom.token_spans import token_span_for_substring
from aom.utils import configure_logprob_computation, get_best_device, set_seed


LABEL_ORDER = ("tiny", "small", "large", "huge")


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one layer")
    return vals


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


def _prob_columns(prefix: str, probs: Mapping[str, float], *, label_order: Sequence[str] = LABEL_ORDER) -> Dict[str, float]:
    return {f"{prefix}_prob_{label}": float(probs.get(str(label), 0.0)) for label in label_order}


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
    raw = md.get(f"ratio_{side_name}", None)
    if raw is None:
        raise ValueError(f"{item.pair_id}: missing ratio_{side_name} metadata")
    return float(raw)


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
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


def _bootstrap_mean_ci(xs: Sequence[float], *, b: int, seed: int) -> Dict[str, float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(int(seed))
    boot: List[float] = []
    n = len(vals)
    for _ in range(int(b)):
        boot.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "mean": _mean(vals),
        "lo": _quantile(boot, 0.025),
        "hi": _quantile(boot, 0.975),
        "n": int(n),
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


@torch.no_grad()
def run_raw_size_patching(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: Sequence[DisambPair],
    device: torch.device,
    layers: Sequence[int],
    patch_site_name: str,
    normalize_by_length: bool,
    require_token_id_match: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                if str(patch_site_name) == "target":
                    donor_span, donor_token_ids = token_span_for_substring(
                        tokenizer,
                        donor.prompt,
                        item.target,
                        int(item.target_occurrence),
                    )
                    recv_span, recv_token_ids = token_span_for_substring(
                        tokenizer,
                        recv.prompt,
                        item.target,
                        int(item.target_occurrence),
                    )
                elif str(patch_site_name) == "final_prompt_token":
                    donor_span = (int(donor_ids.shape[-1]) - 1,)
                    recv_span = (int(recv_ids.shape[-1]) - 1,)
                    donor_token_ids = [int(donor_ids[0, donor_span[0]].item())]
                    recv_token_ids = [int(recv_ids[0, recv_span[0]].item())]
                else:
                    raise ValueError(f"Unknown patch site {patch_site_name!r}")
                if len(donor_span) != len(recv_span) or (require_token_id_match and donor_token_ids != recv_token_ids):
                    rows.append(
                        {
                            "pair_id": str(item.pair_id),
                            "direction": f"{donor_name}_to_{recv_name}",
                            "patch_site": str(patch_site_name),
                            "skipped": 1,
                            "skip_reason": "span_or_token_mismatch",
                            "donor_ratio": donor_ratio,
                            "recv_ratio": recv_ratio,
                            "predictor_delta": predictor_delta,
                        }
                    )
                    continue

                base_scores = score_labels_next_continuations(
                    model,
                    tokenizer,
                    recv.prompt,
                    item.choices,
                    device,
                    normalize_by_length=normalize_by_length,
                )
                base_ordered = _ordered_score(base_scores)
                base_probs = _softmax(base_scores.by_label)
                donor_scores = score_labels_next_continuations(
                    model,
                    tokenizer,
                    donor.prompt,
                    item.choices,
                    device,
                    normalize_by_length=normalize_by_length,
                )
                donor_ordered = _ordered_score(donor_scores)
                donor_probs = _softmax(donor_scores.by_label)
                raw_gap = float(donor_ordered - base_ordered)
                aligned_raw_gap = float(raw_gap * expected_sign)
                donor_blocks = get_block_outputs(model, donor_ids, layers=layers)
                recv_blocks = get_block_outputs(model, recv_ids, layers=layers)

                for layer in layers:
                    replacement = donor_blocks[int(layer)][0, donor_span, :].detach()
                    patched_scores = score_labels_next_continuations_patched(
                        model,
                        tokenizer,
                        recv.prompt,
                        item.choices,
                        device,
                        patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(recv_span)),
                        replacement=replacement,
                        normalize_by_length=normalize_by_length,
                    )
                    sham_replacement = recv_blocks[int(layer)][0, recv_span, :].detach()
                    sham_scores = score_labels_next_continuations_patched(
                        model,
                        tokenizer,
                        recv.prompt,
                        item.choices,
                        device,
                        patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(recv_span)),
                        replacement=sham_replacement,
                        normalize_by_length=normalize_by_length,
                    )
                    patched_ordered = _ordered_score(patched_scores)
                    patched_probs = _softmax(patched_scores.by_label)
                    sham_ordered = _ordered_score(sham_scores)
                    sham_probs = _softmax(sham_scores.by_label)
                    effect = float(patched_ordered - base_ordered)
                    sham_effect = float(sham_ordered - base_ordered)
                    aligned_effect = float(effect * expected_sign)
                    sham_aligned_effect = float(sham_effect * expected_sign)
                    patch_fraction = float(aligned_effect / max(abs(aligned_raw_gap), 1e-9))
                    kl_patched_to_donor = _kl_div(patched_probs, donor_probs)
                    kl_base_to_donor = _kl_div(base_probs, donor_probs)
                    kl_sham_to_donor = _kl_div(sham_probs, donor_probs)
                    rows.append(
                        {
                            "pair_id": str(item.pair_id),
                            "direction": f"{donor_name}_to_{recv_name}",
                            "layer": int(layer),
                            "patch_site": str(patch_site_name),
                            "skipped": 0,
                            "target": str(item.target),
                            "regime": str((item.metadata or {}).get("regime", "")),
                            "context_type": str((item.metadata or {}).get("context_type", "")),
                            "donor_side": str(donor_name),
                            "recv_side": str(recv_name),
                            "donor_expected_label": str(donor.expected_label),
                            "recv_expected_label": str(recv.expected_label),
                            "donor_ratio": donor_ratio,
                            "recv_ratio": recv_ratio,
                            "predictor_delta": predictor_delta,
                            "expected_sign": expected_sign,
                            "donor_ordered_score": donor_ordered,
                            "base_ordered_score": base_ordered,
                            "patched_ordered_score": patched_ordered,
                            "sham_ordered_score": sham_ordered,
                            "raw_gap": raw_gap,
                            "aligned_raw_gap": aligned_raw_gap,
                            "effect": effect,
                            "aligned_effect": aligned_effect,
                            "sham_effect": sham_effect,
                            "sham_aligned_effect": sham_aligned_effect,
                            "patch_fraction": patch_fraction,
                            "kl_patched_to_donor": kl_patched_to_donor,
                            "kl_patched_to_receiver": _kl_div(patched_probs, base_probs),
                            "kl_base_to_donor": kl_base_to_donor,
                            "kl_sham_to_donor": kl_sham_to_donor,
                            "kl_patched_to_donor_delta_vs_base": float(kl_patched_to_donor - kl_base_to_donor),
                            "kl_patched_to_donor_delta_vs_sham": float(kl_patched_to_donor - kl_sham_to_donor),
                            "direction_match": int(aligned_effect > 0.0),
                            **_prob_columns("donor", donor_probs),
                            **_prob_columns("base", base_probs),
                            **_prob_columns("patched", patched_probs),
                            **_prob_columns("sham", sham_probs),
                        }
                    )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, bootstrap_b: int, seed: int) -> Dict[str, Any]:
    patched_rows = [r for r in rows if int(r.get("skipped", 0)) == 0]
    skipped_rows = [r for r in rows if int(r.get("skipped", 0)) != 0]
    by_layer: Dict[str, Dict[str, Any]] = {}
    for layer in sorted({int(r["layer"]) for r in patched_rows}):
        layer_rows = [r for r in patched_rows if int(r["layer"]) == int(layer)]
        aligned = [float(r["aligned_effect"]) for r in layer_rows]
        sham_aligned = [float(r["sham_aligned_effect"]) for r in layer_rows]
        patch_fractions = [float(r["patch_fraction"]) for r in layer_rows]
        kl_base_to_donor = [float(r["kl_base_to_donor"]) for r in layer_rows]
        kl_patched_to_donor = [float(r["kl_patched_to_donor"]) for r in layer_rows]
        kl_patched_to_receiver = [float(r["kl_patched_to_receiver"]) for r in layer_rows]
        kl_patched_to_donor_delta_vs_base = [
            float(r["kl_patched_to_donor_delta_vs_base"]) for r in layer_rows
        ]
        matches = [float(r["direction_match"]) for r in layer_rows]
        by_layer[str(layer)] = {
            "aligned_effect": _bootstrap_mean_ci(aligned, b=bootstrap_b, seed=seed + int(layer) * 13),
            "sham_aligned_effect": _bootstrap_mean_ci(sham_aligned, b=bootstrap_b, seed=seed + int(layer) * 17),
            "patch_fraction": _bootstrap_mean_ci(patch_fractions, b=bootstrap_b, seed=seed + int(layer) * 19),
            "patch_fraction_median": _quantile(patch_fractions, 0.5),
            "kl_base_to_donor": _bootstrap_mean_ci(kl_base_to_donor, b=bootstrap_b, seed=seed + int(layer) * 23),
            "kl_patched_to_donor": _bootstrap_mean_ci(
                kl_patched_to_donor, b=bootstrap_b, seed=seed + int(layer) * 29
            ),
            "kl_patched_to_receiver": _bootstrap_mean_ci(
                kl_patched_to_receiver, b=bootstrap_b, seed=seed + int(layer) * 31
            ),
            "kl_patched_to_donor_delta_vs_base": _bootstrap_mean_ci(
                kl_patched_to_donor_delta_vs_base,
                b=bootstrap_b,
                seed=seed + int(layer) * 37,
            ),
            "direction_match_rate": _mean(matches),
            "n": int(len(layer_rows)),
        }
    return {
        "schema_version": "gradable_size_raw_patch_v2",
        "n_rows": int(len(rows)),
        "n_patched_rows": int(len(patched_rows)),
        "n_skipped_rows": int(len(skipped_rows)),
        "layers": by_layer,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw activation patching gate for size gradable predicates.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--layers", type=str, default="4,8")
    p.add_argument("--patch_site", type=str, default="target", choices=["target", "final_prompt_token"])
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap_B", type=int, default=1000)
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--patch_allow_token_id_mismatch", action="store_true")
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument("--strict_finite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_summary", type=str, required=True)
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
    if (out_csv.exists() or out_summary.exists()) and not bool(args.overwrite):
        raise FileExistsError("Output exists. Use --overwrite to replace raw patching outputs.")

    items = load_disamb_pairs(str(args.data_path))
    if int(args.max_items) > 0:
        items = items[: int(args.max_items)]
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
    rows = run_raw_size_patching(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        device=device,
        layers=_parse_int_list(str(args.layers)),
        patch_site_name=str(args.patch_site),
        normalize_by_length=not bool(args.no_length_norm),
        require_token_id_match=not bool(args.patch_allow_token_id_mismatch),
    )
    summary = summarize_rows(rows, bootstrap_b=int(args.bootstrap_B), seed=int(args.seed))
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "data_path": str(args.data_path),
        "layers": str(args.layers),
        "patch_site": str(args.patch_site),
        "device": str(device),
        "torch_dtype": str(model_torch_dtype or ""),
        "normalize_by_length": not bool(args.no_length_norm),
        "require_token_id_match": not bool(args.patch_allow_token_id_mismatch),
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
    print(f"Wrote raw patch CSV: {out_csv}")
    print(f"Wrote raw patch summary: {out_summary}")


if __name__ == "__main__":
    main()
