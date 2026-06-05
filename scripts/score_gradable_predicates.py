from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

# Ensure repo root importability when invoked as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.data.schemas import DisambPair, PromptSide
from aom.metrics.disamb import _margin, score_labels_next_continuations
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed


DEFAULT_LABEL_ORDER = ("cold", "cool", "warm", "hot")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _parse_label_order(raw: str) -> List[str]:
    labels = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not labels:
        raise ValueError("--label_order must contain at least one label")
    return labels


def _finite_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _softmax_from_log_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    vals = {str(k): float(v) for k, v in scores.items()}
    if not vals:
        return {}
    m = max(vals.values())
    exps = {k: math.exp(v - m) for k, v in vals.items()}
    denom = sum(exps.values())
    if denom <= 0.0:
        return {k: float("nan") for k in vals}
    return {k: float(v / denom) for k, v in exps.items()}


def _entropy(probs: Mapping[str, float]) -> float:
    h = 0.0
    for p in probs.values():
        p_f = float(p)
        if p_f > 0.0 and math.isfinite(p_f):
            h -= p_f * math.log(p_f)
    return float(h)


def _ordered_score(probs: Mapping[str, float], label_order: Sequence[str]) -> float:
    score = 0.0
    mass = 0.0
    for i, label in enumerate(label_order):
        p = float(probs.get(str(label), 0.0))
        if math.isfinite(p):
            score += float(i) * p
            mass += p
    return float(score / mass) if mass > 0.0 else float("nan")


def _label_index(label: str, label_order: Sequence[str]) -> float:
    try:
        return float([str(x) for x in label_order].index(str(label)))
    except ValueError:
        return float("nan")


def _side_row(
    *,
    item: DisambPair,
    side_name: str,
    side: PromptSide,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    label_order: Sequence[str],
    normalize_by_length: bool,
) -> Dict[str, Any]:
    scores = score_labels_next_continuations(
        model,
        tokenizer,
        side.prompt,
        item.choices,
        device,
        normalize_by_length=bool(normalize_by_length),
    )
    pred = scores.argmax_label()
    probs = _softmax_from_log_scores(scores.by_label)
    row: Dict[str, Any] = {
        "pair_id": str(item.pair_id),
        "side": str(side_name),
        "target": str(item.target),
        "prompt": str(side.prompt),
        "expected_label": str(side.expected_label),
        "pred_label": str(pred),
        "correct": int(str(pred) == str(side.expected_label)),
        "margin": float(_margin(scores, str(side.expected_label))),
        "entropy": _entropy(probs),
        "ordered_label_score": _ordered_score(probs, label_order),
        "expected_label_index": _label_index(str(side.expected_label), label_order),
        "pred_label_index": _label_index(str(pred), label_order),
    }
    md = dict(item.metadata or {})
    for key in (
        "type",
        "dimension",
        "unit",
        "value",
        "predicate_family",
        "comparison_a",
        "comparison_b",
        "design",
        "source",
        "standard_type",
        "scale_type",
        "control_type",
        "is_borderline",
        "is_artificial_norm",
    ):
        if key in md:
            row[str(key)] = md[key]
    for label in sorted(scores.by_label.keys()):
        row[f"logscore_{label}"] = float(scores.by_label[str(label)])
        row[f"prob_{label}"] = float(probs.get(str(label), float("nan")))
    return row


def score_items(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: Sequence[DisambPair],
    device: torch.device,
    label_order: Sequence[str],
    normalize_by_length: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    side_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for item in items:
        row_a = _side_row(
            item=item,
            side_name="a",
            side=item.a,
            model=model,
            tokenizer=tokenizer,
            device=device,
            label_order=label_order,
            normalize_by_length=normalize_by_length,
        )
        row_b = _side_row(
            item=item,
            side_name="b",
            side=item.b,
            model=model,
            tokenizer=tokenizer,
            device=device,
            label_order=label_order,
            normalize_by_length=normalize_by_length,
        )
        side_rows.extend([row_a, row_b])

        expected_shift = float(row_b["expected_label_index"]) - float(row_a["expected_label_index"])
        observed_shift = float(row_b["ordered_label_score"]) - float(row_a["ordered_label_score"])
        pred_shift = float(row_b["pred_label_index"]) - float(row_a["pred_label_index"])
        pair_rows.append(
            {
                "pair_id": str(item.pair_id),
                "target": str(item.target),
                "expected_a": str(item.a.expected_label),
                "expected_b": str(item.b.expected_label),
                "pred_a": str(row_a["pred_label"]),
                "pred_b": str(row_b["pred_label"]),
                "correct_a": int(row_a["correct"]),
                "correct_b": int(row_b["correct"]),
                "both_correct": int(int(row_a["correct"]) == 1 and int(row_b["correct"]) == 1),
                "margin_a": float(row_a["margin"]),
                "margin_b": float(row_b["margin"]),
                "expected_shift": expected_shift,
                "observed_ordered_shift": observed_shift,
                "pred_shift": pred_shift,
                "shift_direction_match": int(
                    math.isfinite(expected_shift)
                    and math.isfinite(observed_shift)
                    and expected_shift != 0.0
                    and observed_shift * expected_shift > 0.0
                ),
                "abs_observed_ordered_shift": abs(observed_shift) if math.isfinite(observed_shift) else float("nan"),
                "entropy_a": float(row_a["entropy"]),
                "entropy_b": float(row_b["entropy"]),
                "comparison_a": (item.metadata or {}).get("comparison_a", ""),
                "comparison_b": (item.metadata or {}).get("comparison_b", ""),
                "value": (item.metadata or {}).get("value", ""),
            }
        )

    summary = {
        "n_pairs": int(len(pair_rows)),
        "n_sides": int(len(side_rows)),
        "side_accuracy": _finite_mean([float(r["correct"]) for r in side_rows]),
        "pair_both_correct_rate": _finite_mean([float(r["both_correct"]) for r in pair_rows]),
        "mean_expected_margin": _finite_mean([float(r["margin"]) for r in side_rows]),
        "mean_entropy": _finite_mean([float(r["entropy"]) for r in side_rows]),
        "mean_abs_observed_ordered_shift": _finite_mean([float(r["abs_observed_ordered_shift"]) for r in pair_rows]),
        "shift_direction_match_rate": _finite_mean([float(r["shift_direction_match"]) for r in pair_rows]),
        "label_order": [str(x) for x in label_order],
    }
    return side_rows, pair_rows, summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Score gradable-predicate DISAMB-style items.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument(
        "--data_path",
        type=str,
        default=str(root / "results" / "manifold_groups_poc" / "gradable_temperature_disamb_pairs_v0.jsonl"),
    )
    p.add_argument("--label_order", type=str, default=",".join(DEFAULT_LABEL_ORDER))
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument("--strict_finite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--out_csv",
        type=str,
        default=str(root / "results" / "manifold_groups_poc" / "gradable_v0_behavior.sides.csv"),
    )
    p.add_argument(
        "--out_pairs_csv",
        type=str,
        default=str(root / "results" / "manifold_groups_poc" / "gradable_v0_behavior.pairs.csv"),
    )
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(root / "results" / "manifold_groups_poc" / "gradable_v0_behavior.summary.json"),
    )
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

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])
    model_torch_dtype = str(args.torch_dtype) if args.torch_dtype else None
    if model_torch_dtype is None and str(getattr(device, "type", "")).lower() == "mps":
        model_torch_dtype = "float32"

    out_csv = Path(str(args.out_csv))
    out_pairs_csv = Path(str(args.out_pairs_csv))
    out_summary = Path(str(args.out_summary))
    if any(p.exists() for p in (out_csv, out_pairs_csv, out_summary)) and not bool(args.overwrite):
        raise FileExistsError("Output exists. Use --overwrite to replace existing gradable behavior outputs.")

    items = load_disamb_pairs(str(args.data_path))
    if int(args.max_items) > 0:
        items = items[: int(args.max_items)]
    if not items:
        raise ValueError("No items loaded from --data_path")

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

    label_order = _parse_label_order(str(args.label_order))
    side_rows, pair_rows, summary = score_items(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        device=device,
        label_order=label_order,
        normalize_by_length=not bool(args.no_length_norm),
    )

    _write_csv(side_rows, out_csv)
    _write_csv(pair_rows, out_pairs_csv)
    summary = {
        "schema_version": "gradable_behavior_v1",
        **summary,
        "run": {
            "model_name_or_path": str(args.model_name_or_path),
            "model_revision": str(args.revision or ""),
            "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
            "data_path": str(args.data_path),
            "data_sha256": _sha256_file(Path(str(args.data_path))),
            "device": str(device),
            "torch_dtype": str(model_torch_dtype or ""),
            "logprobs_dtype": str(args.logprobs_dtype),
            "strict_finite": bool(args.strict_finite),
            "normalize_by_length": not bool(args.no_length_norm),
            "seed": int(args.seed),
            "git_commit": str(get_git_commit_hash(repo_root=Path(__file__).resolve().parents[1], required=False)),
            "started_at_utc": str(started_at_utc),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_sec": float(time.perf_counter() - t0),
            "out_csv": str(out_csv),
            "out_pairs_csv": str(out_pairs_csv),
            "out_summary": str(out_summary),
            "versions": collect_versions(),
        },
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote side CSV: {out_csv}")
    print(f"Wrote pair CSV: {out_pairs_csv}")
    print(f"Wrote summary JSON: {out_summary}")


if __name__ == "__main__":
    main()
