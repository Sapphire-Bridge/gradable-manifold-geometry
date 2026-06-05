from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.metrics.disamb import score_labels_next_continuations
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed
from scripts.generate_gradable_size_v2 import READOUTS, cm_phrase, label_for_ratio


DEFAULT_VALUES = (1, 3, 8, 15, 25, 40, 60, 90, 130, 200, 320, 500)
DEFAULT_STANDARDS = (1, 9, 46, 60, 170, 350)


def _softmax(scores: Mapping[str, float]) -> Dict[str, float]:
    vals = {str(k): float(v) for k, v in scores.items()}
    m = max(vals.values())
    exps = {k: math.exp(v - m) for k, v in vals.items()}
    z = sum(exps.values())
    return {k: float(v / z) for k, v in exps.items()}


def _ordered_score(probs: Mapping[str, float], label_order: Sequence[str]) -> float:
    mass = 0.0
    score = 0.0
    for i, label in enumerate(label_order):
        p = float(probs.get(str(label), 0.0))
        mass += p
        score += float(i) * p
    return float(score / mass) if mass > 0.0 else float("nan")


def _signed_score(probs: Mapping[str, float], label_order: Sequence[str]) -> float:
    labels = [str(x) for x in label_order]
    mid = len(labels) // 2
    low = sum(float(probs.get(label, 0.0)) for label in labels[:mid])
    high_labels = labels[mid:] if len(labels) % 2 == 0 else labels[mid + 1 :]
    high = sum(float(probs.get(label, 0.0)) for label in high_labels)
    return float(high - low)


def _entropy(probs: Mapping[str, float]) -> float:
    total = 0.0
    for p in probs.values():
        p = float(p)
        if p > 0.0 and math.isfinite(p):
            total -= p * math.log(p)
    return float(total)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return float(num / (den_x * den_y)) if den_x > 0.0 and den_y > 0.0 else float("nan")


def _parse_ints(raw: str, default: Sequence[int]) -> List[int]:
    if not str(raw).strip():
        return [int(x) for x in default]
    vals = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def _prompt_rows(values: Sequence[int], standards: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in values:
        value_text = cm_phrase(int(value))
        rows.extend(
            [
                {
                    "template": "value_body_feels",
                    "value": int(value),
                    "standard": "",
                    "predictor": float("nan"),
                    "prompt": f"A body length of {value_text} feels",
                },
                {
                    "template": "value_body_is",
                    "value": int(value),
                    "standard": "",
                    "predictor": float("nan"),
                    "prompt": f"A body length of {value_text} is",
                },
                {
                    "template": "value_body_considered",
                    "value": int(value),
                    "standard": "",
                    "predictor": float("nan"),
                    "prompt": f"A body length of {value_text} would be considered",
                },
                {
                    "template": "value_overall_counts",
                    "value": int(value),
                    "standard": "",
                    "predictor": float("nan"),
                    "prompt": f"An overall length of {value_text} counts as",
                },
            ]
        )
    for value in values:
        for standard in standards:
            ratio = float(value) / float(standard)
            if 0.85 <= ratio <= 1.15:
                continue
            rows.append(
                {
                    "template": "standard_overall_counts",
                    "value": int(value),
                    "standard": int(standard),
                    "predictor": float(math.log(ratio)),
                    "prompt": (
                        f"The normal overall length is {cm_phrase(int(standard))}. "
                        f"An overall length of {cm_phrase(int(value))} counts as"
                    ),
                }
            )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_readout_template: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_readout_template[f"{row['readout_family']}::{row['template']}"].append(row)
    groups: Dict[str, Any] = {}
    for key, group in sorted(by_readout_template.items()):
        pred = [float(r["predictor"]) for r in group]
        ordered = [float(r["ordered_score"]) for r in group]
        signed = [float(r["signed_score"]) for r in group]
        groups[key] = {
            "n": int(len(group)),
            "argmax_counts": dict(Counter(str(r["pred_label"]) for r in group)),
            "mean_entropy": _mean(float(r["entropy"]) for r in group),
            "mean_ordered_score": _mean(ordered),
            "mean_signed_score": _mean(signed),
            "corr_ordered_score_vs_predictor": _pearson(pred, ordered),
            "corr_signed_score_vs_predictor": _pearson(pred, signed),
        }
    return {
        "schema_version": "gradable_size_prior_diagnostics_v1",
        "n_rows": int(len(rows)),
        "groups": groups,
    }


@torch.no_grad()
def score_prior_prompts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    readout_families: Sequence[str],
    values: Sequence[int],
    standards: Sequence[int],
    normalize_by_length: bool,
) -> List[Dict[str, Any]]:
    prompt_rows = _prompt_rows(values, standards)
    rows: List[Dict[str, Any]] = []
    for readout_family in readout_families:
        cfg = READOUTS[str(readout_family)]
        choices = cfg["choices"]
        label_order = tuple(str(x) for x in cfg["label_order"])
        for base in prompt_rows:
            scores = score_labels_next_continuations(
                model,
                tokenizer,
                str(base["prompt"]),
                choices,
                device,
                normalize_by_length=normalize_by_length,
            )
            probs = _softmax(scores.by_label)
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            ratio = (
                float(base["value"]) / float(base["standard"])
                if str(base["standard"]).strip()
                else float("nan")
            )
            expected = label_for_ratio(ratio, readout_family=str(readout_family)) if math.isfinite(ratio) else ""
            row: Dict[str, Any] = {
                **dict(base),
                "readout_family": str(readout_family),
                "label_order": ",".join(label_order),
                "pred_label": str(pred),
                "expected_label": str(expected or ""),
                "entropy": _entropy(probs),
                "ordered_score": _ordered_score(probs, label_order),
                "signed_score": _signed_score(probs, label_order),
            }
            for label in label_order:
                row[f"logscore_{label}"] = float(scores.by_label[label])
                row[f"prob_{label}"] = float(probs[label])
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    root = ROOT
    p = argparse.ArgumentParser(description="Measure base/readout priors for gradable size labels.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--readout_families", type=str, default="adjective,normality4,binary")
    p.add_argument("--values", type=str, default="")
    p.add_argument("--standards", type=str, default="")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument("--strict_finite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_csv", type=str, default=str(root / "results" / "manifold_groups_poc" / "gradable_size_prior_diagnostics.csv"))
    p.add_argument("--out_summary", type=str, default=str(root / "results" / "manifold_groups_poc" / "gradable_size_prior_diagnostics.summary.json"))
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
        raise FileExistsError("Output exists. Use --overwrite to replace prior diagnostics outputs.")

    readout_families = [x.strip() for x in str(args.readout_families).split(",") if x.strip()]
    for family in readout_families:
        if family not in READOUTS:
            raise ValueError(f"Unknown readout family {family!r}; expected one of {sorted(READOUTS)}")
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
    rows = score_prior_prompts(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        device=device,
        readout_families=readout_families,
        values=_parse_ints(str(args.values), DEFAULT_VALUES),
        standards=_parse_ints(str(args.standards), DEFAULT_STANDARDS),
        normalize_by_length=not bool(args.no_length_norm),
    )
    summary = summarize(rows)
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "readout_families": readout_families,
        "device": str(device),
        "torch_dtype": str(model_torch_dtype or ""),
        "normalize_by_length": not bool(args.no_length_norm),
        "git_commit": str(get_git_commit_hash(repo_root=ROOT, required=False)),
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": float(time.perf_counter() - t0),
        "versions": collect_versions(),
    }
    _write_csv(rows, out_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote prior CSV: {out_csv}")
    print(f"Wrote prior summary: {out_summary}")


if __name__ == "__main__":
    main()
