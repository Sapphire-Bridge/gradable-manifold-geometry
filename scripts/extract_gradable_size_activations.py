from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.interventions.activation_patching import get_block_outputs
from aom.metrics.disamb import _encode_prompt
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions, get_git_commit_hash
from aom.utils import configure_logprob_computation, get_best_device, set_seed


DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"
DEFAULT_LABEL_ORDER = ("tiny", "small", "large", "huge")


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


def _to_float(raw: Any, default: float = float("nan")) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _side_rows_for_variant(variant: str, *, results_root: Path) -> List[Dict[str, Any]]:
    path = results_root / f"gradable_size_v2_{variant}_behavior_recompute.sides.csv"
    rows = _read_csv(path)
    by_prompt: Dict[str, Dict[str, str]] = {}
    for row in rows:
        by_prompt.setdefault(str(row.get("prompt_hash", "")), row)
    out: List[Dict[str, Any]] = []
    for prompt_hash, row in sorted(by_prompt.items()):
        meta: Dict[str, Any] = {
            "variant": str(variant),
            "prompt_hash": prompt_hash,
            "pair_id": str(row.get("pair_id", "")),
            "side": str(row.get("side", "")),
            "prompt": str(row.get("prompt", "")),
            "comparison": str(row.get("comparison", "")),
            "value": _to_float(row.get("value")),
            "standard": _to_float(row.get("standard")),
            "rho": _to_float(row.get("predictor")),
            "predictor": _to_float(row.get("predictor")),
            "target": str(row.get("target", "")),
            "expected_label": str(row.get("expected_label", "")),
            "expected_label_index": _to_float(row.get("expected_label_index")),
            "pred_label": str(row.get("pred_label", "")),
            "pred_label_index": _to_float(row.get("pred_label_index")),
            "ordered_score": _to_float(row.get("ordered_score")),
            "signed_score": _to_float(row.get("signed_score")),
            "entropy": _to_float(row.get("entropy")),
        }
        for label in DEFAULT_LABEL_ORDER:
            if f"prob_{label}" in row:
                meta[f"prob_{label}"] = _to_float(row.get(f"prob_{label}"))
        out.append(meta)
    return out


@torch.no_grad()
def extract_activations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    side_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    device: torch.device,
    site: str,
) -> tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    if str(site) != "final_prompt_token":
        raise ValueError("Only final_prompt_token extraction is currently supported.")
    x_by_layer: Dict[int, List[np.ndarray]] = {int(layer): [] for layer in layers}
    metadata_rows: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(side_rows):
        prompt = str(row["prompt"])
        ids = _encode_prompt(tokenizer, prompt, device=device)
        token_idx = int(ids.shape[-1]) - 1
        blocks = get_block_outputs(model, ids, layers=layers)
        for layer in layers:
            vec = blocks[int(layer)][0, token_idx, :].detach().to("cpu", dtype=torch.float32).numpy()
            x_by_layer[int(layer)].append(vec)
        metadata_rows.append(
            {
                "row_idx": int(row_idx),
                "site": str(site),
                "token_idx": int(token_idx),
                "prompt_length": int(ids.shape[-1]),
                **dict(row),
            }
        )
    arrays = {int(layer): np.stack(vals, axis=0).astype(np.float32) for layer, vals in x_by_layer.items()}
    return arrays, metadata_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract validated final-token activations for gradable-size geometry.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--variants", type=str, default="fictional_semantic_adjective_counts,iso_ratio_adjective_counts")
    p.add_argument("--layers", type=str, default="20,24,28,32,33")
    p.add_argument("--site", type=str, default="final_prompt_token", choices=["final_prompt_token"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument("--strict_finite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--results_root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--out_prefix", type=str, default=str(DEFAULT_ROOT / "gradable_size_geometry_late_final_token_gemma3"))
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

    out_prefix = Path(str(args.out_prefix))
    out_npz = Path(str(out_prefix) + ".npz")
    out_metadata = Path(str(out_prefix) + ".metadata.csv")
    out_summary = Path(str(out_prefix) + ".summary.json")
    if (out_npz.exists() or out_metadata.exists() or out_summary.exists()) and not bool(args.overwrite):
        raise FileExistsError("Geometry extraction outputs exist. Use --overwrite to replace them.")

    variants = _parse_csv_or_space(str(args.variants))
    layers = _parse_int_list(str(args.layers))
    side_rows: List[Dict[str, Any]] = []
    for variant in variants:
        side_rows.extend(_side_rows_for_variant(variant, results_root=Path(str(args.results_root))))

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
    arrays, metadata_rows = extract_activations(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        side_rows=side_rows,
        layers=layers,
        device=device,
        site=str(args.site),
    )

    npz_payload = {f"X_layer_{layer}": arr for layer, arr in arrays.items()}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **npz_payload)
    _write_csv(metadata_rows, out_metadata)
    summary = {
        "schema_version": "gradable_size_activation_extraction_v1",
        "variants": variants,
        "layers": [int(x) for x in layers],
        "site": str(args.site),
        "n_rows": int(len(metadata_rows)),
        "array_shapes": {str(layer): list(arr.shape) for layer, arr in arrays.items()},
        "run": {
            "model_name_or_path": str(args.model_name_or_path),
            "model_revision": str(args.revision or ""),
            "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
            "device": str(device),
            "torch_dtype": str(model_torch_dtype or ""),
            "replacement_source": "decoder_block_output_hook",
            "git_commit": str(get_git_commit_hash(repo_root=ROOT, required=False)),
            "started_at_utc": str(started_at_utc),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_sec": float(time.perf_counter() - t0),
            "versions": collect_versions(),
        },
        "outputs": {
            "npz": str(out_npz),
            "metadata_csv": str(out_metadata),
            "summary_json": str(out_summary),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote activation NPZ: {out_npz}")
    print(f"Wrote activation metadata: {out_metadata}")
    print(f"Wrote activation summary: {out_summary}")


if __name__ == "__main__":
    main()
