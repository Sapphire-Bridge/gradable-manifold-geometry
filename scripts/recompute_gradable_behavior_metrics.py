from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"
DEFAULT_LABEL_ORDERS = {
    "temperature": ("cold", "cool", "warm", "hot"),
    "size": ("tiny", "small", "large", "huge"),
    "age": ("young", "youthful", "mature", "old"),
}


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _to_float(raw: Any, default: float = float("nan")) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if _is_finite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _sign_label(x: float) -> str:
    if not _is_finite(float(x)):
        return "nan"
    if float(x) > 0.0:
        return "positive"
    if float(x) < 0.0:
        return "negative"
    return "zero"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite(float(x)) and _is_finite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float(num / (den_x * den_y))


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _is_finite(float(x)) and _is_finite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    if den == 0.0:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in pairs) / den)


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


def _ci(xs: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    return {
        "lo": _quantile(xs, alpha / 2.0),
        "hi": _quantile(xs, 1.0 - alpha / 2.0),
    }


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if n <= 0:
        return {"lo": float("nan"), "hi": float("nan")}
    phat = float(k) / float(n)
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return {"lo": float(max(0.0, center - half)), "hi": float(min(1.0, center + half))}


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    probs = [math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(prob for prob in probs if prob <= obs + 1e-18)))


def _bootstrap_resample(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    b: int,
    seed: int,
) -> Dict[str, float]:
    valid = [r for r in rows if _is_finite(_to_float(r.get("predictor"))) and _is_finite(_to_float(r.get("ordered_score")))]
    if len(valid) < 2 or b <= 0:
        return {"lo": float("nan"), "hi": float("nan")}
    rng = random.Random(int(seed))
    vals: List[float] = []
    n = len(valid)
    for _ in range(int(b)):
        sample = [valid[rng.randrange(n)] for _ in range(n)]
        if metric == "corr_predictor":
            vals.append(_pearson([float(r["predictor"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        elif metric == "corr_expected_label":
            vals.append(_pearson([float(r["expected_label_index"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        elif metric == "slope_predictor":
            vals.append(_linear_slope([float(r["predictor"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        else:
            raise ValueError(f"Unknown bootstrap metric: {metric}")
    return _ci(vals)


def _bootstrap_by_pair(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    b: int,
    seed: int,
) -> Dict[str, float]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    keys = sorted(grouped)
    if not keys or b <= 0:
        return {"lo": float("nan"), "hi": float("nan")}
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(int(b)):
        sample: List[Mapping[str, Any]] = []
        for _ in keys:
            sample.extend(grouped[rng.choice(keys)])
        if metric == "corr_predictor":
            vals.append(_pearson([float(r["predictor"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        elif metric == "corr_expected_label":
            vals.append(_pearson([float(r["expected_label_index"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        elif metric == "slope_predictor":
            vals.append(_linear_slope([float(r["predictor"]) for r in sample], [float(r["ordered_score"]) for r in sample]))
        else:
            raise ValueError(f"Unknown bootstrap metric: {metric}")
    return _ci(vals)


def _permutation_p_corr(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_x: str,
    key_y: str,
    observed: float,
    b: int,
    seed: int,
) -> float:
    pairs = [
        (_to_float(r.get(key_x)), _to_float(r.get(key_y)))
        for r in rows
        if _is_finite(_to_float(r.get(key_x))) and _is_finite(_to_float(r.get(key_y)))
    ]
    if len(pairs) < 3 or not _is_finite(observed) or b <= 0:
        return float("nan")
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    rng = random.Random(int(seed))
    ge = 0
    obs_abs = abs(float(observed))
    ys_perm = list(ys)
    for _ in range(int(b)):
        rng.shuffle(ys_perm)
        val = _pearson(xs, ys_perm)
        if _is_finite(val) and abs(val) >= obs_abs:
            ge += 1
    return float((ge + 1) / (int(b) + 1))


def _within_target_permutation_p_corr(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_x: str,
    key_y: str,
    target_key: str,
    observed: float,
    b: int,
    seed: int,
) -> float:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _to_float(row.get(key_x))
        y = _to_float(row.get(key_y))
        if _is_finite(x) and _is_finite(y):
            grouped[str(row.get(target_key, ""))].append((x, y))
    groups = {k: v for k, v in grouped.items() if len(v) >= 2}
    if not groups or not _is_finite(observed) or b <= 0:
        return float("nan")
    if sum(len(v) for v in groups.values()) < 3:
        return float("nan")
    rng = random.Random(int(seed))
    ge = 0
    obs_abs = abs(float(observed))
    for _ in range(int(b)):
        xs_perm: List[float] = []
        ys_perm: List[float] = []
        for vals in groups.values():
            xs = [x for x, _ in vals]
            ys = [y for _, y in vals]
            rng.shuffle(ys)
            xs_perm.extend(xs)
            ys_perm.extend(ys)
        val = _pearson(xs_perm, ys_perm)
        if _is_finite(val) and abs(val) >= obs_abs:
            ge += 1
    return float((ge + 1) / (int(b) + 1))


def _side_key(pair_id: str, side: str) -> Tuple[str, str]:
    return str(pair_id), str(side)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]


def _ordered_score_from_probs(row: Mapping[str, str], label_order: Sequence[str]) -> float:
    total = 0.0
    weighted = 0.0
    for i, label in enumerate(label_order):
        p = _to_float(row.get(f"prob_{label}"))
        if _is_finite(p):
            total += p
            weighted += float(i) * p
    return float(weighted / total) if total > 0.0 else float("nan")


def _signed_score_from_probs(row: Mapping[str, str], label_order: Sequence[str]) -> float:
    labels = [str(x) for x in label_order]
    if len(labels) < 2:
        return float("nan")
    mid = len(labels) // 2
    low_labels = labels[:mid]
    high_labels = labels[mid:] if len(labels) % 2 == 0 else labels[mid + 1 :]
    low = sum(_to_float(row.get(f"prob_{label}"), 0.0) for label in low_labels)
    high = sum(_to_float(row.get(f"prob_{label}"), 0.0) for label in high_labels)
    return float(high - low)


def _prob_mass(row: Mapping[str, str], label_order: Sequence[str]) -> float:
    vals = [_to_float(row.get(f"prob_{label}")) for label in label_order]
    return float(sum(v for v in vals if _is_finite(v)))


def _argmax_from_probs(row: Mapping[str, str], label_order: Sequence[str]) -> str:
    best_label = ""
    best_val = -float("inf")
    for label in label_order:
        val = _to_float(row.get(f"prob_{label}"), -float("inf"))
        if val > best_val:
            best_label = str(label)
            best_val = val
    return best_label


def _label_index(label: str, label_order: Sequence[str]) -> float:
    try:
        return float([str(x) for x in label_order].index(str(label)))
    except ValueError:
        return float("nan")


def _domain_paths(version: str, domain: str) -> Tuple[Path, Path, Path]:
    data = DEFAULT_ROOT / f"gradable_{domain}_disamb_pairs_{version}.jsonl"
    if version == "v1_1" and domain == "temperature":
        prefix = DEFAULT_ROOT / "gradable_v1_1_behavior_gemma3"
    else:
        prefix = DEFAULT_ROOT / f"gradable_{domain}_{version}_behavior_gemma3"
    return data, Path(str(prefix) + ".sides.csv"), Path(str(prefix) + ".pairs.csv")


def _predictor_for_side(domain: str, md: Mapping[str, Any], side: str) -> float:
    suffix = "a" if side == "a" else "b"
    if domain == "temperature":
        return _to_float(md.get(f"delta_{suffix}"))
    ratio = _to_float(md.get(f"ratio_{suffix}"))
    if ratio > 0.0 and _is_finite(ratio):
        return float(math.log(ratio))
    return float("nan")


def _aligned_shift(expected_shift: float, observed_shift: float) -> float:
    if not _is_finite(expected_shift) or not _is_finite(observed_shift) or expected_shift == 0.0:
        return float("nan")
    return float(observed_shift * (1.0 if expected_shift > 0.0 else -1.0))


def _side_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    bootstrap_b: int,
    permutation_b: int,
    seed: int,
    bootstrap_label: str,
) -> Dict[str, Any]:
    pred_corr = _pearson([float(r["predictor"]) for r in rows], [float(r["ordered_score"]) for r in rows])
    signed_corr = _pearson([float(r["predictor"]) for r in rows], [float(r["signed_score"]) for r in rows])
    label_corr = _pearson([float(r["expected_label_index"]) for r in rows], [float(r["ordered_score"]) for r in rows])
    pred_slope = _linear_slope([float(r["predictor"]) for r in rows], [float(r["ordered_score"]) for r in rows])
    pred_counts = Counter(str(r["pred_label"]) for r in rows)
    expected_counts = Counter(str(r["expected_label"]) for r in rows)
    boot_fn = _bootstrap_resample if bootstrap_label == "prompt" else _bootstrap_by_pair
    corr_ci_key = f"bootstrap_{bootstrap_label}95"
    return {
        "side_accuracy": _mean([float(r["correct"]) for r in rows]),
        "mean_entropy": _mean([float(r["entropy"]) for r in rows]),
        "corr_ordered_score_vs_predictor": {
            "predictor": "delta_value_minus_standard" if domain == "temperature" else "log_value_over_standard",
            "r": pred_corr,
            corr_ci_key: boot_fn(rows, metric="corr_predictor", b=bootstrap_b, seed=seed + 11),
            "permutation_p_two_sided": _permutation_p_corr(
                rows,
                key_x="predictor",
                key_y="ordered_score",
                observed=pred_corr,
                b=permutation_b,
                seed=seed + 17,
            ),
            "within_target_permutation_p_two_sided": _within_target_permutation_p_corr(
                rows,
                key_x="predictor",
                key_y="ordered_score",
                target_key="target",
                observed=pred_corr,
                b=permutation_b,
                seed=seed + 19,
            ),
        },
        "slope_ordered_score_vs_predictor": {
            "slope": pred_slope,
            corr_ci_key: boot_fn(rows, metric="slope_predictor", b=bootstrap_b, seed=seed + 23),
        },
        "corr_signed_score_vs_predictor": {
            "predictor": "delta_value_minus_standard" if domain == "temperature" else "log_value_over_standard",
            "r": signed_corr,
            "permutation_p_two_sided": _permutation_p_corr(
                rows,
                key_x="predictor",
                key_y="signed_score",
                observed=signed_corr,
                b=permutation_b,
                seed=seed + 37,
            ),
            "within_target_permutation_p_two_sided": _within_target_permutation_p_corr(
                rows,
                key_x="predictor",
                key_y="signed_score",
                target_key="target",
                observed=signed_corr,
                b=permutation_b,
                seed=seed + 41,
            ),
        },
        "corr_ordered_score_vs_expected_label_index": {
            "r": label_corr,
            corr_ci_key: boot_fn(rows, metric="corr_expected_label", b=bootstrap_b, seed=seed + 29),
            "permutation_p_two_sided": _permutation_p_corr(
                rows,
                key_x="expected_label_index",
                key_y="ordered_score",
                observed=label_corr,
                b=permutation_b,
                seed=seed + 31,
            ),
        },
        "argmax_label_counts": dict(pred_counts),
        "expected_label_counts": dict(expected_counts),
    }


def recompute_domain(
    *,
    domain: str,
    data_path: Path,
    sides_path: Path,
    pairs_path: Path,
    label_order: Sequence[str],
    bootstrap_b: int,
    permutation_b: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = _read_jsonl(data_path)
    sides = _read_csv(sides_path)
    _ = _read_csv(pairs_path)

    metadata_by_side: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    item_by_pair: Dict[str, Mapping[str, Any]] = {}
    for item in data:
        md = dict(item.get("metadata") or {})
        pair_id = str(item["pair_id"])
        item_by_pair[pair_id] = item
        metadata_by_side[_side_key(pair_id, "a")] = md
        metadata_by_side[_side_key(pair_id, "b")] = md

    side_rows: List[Dict[str, Any]] = []
    missing_prob_columns: List[str] = []
    expected_labels_not_in_label_order = 0
    for row in sides:
        side = str(row.get("side", ""))
        pair_id = str(row.get("pair_id", ""))
        md = metadata_by_side.get(_side_key(pair_id, side), {})
        expected_label = str(row.get("expected_label", ""))
        prompt = str(row.get("prompt", ""))
        for label in label_order:
            if f"prob_{label}" not in row:
                missing_prob_columns.append(f"prob_{label}")
        pred_label = _argmax_from_probs(row, label_order)
        ordered = _ordered_score_from_probs(row, label_order)
        signed_score = _signed_score_from_probs(row, label_order)
        expected_idx = _label_index(expected_label, label_order)
        if not _is_finite(expected_idx):
            expected_labels_not_in_label_order += 1
        pred_idx = _label_index(pred_label, label_order)
        out_row: Dict[str, Any] = {
            "domain": str(domain),
            "pair_id": pair_id,
            "side": side,
            "prompt": prompt,
            "prompt_hash": _prompt_hash(prompt),
            "target": str(row.get("target", "")),
            "value": _to_float(md.get("value", row.get("value"))),
            "comparison": str(md.get(f"comparison_{side}", "")),
            "standard": _to_float(md.get(f"standard_{side}")),
            "predictor": _predictor_for_side(domain, md, side),
            "expected_label": expected_label,
            "expected_label_index": expected_idx,
            "pred_label": pred_label,
            "pred_label_index": pred_idx,
            "ordered_score": ordered,
            "signed_score": signed_score,
            "entropy": _to_float(row.get("entropy")),
            "prob_mass_in_label_order": _prob_mass(row, label_order),
            "correct": int(pred_label == expected_label),
        }
        for label in label_order:
            out_row[f"prob_{label}"] = _to_float(row.get(f"prob_{label}"))
        side_rows.append(out_row)

    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in side_rows:
        by_pair[str(row["pair_id"])][str(row["side"])] = row

    pair_rows: List[Dict[str, Any]] = []
    for pair_id, pair in sorted(by_pair.items()):
        if "a" not in pair or "b" not in pair:
            continue
        a = pair["a"]
        b = pair["b"]
        item = item_by_pair.get(pair_id, {})
        md = dict(item.get("metadata") or {}) if item else {}
        expected_shift = float(b["expected_label_index"]) - float(a["expected_label_index"])
        observed_shift = float(b["ordered_score"]) - float(a["ordered_score"])
        predictor_shift = float(b["predictor"]) - float(a["predictor"])
        match = (
            _is_finite(expected_shift)
            and _is_finite(observed_shift)
            and expected_shift != 0.0
            and observed_shift * expected_shift > 0.0
        )
        pair_rows.append(
            {
                "domain": str(domain),
                "pair_id": pair_id,
                "target": str(a.get("target", "")),
                "value": _to_float(md.get("value")),
                "comparison_a": str(a.get("comparison", "")),
                "comparison_b": str(b.get("comparison", "")),
                "standard_a": _to_float(a.get("standard")),
                "standard_b": _to_float(b.get("standard")),
                "predictor_a": float(a["predictor"]),
                "predictor_b": float(b["predictor"]),
                "predictor_shift": predictor_shift,
                "expected_a": str(a["expected_label"]),
                "expected_b": str(b["expected_label"]),
                "expected_shift": expected_shift,
                "ordered_score_a": float(a["ordered_score"]),
                "ordered_score_b": float(b["ordered_score"]),
                "observed_ordered_shift": observed_shift,
                "aligned_observed_shift": _aligned_shift(expected_shift, observed_shift),
                "shift_direction_match": int(match),
                "abs_observed_ordered_shift": abs(observed_shift) if _is_finite(observed_shift) else float("nan"),
                "expected_shift_sign": _sign_label(expected_shift),
                "predictor_shift_sign": _sign_label(predictor_shift),
                "observed_ordered_shift_sign": _sign_label(observed_shift),
                "prompt_hash_a": str(a.get("prompt_hash", "")),
                "prompt_hash_b": str(b.get("prompt_hash", "")),
            }
        )

    valid_pair_rows = [
        row
        for row in pair_rows
        if _is_finite(float(row["expected_shift"]))
        and float(row["expected_shift"]) != 0.0
        and _is_finite(float(row["observed_ordered_shift"]))
    ]
    n_match = sum(int(row["shift_direction_match"]) for row in valid_pair_rows)
    n_valid_pairs = len(valid_pair_rows)
    expected_sign_counts = Counter(str(row["expected_shift_sign"]) for row in valid_pair_rows)
    predictor_sign_counts = Counter(str(row["predictor_shift_sign"]) for row in valid_pair_rows)
    observed_sign_counts = Counter(str(row["observed_ordered_shift_sign"]) for row in valid_pair_rows)

    unique_by_prompt: Dict[str, Dict[str, Any]] = {}
    for row in side_rows:
        unique_by_prompt.setdefault(str(row["prompt_hash"]), row)
    unique_side_rows = list(unique_by_prompt.values())

    prob_masses = [float(r["prob_mass_in_label_order"]) for r in side_rows if _is_finite(float(r["prob_mass_in_label_order"]))]
    warnings: List[str] = []
    if len(expected_sign_counts) < 2:
        warnings.append("Pair orientation is one-sided; sign-flip null is not valid for this dataset.")
    if expected_labels_not_in_label_order:
        warnings.append(f"{expected_labels_not_in_label_order} expected labels were not in label_order.")
    if missing_prob_columns:
        warnings.append(f"Missing probability columns: {sorted(set(missing_prob_columns))}")

    result = {
        "domain": str(domain),
        "inputs": {
            "data_jsonl": str(data_path),
            "sides_csv": str(sides_path),
            "pairs_csv": str(pairs_path),
            "label_order": [str(x) for x in label_order],
            "predictor": "delta" if domain == "temperature" else "log_ratio",
            "data_sha256": _sha256_file(data_path),
            "sides_sha256": _sha256_file(sides_path),
            "pairs_sha256": _sha256_file(pairs_path),
        },
        "counts": {
            "n_pairs_jsonl": int(len(data)),
            "n_pair_rows_recomputed": int(len(pair_rows)),
            "n_valid_direction_pairs": int(n_valid_pairs),
            "n_side_rows": int(len(side_rows)),
            "n_unique_sides": int(len(unique_side_rows)),
        },
        "orientation": {
            "expected_shift_sign_counts": dict(expected_sign_counts),
            "predictor_shift_sign_counts": dict(predictor_sign_counts),
            "observed_ordered_shift_sign_counts": dict(observed_sign_counts),
            "is_orientation_balanced": bool(len(expected_sign_counts) >= 2),
        },
        "side_metrics_unique": _side_metrics(
            unique_side_rows,
            domain=domain,
            bootstrap_b=bootstrap_b,
            permutation_b=permutation_b,
            seed=seed + 100,
            bootstrap_label="prompt",
        ),
        "side_metrics_expanded": _side_metrics(
            side_rows,
            domain=domain,
            bootstrap_b=bootstrap_b,
            permutation_b=permutation_b,
            seed=seed + 200,
            bootstrap_label="pair",
        ),
        "pair_metrics": {
            "mean_abs_observed_ordered_shift": _mean([float(r["abs_observed_ordered_shift"]) for r in valid_pair_rows]),
            "mean_aligned_observed_shift": _mean([float(r["aligned_observed_shift"]) for r in valid_pair_rows]),
        },
        "shift_direction_match": {
            "count": int(n_match),
            "n": int(n_valid_pairs),
            "rate": float(n_match / n_valid_pairs) if n_valid_pairs else float("nan"),
            "wilson95_naive": _wilson(n_match, n_valid_pairs),
            "binomial_p_two_sided_vs_0_5_naive": _binom_two_sided_p(n_match, n_valid_pairs),
        },
        "consistency_checks": {
            "missing_prob_columns": sorted(set(missing_prob_columns)),
            "expected_labels_not_in_label_order": int(expected_labels_not_in_label_order),
            "prob_mass_min": min(prob_masses) if prob_masses else float("nan"),
            "prob_mass_max": max(prob_masses) if prob_masses else float("nan"),
        },
        "warnings": warnings,
    }
    return result, side_rows, pair_rows


def _format_float(x: float, digits: int = 3) -> str:
    if not _is_finite(float(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# Gradable Behavior Recompute",
        "",
        f"- Version: `{summary['version']}`",
        f"- Bootstrap resamples: `{summary['config']['bootstrap_B']}`",
        f"- Permutations: `{summary['config']['permutation_B']}`",
        "- Primary side metrics deduplicate repeated prompts by `prompt_hash`.",
        "",
        "| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(ordered,predictor) | r(signed,predictor) | r(ordered,label idx) | argmax counts |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for domain, row in summary["domains"].items():
        dm = row["shift_direction_match"]
        signs = row["orientation"]["expected_shift_sign_counts"]
        unique = row["side_metrics_unique"]
        pred_counts = unique["argmax_label_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(domain),
                    str(row["counts"]["n_pair_rows_recomputed"]),
                    str(row["counts"]["n_unique_sides"]),
                    _format_float(float(unique["side_accuracy"])),
                    f"{dm['count']}/{dm['n']}={_format_float(float(dm['rate']))}",
                    ", ".join(f"{k}:{v}" for k, v in sorted(signs.items())),
                    _format_float(float(unique["corr_ordered_score_vs_predictor"]["r"])),
                    _format_float(float(unique["corr_signed_score_vs_predictor"]["r"])),
                    _format_float(float(unique["corr_ordered_score_vs_expected_label_index"]["r"])),
                    ", ".join(f"{k}:{v}" for k, v in sorted(pred_counts.items())),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `temperature` uses `value - standard` as predictor.",
            "- `size` and `age` use `log(value / standard)` as predictor.",
            "- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.",
            "- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute gradable-predicate behavior metrics from existing CSV/JSONL artifacts.")
    p.add_argument("--version", type=str, default="v1_1")
    p.add_argument("--domains", type=str, default="temperature,size,age")
    p.add_argument("--bootstrap_B", type=int, default=2000)
    p.add_argument("--permutation_B", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature_data", type=str, default="")
    p.add_argument("--temperature_sides", type=str, default="")
    p.add_argument("--temperature_pairs", type=str, default="")
    p.add_argument("--size_data", type=str, default="")
    p.add_argument("--size_sides", type=str, default="")
    p.add_argument("--size_pairs", type=str, default="")
    p.add_argument("--age_data", type=str, default="")
    p.add_argument("--age_sides", type=str, default="")
    p.add_argument("--age_pairs", type=str, default="")
    p.add_argument("--temperature_label_order", type=str, default="")
    p.add_argument("--size_label_order", type=str, default="")
    p.add_argument("--age_label_order", type=str, default="")
    p.add_argument("--out_json", type=str, default=str(DEFAULT_ROOT / "gradable_behavior_recompute.json"))
    p.add_argument("--out_md", type=str, default=str(DEFAULT_ROOT / "gradable_behavior_recompute.md"))
    p.add_argument("--out_sides_csv", type=str, default="")
    p.add_argument("--out_pairs_csv", type=str, default="")
    return p.parse_args()


def _override_path(args: argparse.Namespace, domain: str, kind: str, default: Path) -> Path:
    raw = getattr(args, f"{domain}_{kind}")
    return Path(str(raw)) if str(raw).strip() else default


def _label_order_for(args: argparse.Namespace, domain: str) -> Tuple[str, ...]:
    raw = str(getattr(args, f"{domain}_label_order", "")).strip()
    if raw:
        vals = tuple(part.strip() for part in raw.split(",") if part.strip())
        if not vals:
            raise ValueError(f"{domain}_label_order was provided but empty")
        return vals
    return tuple(DEFAULT_LABEL_ORDERS[domain])


def main() -> None:
    args = parse_args()
    domains = [part.strip() for part in str(args.domains).split(",") if part.strip()]
    if not domains:
        raise ValueError("--domains must contain at least one domain")
    summary: Dict[str, Any] = {
        "schema_version": "gradable_behavior_recompute_v1",
        "version": str(args.version),
        "config": {
            "ci": 0.95,
            "bootstrap_B": int(args.bootstrap_B),
            "permutation_B": int(args.permutation_B),
            "seed": int(args.seed),
            "side_dedup_key": "prompt_hash",
            "strict": True,
        },
        "domains": {},
    }
    all_side_rows: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    for domain in domains:
        if domain not in DEFAULT_LABEL_ORDERS:
            raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(DEFAULT_LABEL_ORDERS)}")
        data_default, sides_default, pairs_default = _domain_paths(str(args.version), domain)
        data_path = _override_path(args, domain, "data", data_default)
        sides_path = _override_path(args, domain, "sides", sides_default)
        pairs_path = _override_path(args, domain, "pairs", pairs_default)
        for label, path in (("data", data_path), ("sides", sides_path), ("pairs", pairs_path)):
            if not path.exists():
                raise FileNotFoundError(f"Missing {domain} {label} file: {path}")
        domain_summary, side_rows, pair_rows = recompute_domain(
            domain=domain,
            data_path=data_path,
            sides_path=sides_path,
            pairs_path=pairs_path,
            label_order=_label_order_for(args, domain),
            bootstrap_b=int(args.bootstrap_B),
            permutation_b=int(args.permutation_B),
            seed=int(args.seed),
        )
        summary["domains"][domain] = domain_summary
        all_side_rows.extend(side_rows)
        all_pair_rows.extend(pair_rows)

    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    if str(args.out_sides_csv).strip():
        _write_csv(all_side_rows, Path(str(args.out_sides_csv)))
    if str(args.out_pairs_csv).strip():
        _write_csv(all_pair_rows, Path(str(args.out_pairs_csv)))
    print(f"Wrote recompute JSON: {out_json}")
    print(f"Wrote recompute Markdown: {out_md}")
    if str(args.out_sides_csv).strip():
        print(f"Wrote recomputed side CSV: {args.out_sides_csv}")
    if str(args.out_pairs_csv).strip():
        print(f"Wrote recomputed pair CSV: {args.out_pairs_csv}")


if __name__ == "__main__":
    main()
