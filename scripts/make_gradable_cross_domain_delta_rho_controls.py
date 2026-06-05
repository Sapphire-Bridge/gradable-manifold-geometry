from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.data.schemas import DisambPair, PromptSide
from aom.repro import get_git_commit_hash


DEFAULT_ROOT = ROOT / "results" / "manifold_groups_poc"


def _parse_source_specs(raw: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for part in re.split(r"[, \n\t]+", str(raw)):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Source specs must look like domain=path, got {part!r}")
        domain, path = part.split("=", 1)
        domain = domain.strip()
        path = path.strip()
        if not domain or not path:
            raise ValueError(f"Invalid source spec {part!r}")
        out.append((domain, Path(path)))
    if not out:
        raise ValueError("Expected at least one source spec")
    return out


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


def _side(item: DisambPair, side_name: str) -> PromptSide:
    if str(side_name) == "a":
        return item.a
    if str(side_name) == "b":
        return item.b
    raise ValueError(f"Unknown side {side_name!r}")


def _metadata_coordinate(item: DisambPair, side_name: str) -> Tuple[float, str]:
    md = dict(item.metadata or {})
    suffix = str(side_name)
    log_ratio_key = f"log_ratio_{suffix}"
    ratio_key = f"ratio_{suffix}"
    delta_key = f"delta_{suffix}"
    standard_key = f"standard_{suffix}"
    if log_ratio_key in md:
        return float(md[log_ratio_key]), "log_ratio"
    if ratio_key in md:
        ratio = float(md[ratio_key])
        if ratio <= 0.0:
            raise ValueError(f"{item.pair_id}: nonpositive {ratio_key}={ratio!r}")
        return float(math.log(ratio)), "log_ratio"
    if delta_key in md:
        return float(md[delta_key]), "delta_from_standard"
    if "value" in md and standard_key in md:
        value = float(md["value"])
        standard = float(md[standard_key])
        if value > 0.0 and standard > 0.0:
            return float(math.log(value / standard)), "log_ratio"
        return float(value - standard), "delta_from_standard"
    raise ValueError(f"{item.pair_id}: cannot infer scalar coordinate for side {side_name!r}")


def _side_records(items: Sequence[DisambPair], *, domain: str, data_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in items:
        for side_name in ("a", "b"):
            coord_raw, coord_kind = _metadata_coordinate(item, side_name)
            side = _side(item, side_name)
            md = dict(item.metadata or {})
            records.append(
                {
                    "domain": str(domain),
                    "data_path": str(data_path),
                    "pair_id": str(item.pair_id),
                    "side": str(side_name),
                    "prompt": str(side.prompt),
                    "expected_label": str(side.expected_label),
                    "coordinate_raw": float(coord_raw),
                    "coordinate_kind": str(coord_kind),
                    "dimension": str(md.get("dimension", domain)),
                    "predicate_family": str(md.get("predicate_family", "")),
                    "value": md.get("value", ""),
                    "target": str(item.target),
                }
            )
    coords = [float(r["coordinate_raw"]) for r in records]
    mu = _mean(coords)
    var = _mean((float(x) - float(mu)) ** 2 for x in coords)
    sd = math.sqrt(var) if _is_finite(var) and var > 1e-12 else 1.0
    for row in records:
        row["coordinate_z"] = float((float(row["coordinate_raw"]) - float(mu)) / float(sd))
        row["domain_coordinate_mean"] = float(mu)
        row["domain_coordinate_sd"] = float(sd)
    return records


def _records_by_pair(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    out: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        out[str(row["pair_id"])][str(row["side"])] = row
    return out


def _directed_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    match_mode: str,
    role: str,
) -> List[Dict[str, Any]]:
    if str(match_mode) not in {"domain_z", "raw"}:
        raise ValueError(f"Unknown match_mode {match_mode!r}")
    coord_key = "coordinate_z" if str(match_mode) == "domain_z" else "coordinate_raw"
    rows: List[Dict[str, Any]] = []
    by_pair = _records_by_pair(records)
    for pair_id, sides in sorted(by_pair.items()):
        if set(sides.keys()) != {"a", "b"}:
            continue
        for donor_side, recv_side in (("a", "b"), ("b", "a")):
            donor = sides[donor_side]
            recv = sides[recv_side]
            delta_raw = float(donor["coordinate_raw"]) - float(recv["coordinate_raw"])
            delta_z = float(donor["coordinate_z"]) - float(recv["coordinate_z"])
            match_delta = float(donor[coord_key]) - float(recv[coord_key])
            if abs(match_delta) <= 1e-12:
                continue
            rows.append(
                {
                    f"{role}_domain": str(domain),
                    f"{role}_pair_id": str(pair_id),
                    f"{role}_direction": f"{donor_side}_to_{recv_side}",
                    f"{role}_donor_side": str(donor_side),
                    f"{role}_recv_side": str(recv_side),
                    f"{role}_donor_prompt": str(donor["prompt"]),
                    f"{role}_recv_prompt": str(recv["prompt"]),
                    f"{role}_donor_expected_label": str(donor["expected_label"]),
                    f"{role}_recv_expected_label": str(recv["expected_label"]),
                    f"{role}_donor_coordinate_raw": float(donor["coordinate_raw"]),
                    f"{role}_recv_coordinate_raw": float(recv["coordinate_raw"]),
                    f"{role}_donor_coordinate_z": float(donor["coordinate_z"]),
                    f"{role}_recv_coordinate_z": float(recv["coordinate_z"]),
                    f"{role}_delta_raw": float(delta_raw),
                    f"{role}_delta_z": float(delta_z),
                    f"{role}_match_delta": float(match_delta),
                    f"{role}_coordinate_kind": str(donor["coordinate_kind"]),
                    f"{role}_data_path": str(donor["data_path"]),
                    f"{role}_dimension": str(donor["dimension"]),
                    f"{role}_predicate_family": str(donor["predicate_family"]),
                }
            )
    return rows


def _greedy_unique_matches(
    *,
    size_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    max_abs_match_error: float | None,
    allow_source_reuse: bool,
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any], float]]:
    candidates: List[Tuple[float, str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for size_row in size_rows:
        size_delta = float(size_row["size_match_delta"])
        for source_row in source_rows:
            source_delta = float(source_row["source_match_delta"])
            err = abs(float(size_delta) - float(source_delta))
            if max_abs_match_error is not None and err > float(max_abs_match_error):
                continue
            candidates.append(
                (
                    err,
                    str(size_row["size_pair_id"]) + "::" + str(size_row["size_direction"]),
                    str(source_row["source_pair_id"]) + "::" + str(source_row["source_direction"]),
                    size_row,
                    source_row,
                )
            )
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    matched_size: set[str] = set()
    matched_source: set[str] = set()
    out: List[Tuple[Mapping[str, Any], Mapping[str, Any], float]] = []
    for err, size_key, source_key, size_row, source_row in candidates:
        if size_key in matched_size:
            continue
        if not allow_source_reuse and source_key in matched_source:
            continue
        matched_size.add(size_key)
        matched_source.add(source_key)
        out.append((size_row, source_row, float(err)))
    out.sort(key=lambda row: (str(row[0]["size_pair_id"]), str(row[0]["size_direction"]), str(row[1]["source_domain"])))
    return out


def make_controls(
    *,
    size_data_path: Path,
    source_specs: Sequence[Tuple[str, Path]],
    match_mode: str,
    max_abs_match_error: float | None,
    min_unique_source_pairs: int,
    allow_source_reuse: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    size_items = load_disamb_pairs(str(size_data_path))
    size_records = _side_records(size_items, domain="size", data_path=size_data_path)
    size_directed = _directed_rows(size_records, domain="size", match_mode=match_mode, role="size")
    all_rows: List[Dict[str, Any]] = []
    source_summaries: Dict[str, Any] = {}
    for source_domain, source_path in source_specs:
        source_items = load_disamb_pairs(str(source_path))
        source_records = _side_records(source_items, domain=source_domain, data_path=source_path)
        source_directed = _directed_rows(source_records, domain=source_domain, match_mode=match_mode, role="source")
        matches = _greedy_unique_matches(
            size_rows=size_directed,
            source_rows=source_directed,
            max_abs_match_error=max_abs_match_error,
            allow_source_reuse=bool(allow_source_reuse),
        )
        unique_source_pairs = sorted({str(source["source_pair_id"]) for _, source, _ in matches})
        unique_size_pairs = sorted({str(size["size_pair_id"]) for size, _, _ in matches})
        errors = [err for _, _, err in matches]
        source_summaries[str(source_domain)] = {
            "source_data_path": str(source_path),
            "n_source_pairs": int(len(source_items)),
            "n_source_directed": int(len(source_directed)),
            "n_matches": int(len(matches)),
            "n_unique_source_pairs": int(len(unique_source_pairs)),
            "n_unique_size_pairs": int(len(unique_size_pairs)),
            "mean_abs_match_error": _mean(errors),
            "max_abs_match_error_observed": max(errors) if errors else float("nan"),
            "passes_min_unique_source_pairs": bool(len(unique_source_pairs) >= int(min_unique_source_pairs)),
        }
        if len(unique_source_pairs) < int(min_unique_source_pairs):
            raise ValueError(
                f"{source_domain}: only {len(unique_source_pairs)} unique source pairs matched; "
                f"minimum is {int(min_unique_source_pairs)}. Increase max_abs_match_error or inspect source coverage."
            )
        for match_idx, (size_row, source_row, err) in enumerate(matches):
            row: Dict[str, Any] = {
                "control_id": f"{source_domain}-{match_idx:04d}",
                "match_mode": str(match_mode),
                "match_abs_error": float(err),
                "match_rank": int(match_idx),
                "expected_sign": 1 if float(size_row["size_match_delta"]) > 0 else -1,
                "max_abs_match_error": "" if max_abs_match_error is None else float(max_abs_match_error),
                "allow_source_reuse": bool(allow_source_reuse),
                **dict(size_row),
                **dict(source_row),
            }
            all_rows.append(row)
    summary = {
        "schema_version": "gradable_cross_domain_delta_rho_controls_v1",
        "size_data_path": str(size_data_path),
        "source_specs": {str(domain): str(path) for domain, path in source_specs},
        "match_mode": str(match_mode),
        "max_abs_match_error": None if max_abs_match_error is None else float(max_abs_match_error),
        "min_unique_source_pairs": int(min_unique_source_pairs),
        "allow_source_reuse": bool(allow_source_reuse),
        "n_size_pairs": int(len(size_items)),
        "n_size_directed": int(len(size_directed)),
        "n_rows": int(len(all_rows)),
        "source_summaries": source_summaries,
    }
    return all_rows, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build matched signed-delta controls for cross-domain gradable low-rank patching.")
    p.add_argument(
        "--size_data_path",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl"),
    )
    p.add_argument(
        "--source_specs",
        type=str,
        default="temperature=results/manifold_groups_poc/gradable_temperature_disamb_pairs_v1_2.jsonl,age=results/manifold_groups_poc/gradable_age_disamb_pairs_v1_2.jsonl",
        help="Comma/space separated specs like temperature=path,age=path.",
    )
    p.add_argument("--match_mode", type=str, default="domain_z", choices=["domain_z", "raw"])
    p.add_argument("--max_abs_match_error", type=float, default=None)
    p.add_argument("--min_unique_source_pairs", type=int, default=20)
    p.add_argument("--allow_source_reuse", action="store_true")
    p.add_argument(
        "--out_csv",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_matched_delta_rho_controls.csv"),
    )
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(DEFAULT_ROOT / "gradable_cross_domain_matched_delta_rho_controls.summary.json"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_csv = Path(str(args.out_csv))
    out_summary = Path(str(args.out_summary))
    if (out_csv.exists() or out_summary.exists()) and not bool(args.overwrite):
        raise FileExistsError("Control outputs exist. Use --overwrite to replace them.")
    rows, summary = make_controls(
        size_data_path=Path(str(args.size_data_path)),
        source_specs=_parse_source_specs(str(args.source_specs)),
        match_mode=str(args.match_mode),
        max_abs_match_error=args.max_abs_match_error,
        min_unique_source_pairs=int(args.min_unique_source_pairs),
        allow_source_reuse=bool(args.allow_source_reuse),
    )
    summary["run"] = {
        "git_commit": str(get_git_commit_hash(repo_root=ROOT, required=False)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    _write_csv(rows, out_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote controls CSV: {out_csv}")
    print(f"Wrote controls summary: {out_summary}")
    for domain, dom_summary in summary["source_summaries"].items():
        print(
            f"{domain}: n_matches={dom_summary['n_matches']} "
            f"unique_source_pairs={dom_summary['n_unique_source_pairs']} "
            f"mean_abs_match_error={dom_summary['mean_abs_match_error']:.4f}"
        )


if __name__ == "__main__":
    main()
