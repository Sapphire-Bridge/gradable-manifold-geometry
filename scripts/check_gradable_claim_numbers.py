#!/usr/bin/env python3
"""Gate the gradable claim numbers against their source artifacts and docs.

For each claim in ``tables/gradable_release/claim_numbers.json``:

1. (artifact binding) re-derive the value(s) from the listed source artifact at
   the given JSON path, round to 3 decimal places, and assert they equal the
   canonical numbers. This fails if a result file is regenerated and drifts.
2. (doc consistency) assert the canonical display string appears verbatim in
   every listed doc. This fails if README / one-pager text drifts from the
   canonical numbers.

Exit non-zero on any drift so this can run as a release gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "tables" / "gradable_release" / "claim_numbers.json"


def get_path(obj, path):
    for part in path.strip("/").split("/"):
        obj = obj[part]
    return obj


def main() -> int:
    spec = json.loads(SPEC.read_text())
    docs = {}
    errors = []
    n_bound = 0

    for claim in spec["claims"]:
        cid = claim["id"]

        binding = claim.get("binding")
        if binding:
            src = ROOT / binding["source_file"]
            try:
                data = json.loads(src.read_text())
                node = get_path(data, binding["json_path"])
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{cid}: cannot read {binding['source_file']}{binding['json_path']} ({exc})")
                continue
            for key, expected in binding["fields"].items():
                # key "value" means the node itself is the scalar; otherwise index into it
                raw = node if key == "value" else node[key]
                actual = round(float(raw), 3)
                if actual != expected:
                    errors.append(
                        f"{cid}: artifact {binding['json_path']}/{key} = {actual} "
                        f"!= canonical {expected}"
                    )
                else:
                    n_bound += 1

        doc_string = claim.get("doc_string")
        if doc_string:
            for d in claim.get("docs", []):
                if d not in docs:
                    docs[d] = (ROOT / d).read_text()
                if doc_string not in docs[d]:
                    errors.append(f"{cid}: '{doc_string}' not found in {d}")

    if errors:
        print("CLAIM NUMBER DRIFT DETECTED:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    print(
        f"OK: {len(spec['claims'])} claims consistent "
        f"({n_bound} artifact-bound fields verified, docs in sync)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
