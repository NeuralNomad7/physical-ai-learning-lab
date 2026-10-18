"""Review check (not a project feature) for the Phase 2 review: does a rerun of the gate reproduce the original?

usage: python review2_rerun_compare.py ORIGINAL_RUN_DIR RERUN_DIR
Compares gate_result.json record by record (every field) and the trajectory CSVs byte for byte.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


def diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(f"{path}/{k} (only on one side)")
            else:
                out += diff(x[k], y[k], f"{path}/{k}")
        return out
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [f"{path} (length {len(x)} vs {len(y)})"]
        out = []
        for i, (u, v) in enumerate(zip(x, y)):
            out += diff(u, v, f"{path}[{i}]")
        return out
    return [] if x == y else [path]


def main():
    original, rerun = Path(sys.argv[1]), Path(sys.argv[2])
    a = json.loads((original / "gate_result.json").read_text(encoding="utf-8"))
    b = json.loads((rerun / "gate_result.json").read_text(encoding="utf-8"))
    print(f"original: {original} (started {a['started_utc']}), rerun: {rerun} (started {b['started_utc']})")
    print(f"successes {a['successes']}/{a['episodes']} vs {b['successes']}/{b['episodes']}")
    top = [k for k in sorted(set(a) | set(b)) if k != "records" and a.get(k) != b.get(k)]
    print("top-level fields that differ:", top)
    kinds = Counter()
    for ra, rb in zip(a["records"], b["records"]):
        for p in diff(ra, rb):
            kinds[re.sub(r"\[\d+\]", "[]", p)] += 1
    print("record fields that differ (count over the 20 records):")
    for k, n in kinds.most_common():
        print(f"  {n:4d}  {k}")
    same = total = 0
    for f in sorted((original / "trajectories").glob("seed_*.csv")):
        total += 1
        same += hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256((rerun / "trajectories" / f.name).read_bytes()).digest()
    print(f"trajectory CSVs byte-identical: {same} of {total}")


if __name__ == "__main__":
    main()
