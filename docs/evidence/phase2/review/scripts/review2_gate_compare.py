"""Review check (not a project feature) for the Phase 2 review (D25): original gate vs gate after the fix.

usage: python review2_gate_compare.py BEFORE_gate_result.json AFTER_gate_result.json
Reads only the two result files written by `physical_ai_lab gate`.
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summary(name, path, d):
    recs = d["records"]
    st = [r["stats"] for r in recs]
    succ = [r for r in recs if r["success"]]
    finals = [r["final"]["block_target_distance"] * 1e3 for r in succ]
    events = [c for r in recs for c in r["contact_events"]]
    print(f"== {name}: {path}")
    print(f"   started {d['started_utc']}  successes {d['successes']}/{d['episodes']}  gate_passed {d['gate_passed']}"
          f"  failures {d['failures_by_reason']}")
    print(f"   successful episodes: pushes mean {sum(r['pushes_used'] for r in succ) / len(succ):.2f}"
          f" max {max(r['pushes_used'] for r in succ)} | sim time mean {sum(r['end_time'] for r in succ) / len(succ):.2f} s"
          f" max {max(r['end_time'] for r in succ):.2f} s")
    print(f"   final distance mm {min(finals):.1f} .. {max(finals):.1f}")
    print(f"   lowest measured TCP mm {min(s['min_tcp_z'] for s in st) * 1e3:.2f}"
          f" | lowest commanded TCP mm {min(s['min_commanded_tcp_z'] for s in st) * 1e3:.2f}")
    print(f"   max joint speed ratio {max(s['max_joint_speed_ratio'] for s in st):.4f}"
          f" | max joint range excess {max(s['max_joint_range_excess'] for s in st)}")
    print(f"   rate-limited updates {sum(s['rate_limited_updates'] for s in st)}"
          f" of {sum(s['control_updates'] for s in st)} control updates")
    print(f"   max IK position error {max(s['max_ik_position_error'] for s in st) * 1e3:.3f} mm"
          f" | max IK orientation error {max(s['max_ik_orientation_error'] for s in st):.6f}")
    print(f"   contact events {len(events)}, forbidden {sum(1 for c in events if c['forbidden'])},"
          f" unplanned robot-block contacts {sum(r['unplanned_robot_block_contacts'] for r in recs)}")


def main():
    before_path, after_path = sys.argv[1], sys.argv[2]
    before, after = load(before_path), load(after_path)
    summary("original gate (before the fix)", before_path, before)
    summary("gate after the fix", after_path, after)
    print()
    print("seed | pushes | final mm | lowest TCP mm | rate-limited updates | first push plan identical")
    same_first = changed = 0
    for ra, rb in zip(before["records"], after["records"]):
        assert ra["seed"] == rb["seed"]
        first_same = json.dumps(ra["pushes"][0]["plan"], sort_keys=True) == json.dumps(rb["pushes"][0]["plan"], sort_keys=True)
        same_first += first_same
        changed += ra["pushes_used"] != rb["pushes_used"]
        print(f"{ra['seed']} | {ra['pushes_used']} -> {rb['pushes_used']}"
              f" | {ra['final']['block_target_distance'] * 1e3:5.1f} -> {rb['final']['block_target_distance'] * 1e3:5.1f}"
              f" | {ra['stats']['min_tcp_z'] * 1e3:5.2f} -> {rb['stats']['min_tcp_z'] * 1e3:5.2f}"
              f" | {ra['stats']['rate_limited_updates']:3d} -> {rb['stats']['rate_limited_updates']:3d}"
              f" | {first_same}")
    print(f"seeds with a different number of pushes: {changed}")
    print(f"first push plan identical (all fields) on {same_first} of {len(before['records'])} seeds")


if __name__ == "__main__":
    main()
