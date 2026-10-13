"""Large-scale numerical verification of Theorems 1-3 and Proposition 4 on random instances.

Writes results/verification.json (results/verification_quick.json with --quick).  Every reported
number is a worst case over all instances.  Errors should be at most ~1e-9 (the LP check: the solver
tolerance); for the sampling checks, the lower-bound ratio should be >= 1, the upper-bound ratio <= 1,
and the failure excess <= 0.

Usage:  python scripts/verify_theory.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tailreduce import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()
    s = 0.1 if args.quick else 1.0
    rng = np.random.default_rng(args.seed)
    res = {"seed": args.seed}
    t0 = time.time()
    n1 = int(3000 * s)
    viol, gap = verify.check_theorem1(rng, n_inst=n1)
    res["theorem1"] = {"instances": n1, "max_violation_of_upper_bounds": viol,
                       "max_shortfall_of_tight_instance": gap}
    print("Theorem 1", res["theorem1"], f"{time.time() - t0:.1f}s", flush=True)
    n2 = int(400 * s)
    res["theorem2"] = verify.check_theorem2(rng, n_inst=n2, n_random=2000)
    res["theorem2"]["reference_models"] = n2
    print("Theorem 2", res["theorem2"], f"{time.time() - t0:.1f}s", flush=True)
    ng = int(5000 * s)
    res["grid_lemma"] = {"instances": ng, "max_increase": verify.check_grid_lemma(rng, n_inst=ng)}
    print("Grid lemma", res["grid_lemma"], flush=True)
    nt = int(3000 * s)
    we, wm = verify.check_threshold(rng, n_inst=nt)
    res["threshold"] = {"instances": nt, "max_discrepancy_of_exact_model": we,
                        "max_shortfall_when_tail_scenario_missing": wm}
    print("Threshold", res["threshold"], flush=True)
    n3 = int(1500 * s)
    res["theorem3"] = verify.check_theorem3(rng, n_inst=n3)
    res["theorem3"]["instances_per_part"] = n3
    print("Theorem 3", res["theorem3"], flush=True)
    n4 = int(1000 * s)
    res["proposition4"] = verify.check_proposition4(rng, n_inst=n4, n_mc=int(20000 * s))
    res["proposition4"]["instances"] = n4
    print("Proposition 4", res["proposition4"], flush=True)
    res["seconds"] = time.time() - t0
    out = ROOT / "results" / ("verification_quick.json" if args.quick else "verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
