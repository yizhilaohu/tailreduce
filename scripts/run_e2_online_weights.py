"""Experiment E2: data-driven (time-decayed) weights, as produced by an online weighting step.

Weights w_t proportional to lambda^(age_t) with half-life h (lambda = 2^(-1/h)).  For rolling windows,
records (i) the minimum exact size K_min = n_a and its ratio to N, (ii) the largest profile jump J,
(iii) D+ + D- of the minimax-optimal model (OPT), the best conservative model (OPT-C) and the
tail-block rule (TB), together with the bounds 1/K - J <= beta_K and TB < 1/K of Theorem 3.
A second loop records K_min for N in {500, 1000, 2000, 5000} (table of K_min by index, N and a).

Usage:  python scripts/run_e2_online_weights.py [--quick]
Outputs: results/e2_online.csv, results/e2_kmin_curve.csv, results/e2_kmin_table.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tailreduce import data, discrepancies, normalize_model, tail_profile  # noqa: E402
from tailreduce import reduction as R  # noqa: E402

HALF_LIVES = (None, 63, 250, 1000)  # None = uniform weights
A_LIST = (0.01, 0.05, 0.10)
K_LIST = (5, 10, 20, 50)
A_CURVE = np.round(np.r_[0.005, np.arange(0.01, 0.2001, 0.01)], 4)


def ewma_weights(N, half_life):
    if half_life is None:
        return np.full(N, 1.0 / N)
    lam = 2.0 ** (-1.0 / half_life)
    w = lam ** np.arange(N)[::-1]  # most recent observation (last) has the largest weight
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--step", type=int, default=63)
    args = ap.parse_args()
    N = args.N
    step = 504 if args.quick else args.step
    rows, curve = [], []
    t_start = time.time()
    for ds in data.MAIN_DATASETS:
        L = data.load_losses(ROOT / "data" / "processed", ds).to_numpy()
        T = len(L)
        for end in range(N, T + 1, step):
            vals = L[end - N:end]
            for hl in HALF_LIVES:
                xi, w = normalize_model(vals, ewma_weights(N, hl))
                hl_lab = "uniform" if hl is None else int(hl)
                for a in A_CURVE:
                    x, n_a = tail_profile(w, a)
                    curve.append({"dataset": ds, "end": end, "half_life": hl_lab, "a": a, "n_a": n_a,
                                  "kmin_ratio": n_a / N})
                for a in A_LIST:
                    x, n_a = tail_profile(w, a)
                    J = float(np.diff(x[: n_a + 1]).max())
                    for K in K_LIST:
                        rec = {"dataset": ds, "end": end, "half_life": hl_lab, "a": a, "K": K, "n_a": n_a,
                               "J": J, "lb": 1.0 / K - J, "ub": 1.0 / K, "w_max_over_a": w.max() / a}
                        for name, fn in (("OPT", lambda: R.optimal_reduction(xi, w, a, K)),
                                         ("OPT-C", lambda: R.optimal_reduction(xi, w, a, K, conservative=True)),
                                         ("TB", lambda: R.tail_block_rule(xi, w, a, K))):
                            t0 = time.perf_counter()
                            qv, qw = fn()
                            dt = time.perf_counter() - t0
                            Dp, Dm = discrepancies(xi, w, qv, qw, a)
                            rec[f"D_{name}"] = Dp + Dm
                            rec[f"Dminus_{name}"] = Dm
                            rec[f"atoms_{name}"] = len(qv)
                            rec[f"sec_{name}"] = dt
                        rows.append(rec)
        print(f"{ds} done ({time.time() - t_start:.0f}s)", flush=True)
    # K_min for several sample sizes (no reduction needed)
    kmin_rows = []
    for ds in data.MAIN_DATASETS:
        L = data.load_losses(ROOT / "data" / "processed", ds).to_numpy()
        for N_ in (500, 1000, 2000, 5000):
            for end in range(N_, len(L) + 1, step):
                vals = L[end - N_:end]
                for hl in HALF_LIVES:
                    xi, w = normalize_model(vals, ewma_weights(N_, hl))
                    for a in A_LIST:
                        kmin_rows.append({"dataset": ds, "N": N_, "end": end,
                                          "half_life": "uniform" if hl is None else int(hl), "a": a,
                                          "n_a": tail_profile(w, a)[1]})
    suffix = "_quick" if args.quick else ""
    pd.DataFrame(kmin_rows).to_csv(ROOT / "results" / f"e2_kmin_table{suffix}.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"e2_online{suffix}.csv", index=False)
    pd.DataFrame(curve).to_csv(ROOT / "results" / f"e2_kmin_curve{suffix}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
