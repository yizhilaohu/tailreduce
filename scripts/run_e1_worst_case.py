"""Experiment E1: exact worst-case regret of reduction rules on index loss data (uniform weights).

For every dataset, sample size N, window (seed), tail mass a, budget K and rule, computes the exact
overstatement D+ and understatement D- of the reduced model; by Theorem 1 the worst-case regret over
all losses that are nondecreasing in the index loss with range at most M equals M (D+ + D-).  Also
records the number of atoms each rule places in the tail (above xi_{n_a+1}; Corollary 1).

Usage:  python scripts/run_e1_worst_case.py [--quick]
Output: results/e1_worst_case.csv
"""
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tailreduce import data, discrepancies, normalize_model, tail_atoms, tail_profile  # noqa: E402
from tailreduce import reduction as R  # noqa: E402

A_LIST = (0.01, 0.05, 0.10)
K_LIST = (2, 3, 5, 10, 20, 50, 100, 200)
N_LIST = (500, 1000, 2000, 5000)
A_FREE = {"QM": R.quantile_medians, "W1": R.kmedians_1d, "W2": R.kmeans_1d}  # rules that do not use a
A_DEP = {"OPT": lambda xi, w, a, K: R.optimal_reduction(xi, w, a, K),
         "OPT-C": lambda xi, w, a, K: R.optimal_reduction(xi, w, a, K, conservative=True),
         "TB": R.tail_block_rule, "KW": R.keep_k_worst, "TW2": R.tail_kmeans}


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--windows", type=int, default=20)
    args = ap.parse_args()
    n_win = 3 if args.quick else args.windows
    N_list = (500, 2000) if args.quick else N_LIST
    rows = []
    t_start = time.time()
    # warm up numba
    R.kmeans_1d(np.array([3.0, 2.0, 1.0]), np.ones(3) / 3, 0.5, 2)
    R.kmedians_1d(np.array([3.0, 2.0, 1.0]), np.ones(3) / 3, 0.5, 2)
    for ds in data.MAIN_DATASETS:
        L = data.load_losses(ROOT / "data" / "processed", ds).to_numpy()
        T = len(L)
        for N in N_list:
            for win in range(n_win):
                rng_w = np.random.default_rng(seed_of("E1-window", ds, N, win))
                end = int(rng_w.integers(N, T + 1))
                xi, w = normalize_model(L[end - N:end])
                for K in K_LIST:
                    if K >= len(xi):
                        continue
                    models = {}
                    for name, fn in A_FREE.items():
                        t0 = time.perf_counter()
                        models[name] = (fn(xi, w, None, K), time.perf_counter() - t0)
                    rng_s = np.random.default_rng(seed_of("E1-sample", ds, N, win, K))
                    t0 = time.perf_counter()
                    models["RS"] = (R.random_sampling(xi, w, None, K, rng_s), time.perf_counter() - t0)
                    for a in A_LIST:
                        x, n_a = tail_profile(w, a)
                        J = float(np.diff(x[: n_a + 1]).max())
                        mods = dict(models)
                        for name, fn in A_DEP.items():
                            t0 = time.perf_counter()
                            mods[name] = (fn(xi, w, a, K), time.perf_counter() - t0)
                        for name, ((qv, qw), dt) in mods.items():
                            Dp, Dm = discrepancies(xi, w, qv, qw, a)
                            rows.append({"dataset": ds, "N": N, "window": win, "end": end, "a": a, "K": K,
                                         "n_a": n_a, "J": J, "method": name, "Dplus": Dp, "Dminus": Dm,
                                         "D": Dp + Dm, "atoms": len(qv), "tail_atoms": tail_atoms(xi, w, a, qv),
                                         "seconds": dt})
            print(f"{ds} N={N} done ({time.time() - t_start:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / ("e1_worst_case_quick.csv" if args.quick else "e1_worst_case.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")


if __name__ == "__main__":
    main()
