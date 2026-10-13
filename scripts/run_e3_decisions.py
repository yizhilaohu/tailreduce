"""Experiment E3: decisions computed from reduced scenario sets (two families of random problems).

Family A (bounded slope): CVaR-optimal tail hedge of an index position with put-spread layers
          (tailreduce.hedge), random layer grid, loadings increasing with depth, random premium limit.
Family B (with a jump):   capital buffer with a breach cost,
          g(z, xi) = c z + 1{xi > z} + lam (xi - z)^+,  z in a grid of 101 candidate buffers.
Both losses are nondecreasing in the daily index loss xi, so Theorem 1 bounds the in-sample regret
by M (D+ + D-).  Each rolling window has N = 2000 training days and 250 test days.  The reference
model gives the training days equal weights ("equal") or time-decayed weights with a half-life of
63 days ("h63").  For each problem we record the in-sample regret (under the reference model), the
bound, the out-of-sample CVaR of the decision over the test days, and, for family A, the premium
spent and the Sharpe ratio of the hedged position.

Every problem has its own seed (CRC32 of family, index, window and problem number), so it can be
regenerated on its own; the same problems are used for both weightings.

Usage:  python scripts/run_e3_decisions.py [--quick]
Output: results/e3_decisions.csv
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

from tailreduce import cvar, data, discrepancies, normalize_model, tail_atoms  # noqa: E402
from tailreduce import reduction as R  # noqa: E402
from tailreduce.hedge import HedgeProblem  # noqa: E402

A = 0.05
N_TRAIN = 2000
N_TEST = 250
K_LIST = (10, 20, 50)
METHODS = ("OPT", "OPT-C", "TB", "TW2", "W2", "W1", "QM", "RS", "KW")
WEIGHTINGS = {"equal": None, "h63": 63}


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def decay_weights(n, half_life):
    if half_life is None:
        return np.full(n, 1.0 / n)
    w = 2.0 ** (-np.arange(n)[::-1] / half_life)  # the most recent day (last) has the largest weight
    return w / w.sum()


def reduce_all(xi, w, K, rng):
    return {
        "OPT": R.optimal_reduction(xi, w, A, K),
        "OPT-C": R.optimal_reduction(xi, w, A, K, conservative=True),
        "TB": R.tail_block_rule(xi, w, A, K),
        "TW2": R.tail_kmeans(xi, w, A, K),
        "W2": R.kmeans_1d(xi, w, A, K),
        "W1": R.kmedians_1d(xi, w, A, K),
        "QM": R.quantile_medians(xi, w, A, K),
        "RS": R.random_sampling(xi, w, A, K, rng),
        "KW": R.keep_k_worst(xi, w, A, K),
    }


def cvar_columns(G, w, a):
    """CVaR of each column of the loss matrix G (n x m) under weights w (n)."""
    order = np.argsort(-G, axis=0, kind="mergesort")
    Gs = np.take_along_axis(G, order, axis=0)
    Ws = np.cumsum(w[order], axis=0)
    Wp = Ws - w[order]
    take = np.clip(np.minimum(Ws, a) - Wp, 0.0, None)
    return (take * Gs).sum(axis=0) / a


def buffer_losses(xi, zgrid, c, lam):
    X = np.asarray(xi, dtype=float)[:, None]
    Z = np.asarray(zgrid, dtype=float)[None, :]
    return c * Z + (X > Z) + lam * np.maximum(X - Z, 0.0)


def family_a_problem(ds, end, p, train):
    rng = np.random.default_rng(seed_of("E3", "A", ds, end, p))
    mL = int(rng.integers(3, 11))
    attach = np.unique(np.round(np.sort(rng.uniform(0.5, 12.0, size=mL + 1)), 3))
    loads = np.sort(rng.uniform(0.0, 1.5, size=len(attach) - 1))
    bud = float(rng.uniform(0.1, 1.0))
    return HedgeProblem(train, attach, loads, bud, A)


def family_b_problem(ds, end, p):
    rng = np.random.default_rng(seed_of("E3", "B", ds, end, p))
    return 10.0 ** rng.uniform(-2.5, 0.0), float(rng.uniform(0.0, 0.5))


def sharpe(pnl):
    sd = pnl.std()
    return pnl.mean() / sd * np.sqrt(252) if sd > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--problems", type=int, default=20)
    ap.add_argument("--step", type=int, default=250)
    args = ap.parse_args()
    n_prob = 4 if args.quick else args.problems
    step = 1000 if args.quick else args.step
    rows = []
    t_start = time.time()
    R.kmeans_1d(np.array([3.0, 2.0, 1.0]), np.ones(3) / 3, 0.5, 2)  # numba warm-up
    R.kmedians_1d(np.array([3.0, 2.0, 1.0]), np.ones(3) / 3, 0.5, 2)
    for ds in data.MAIN_DATASETS:
        L = data.load_losses(ROOT / "data" / "processed", ds).to_numpy()
        T = len(L)
        for end in range(N_TRAIN, T - N_TEST + 1, step):
            train, test = L[end - N_TRAIN:end], L[end:end + N_TEST]
            w_test = np.full(len(test), 1.0 / len(test))
            probs_a = [family_a_problem(ds, end, p, train) for p in range(n_prob)]
            probs_b = [family_b_problem(ds, end, p) for p in range(n_prob)]
            zgrid = np.quantile(train, np.linspace(0.90, 1.0, 101))
            for wname, hl in WEIGHTINGS.items():
                xi, w = normalize_model(train, decay_weights(len(train), hl))
                reds = {}
                for K in K_LIST:
                    rng_s = np.random.default_rng(seed_of("E3-sample", wname, ds, end, K))
                    reds[K] = reduce_all(xi, w, K, rng_s)
                disc = {(K, m): discrepancies(xi, w, *reds[K][m], A) for K in K_LIST for m in METHODS}
                ktail = {(K, m): tail_atoms(xi, w, A, reds[K][m][0]) for K in K_LIST for m in METHODS}
                base = {"weights": wname, "dataset": ds, "end": end}
                # ---------------- family A: tail hedge (LP)
                for p, hp in enumerate(probs_a):
                    z_f, f_star, t_full = hp.solve(xi, w)
                    loss_f = hp.losses(z_f, test)
                    oos_f = cvar(loss_f, w_test, A)
                    sh_f, sh_f_in = sharpe(-loss_f), sharpe(-hp.losses(z_f, train))
                    prem_f = float(hp.price @ z_f)
                    for K in K_LIST:
                        for m in METHODS:
                            qv, qw = reds[K][m]
                            z_q, _, t_q = hp.solve(qv, qw)
                            loss_q = hp.losses(z_q, test)
                            Dp, Dm = disc[(K, m)]
                            M = hp.oscillation_bound(min(xi.min(), qv.min()), max(xi.max(), qv.max()))
                            rows.append({**base, "family": "A", "problem": p, "K": K, "method": m,
                                         "f_star": f_star, "regret": hp.objective(z_q, xi, w) - f_star,
                                         "bound": M * (Dp + Dm), "Dplus": Dp, "Dminus": Dm,
                                         "oos_cvar": cvar(loss_q, w_test, A), "oos_cvar_full": oos_f,
                                         "oos_sharpe": sharpe(-loss_q), "oos_sharpe_full": sh_f,
                                         "is_sharpe": sharpe(-hp.losses(z_q, train)), "is_sharpe_full": sh_f_in,
                                         "premium": float(hp.price @ z_q), "premium_full": prem_f,
                                         "budget": float(hp.budget), "lp_seconds": t_q, "lp_seconds_full": t_full,
                                         "atoms": len(qv), "tail_atoms": ktail[(K, m)]})
                # ---------------- family B: capital buffer with breach cost (enumeration)
                for p, (c, lam) in enumerate(probs_b):
                    F_full = cvar_columns(buffer_losses(xi, zgrid, c, lam), w, A)
                    i_star = int(np.argmin(F_full))
                    f_star = F_full[i_star]
                    oos_f = cvar(buffer_losses(test, zgrid[i_star:i_star + 1], c, lam)[:, 0], w_test, A)
                    for K in K_LIST:
                        for m in METHODS:
                            qv, qw = reds[K][m]
                            F_q = cvar_columns(buffer_losses(qv, zgrid, c, lam), qw, A)
                            cand = np.flatnonzero(F_q <= F_q.min() + 1e-12)
                            i_q = int(cand[np.argmax(F_full[cand])])  # worst minimizer of the reduced problem
                            Dp, Dm = disc[(K, m)]
                            M = 1.0 + lam * (max(xi.max(), qv.max()) - min(xi.min(), qv.min()))
                            loss_test = buffer_losses(test, zgrid[i_q:i_q + 1], c, lam)[:, 0]
                            rows.append({**base, "family": "B", "problem": p, "K": K, "method": m,
                                         "f_star": f_star, "regret": F_full[i_q] - f_star,
                                         "bound": M * (Dp + Dm), "Dplus": Dp, "Dminus": Dm,
                                         "oos_cvar": cvar(loss_test, w_test, A), "oos_cvar_full": oos_f,
                                         "atoms": len(qv), "tail_atoms": ktail[(K, m)]})
            print(f"{ds} end={end} ({time.time() - t_start:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / ("e3_decisions_quick.csv" if args.quick else "e3_decisions.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")
    print("max(regret - worst-case bound) =", (df["regret"] - df["bound"]).max())


if __name__ == "__main__":
    main()
