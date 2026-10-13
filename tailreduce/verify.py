"""Brute-force checks of every statement in the paper on small random instances.

Each check is independent of the closed forms it verifies:
  * Theorem 1 is checked by building decision problems and evaluating CVaR by sorting.
  * Theorem 2 is checked against a linear program that minimizes D+ + D- over the weights of
    every support, written directly from the definition of D+ and D- (one LP per support and per
    position of the saturation index), and against random models with atoms anywhere.
  * Theorem 3, the Corollary (tail atoms) and Proposition 4 are checked against their statements.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from .core import cvar, discrepancies, normalize_model, path_pairs, phi, tail_profile
from .reduction import (
    cyclic_bottleneck,
    keep_k_worst,
    optimal_reduction,
    quantile_medians,
    random_sampling,
    tail_atoms,
    tail_block_rule,
    two_sided_model,
)


def random_reference(rng, N, equal=False):
    vals = np.sort(rng.normal(size=N))[::-1]
    vals = np.unique(np.round(vals, 6))[::-1]
    N = len(vals)
    w = np.full(N, 1.0 / N) if equal else rng.dirichlet(np.full(N, rng.uniform(0.3, 3.0)))
    return normalize_model(vals, w)


# ---------------------------------------------------------------------------------- Theorem 1

def worst_instance_regret(xi, w, qv, qw, a, M=1.0, eps=1e-9):
    """Regret of the two-decision instance in the proof of Theorem 1(iii)."""
    P, Q = path_pairs(xi, w, qv, qw)
    d = phi(Q, a) - phi(P, a)
    i_plus, i_minus = int(np.argmax(d)), int(np.argmax(-d))
    pts = np.unique(np.concatenate([xi, qv]))[::-1]  # distinct merged points, decreasing

    def indicator(i):
        if i == 0:
            return lambda v: np.zeros_like(v, dtype=float)
        t = pts[i - 1]
        return lambda v: (np.asarray(v) >= t).astype(float)

    fp, fm = indicator(i_plus), indicator(i_minus)
    # g(0, .) = M f^- + k0,  g(1, .) = M f^+ + k1 with F_Q(0) = F_Q(1) - eps
    FQ0 = cvar(M * fm(qv), qw, a)
    FQ1 = cvar(M * fp(qv), qw, a)
    k1 = 0.0
    k0 = FQ1 - FQ0 - eps
    FQ = np.array([FQ0 + k0, FQ1 + k1])
    FP = np.array([cvar(M * fm(xi), w, a) + k0, cvar(M * fp(xi), w, a) + k1])
    zhat = int(np.argmin(FQ))
    return FP[zhat] - FP.min()


def random_monotone_losses(rng, lo, hi, M, m):
    """m random nondecreasing loss shapes with range exactly in [0, M] on [lo, hi]."""
    fs = []
    for _ in range(m):
        kind = rng.integers(3)
        if kind == 0:  # ramp
            l, u = np.sort(rng.uniform(lo, hi, size=2))
            u = max(u, l + 1e-6)
            fs.append(lambda v, l=l, u=u: M * np.clip((np.asarray(v) - l) / (u - l), 0, 1))
        elif kind == 1:  # step
            t = rng.uniform(lo, hi)
            closed = bool(rng.integers(2))
            fs.append(lambda v, t=t, c=closed: M * ((np.asarray(v) >= t) if c else (np.asarray(v) > t)).astype(float))
        else:  # random staircase mixture
            ts = np.sort(rng.uniform(lo, hi, size=4))
            hs = rng.dirichlet(np.ones(4))
            fs.append(lambda v, ts=ts, hs=hs: M * np.sum(hs[None, :] * (np.asarray(v)[:, None] >= ts[None, :]), axis=1))
    return fs


def check_theorem1(rng, n_inst=300):
    """Returns (max violation of the upper bound, max shortfall of the lower-bound construction)."""
    worst_violation = 0.0
    worst_tightness_gap = 0.0
    for _ in range(n_inst):
        N = int(rng.integers(2, 12))
        xi, w = random_reference(rng, N)
        a = float(rng.uniform(0.05, 1.0))
        K = int(rng.integers(1, 6))
        qv = rng.uniform(xi.min() - 1, xi.max() + 1, size=K)
        if rng.uniform() < 0.5:  # some atoms exactly on scenarios
            qv[: K // 2] = rng.choice(xi, size=K // 2)
        qv, qw = normalize_model(qv, rng.dirichlet(np.ones(K)))
        Dp, Dm = discrepancies(xi, w, qv, qw, a)
        M = float(rng.uniform(0.5, 3.0))
        # (iii) tightness construction
        reg = worst_instance_regret(xi, w, qv, qw, a, M=M, eps=1e-9)
        worst_tightness_gap = max(worst_tightness_gap, M * (Dp + Dm) - reg - 3 * M * 1e-9)
        # (i)-(ii) on random monotone decision problems
        lo, hi = min(xi.min(), qv.min()) - 0.5, max(xi.max(), qv.max()) + 0.5
        fs = random_monotone_losses(rng, lo, hi, M, int(rng.integers(2, 15)))
        FP = np.array([cvar(f(xi), w, a) for f in fs])
        FQ = np.array([cvar(f(qv), qw, a) for f in fs])
        dev = FQ - FP
        worst_violation = max(worst_violation, dev.max() - M * Dp, -dev.min() - M * Dm)
        mins = np.flatnonzero(FQ <= FQ.min() + 1e-14)
        regret = max(FP[mins]) - FP.min()
        worst_violation = max(worst_violation, regret - M * (Dp + Dm))
    return worst_violation, worst_tightness_gap


# ---------------------------------------------------------------------------------- Theorem 2

def lp_min_discrepancy(x, a, S, conservative=False):
    """min over weights on support S (1-based scenario indices) of D+ + D-, written from the definition.

    Path points j = 0..N with V_j = sum_{s in S, s <= j} q_s.  phi(V_j) = V_j / a if V_j <= a and 1
    otherwise; since V is nondecreasing, {j : V_j >= a} is a suffix starting at some r, which we enumerate.
    """
    N = len(x) - 1
    S = sorted(S)
    m = len(S)
    best = np.inf
    for r in range(1, N + 1):  # V_0 = 0 < a, V_N = 1 >= a
        # variables: q_1..q_m, dp, dm
        nvar = m + 2
        A, b = [], []
        for j in range(0, N + 1):
            row = np.zeros(nvar)
            for k, s in enumerate(S):
                if s <= j:
                    row[k] = 1.0
            if j < r:
                # V_j <= a
                A.append(row.copy()); b.append(a)
                # V_j / a - x_j <= dp
                r1 = row / a; r1[m] = -1.0; A.append(r1); b.append(x[j])
                # x_j - V_j / a <= dm
                r2 = -row / a; r2[m + 1] = -1.0; A.append(r2); b.append(-x[j])
            else:
                # V_j >= a
                A.append(-row); b.append(-a)
                # 1 - x_j <= dp
                r1 = np.zeros(nvar); r1[m] = -1.0; A.append(r1); b.append(x[j] - 1.0)
        Aeq = np.zeros((1, nvar)); Aeq[0, :m] = 1.0
        c = np.zeros(nvar); c[m] = 1.0; c[m + 1] = 1.0
        bounds = [(0, None)] * m + [(0, None), (0, 0) if conservative else (0, None)]
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[1.0], bounds=bounds, method="highs")
        if res.status == 0:
            best = min(best, res.fun)
    return best


def check_theorem2(rng, n_inst=120, n_random=1500):
    """Compares beta_K with brute force; returns dict of maximal discrepancies (all should be ~0)."""
    out = {"lp_gap": 0.0, "lp_gap_conservative": 0.0, "random_below_beta": 0.0,
           "constructed_gap": 0.0, "constructed_gap_conservative": 0.0, "too_many_atoms": 0,
           "conservative_understates": 0.0, "tau_family_violation": 0.0, "instances": 0}
    for _ in range(n_inst):
        N = int(rng.integers(2, 8))
        xi, w = random_reference(rng, N)
        N = len(w)
        a = float(rng.uniform(0.15, 1.0))
        x, n_a = tail_profile(w, a)
        delta = np.diff(x[: n_a + 1])
        for K in range(1, min(n_a, 3) + 1):
            out["instances"] += 1
            beta, _ = cyclic_bottleneck(delta, K)
            beta_c, _ = cyclic_bottleneck(delta, K, conservative=True)
            # brute force over supports on the scenarios
            bf = np.inf
            bf_c = np.inf
            for m in range(1, K + 1):
                for S in itertools.combinations(range(1, N + 1), m):
                    bf = min(bf, lp_min_discrepancy(x, a, S))
                    if 1 in S:
                        bf_c = min(bf_c, lp_min_discrepancy(x, a, S, conservative=True))
            out["lp_gap"] = max(out["lp_gap"], abs(bf - beta))
            out["lp_gap_conservative"] = max(out["lp_gap_conservative"], abs(bf_c - beta_c))
            # constructed models attain the value
            for cons, key in ((False, "constructed_gap"), (True, "constructed_gap_conservative")):
                qv, qw = optimal_reduction(xi, w, a, K, conservative=cons)
                if len(qv) > K:
                    out["too_many_atoms"] += 1
                Dp, Dm = discrepancies(xi, w, qv, qw, a)
                target = beta_c if cons else beta
                out[key] = max(out[key], abs(Dp + Dm - target))
                if cons:
                    out["conservative_understates"] = max(out["conservative_understates"], Dm)
            # the whole family of Theorem 2(a): tau in [T, beta - B] gives D- <= tau, D+ <= beta - tau
            _, S_opt = cyclic_bottleneck(delta, K)
            s_arr = np.array(S_opt)
            T, B = x[s_arr[0] - 1], 1.0 - x[s_arr[-1]]
            for tau in np.linspace(T, max(T, beta - B), 5):
                qv, qw = two_sided_model(xi, w, a, S_opt, tau=tau)
                Dp, Dm = discrepancies(xi, w, qv, qw, a)
                viol = max(abs(Dp + Dm - beta), Dm - tau, Dp - (beta - tau), len(qv) - K)
                out["tau_family_violation"] = max(out["tau_family_violation"], viol)
            # random models with atoms anywhere never beat beta
            for _ in range(n_random // 10):
                qv = rng.uniform(xi.min() - 0.5, xi.max() + 0.5, size=K)
                k_on = int(rng.integers(0, K + 1))
                qv[:k_on] = rng.choice(xi, size=k_on)
                qv, qw = normalize_model(qv, rng.dirichlet(np.full(K, rng.uniform(0.2, 2))))
                Dp, Dm = discrepancies(xi, w, qv, qw, a)
                out["random_below_beta"] = max(out["random_below_beta"], beta - (Dp + Dm))
    return out


def check_grid_lemma(rng, n_inst=500):
    """Moving atoms up to the nearest scenario (down to xi_1 above) never increases D+ or D-."""
    worst = 0.0
    for _ in range(n_inst):
        xi, w = random_reference(rng, int(rng.integers(2, 12)))
        a = float(rng.uniform(0.05, 1.0))
        K = int(rng.integers(1, 6))
        qv, qw = normalize_model(rng.uniform(xi.min() - 1, xi.max() + 1, size=K), rng.dirichlet(np.ones(K)))
        Dp, Dm = discrepancies(xi, w, qv, qw, a)
        asc = xi[::-1]
        idx = np.searchsorted(asc, qv, side="left")
        moved = np.where(idx >= len(asc), xi[0], asc[np.minimum(idx, len(asc) - 1)])
        mv, mw = normalize_model(moved, qw)
        Dp2, Dm2 = discrepancies(xi, w, mv, mw, a)
        worst = max(worst, Dp2 - Dp, Dm2 - Dm)
    return worst


def check_threshold(rng, n_inst=300):
    """K >= n_a gives an exact model; any model missing a tail scenario has D+ + D- >= its jump."""
    worst_exact = 0.0
    worst_missing = 0.0
    for _ in range(n_inst):
        xi, w = random_reference(rng, int(rng.integers(2, 15)))
        a = float(rng.uniform(0.05, 1.0))
        x, n_a = tail_profile(w, a)
        qv, qw = optimal_reduction(xi, w, a, n_a)
        Dp, Dm = discrepancies(xi, w, qv, qw, a)
        worst_exact = max(worst_exact, Dp + Dm)
        # drop the atom at a random tail scenario j and reweight arbitrarily
        j = int(rng.integers(1, n_a + 1))
        others = np.setdiff1d(np.arange(1, len(w) + 1), [j])
        if len(others) == 0:
            continue
        K = min(len(others), int(rng.integers(1, len(others) + 1)))
        S = rng.choice(others, size=K, replace=False)
        qv, qw = normalize_model(xi[S - 1], rng.dirichlet(np.ones(K)))
        Dp, Dm = discrepancies(xi, w, qv, qw, a)
        worst_missing = max(worst_missing, (x[j] - x[j - 1]) - (Dp + Dm))
    return worst_exact, worst_missing


# ---------------------------------------------------------------------------------- Theorem 3, Prop 4

def check_theorem3(rng, n_inst=400):
    """Theorem 3 and the Corollary on tail atoms.

    (a) beta_K >= 1/K - J;  (b) the tail-block rule has <= K atoms, D- = 0, D+ < 1/K, and is within J
    of beta_K;  (c) equal weights: beta^c_K = beta_K for every a, and for integer aN the closed form
    (ceil(n/K) - 1)/n, attained by the rule.  Corollary: a model with k atoms above xi_{n_a+1} has
    D+ + D- >= beta_{k+1}.
    """
    out = {"tb_understates": 0.0, "tb_above_1_over_K": -np.inf, "tb_minus_opt_minus_J": -np.inf,
           "lower_bound_violation": -np.inf, "equal_weights_gap_opt": 0.0, "equal_weights_gap_tb": 0.0,
           "equal_weights_gap_cons": 0.0, "equal_weights_cons_minus_opt_any_a": 0.0,
           "corollary_violation": -np.inf, "tb_too_many_atoms": 0}
    for _ in range(n_inst):
        N = int(rng.integers(5, 300))
        xi, w = random_reference(rng, N)
        a = float(rng.uniform(0.01, 0.6))
        x, n_a = tail_profile(w, a)
        delta = np.diff(x[: n_a + 1])
        J = float(delta.max())
        for K in sorted({1, 2, 3, 5, 10, max(1, n_a // 2), max(1, n_a - 1)}):
            if K >= n_a:
                continue
            qv, qw = tail_block_rule(xi, w, a, K)
            if len(qv) > K:
                out["tb_too_many_atoms"] += 1
            Dp, Dm = discrepancies(xi, w, qv, qw, a)
            beta, _ = cyclic_bottleneck(delta, K)
            out["tb_understates"] = max(out["tb_understates"], Dm)
            out["tb_above_1_over_K"] = max(out["tb_above_1_over_K"], Dp - 1.0 / K)
            out["tb_minus_opt_minus_J"] = max(out["tb_minus_opt_minus_J"], Dp + Dm - beta - J)
            out["lower_bound_violation"] = max(out["lower_bound_violation"], (1.0 / K - J) - beta)
            # Corollary: random models with atoms anywhere
            for _ in range(5):
                k = int(rng.integers(1, K + 2))
                qv = rng.uniform(xi.min() - 0.5, xi.max() + 0.5, size=k)
                qv, qw = normalize_model(qv, rng.dirichlet(np.ones(k)))
                kt = tail_atoms(xi, w, a, qv)
                Dp, Dm = discrepancies(xi, w, qv, qw, a)
                b_k1 = cyclic_bottleneck(delta, kt + 1)[0]
                out["corollary_violation"] = max(out["corollary_violation"], b_k1 - (Dp + Dm))
    for _ in range(n_inst):  # equal weights, aN integer
        n = int(rng.integers(2, 60))
        mult = int(rng.integers(2, 20))
        N = n * mult
        a = n / N
        xi = np.sort(rng.normal(size=N))[::-1]
        xi, w = normalize_model(xi + 1e-9 * np.arange(N)[::-1], np.full(N, 1.0 / N))
        if len(w) != N:
            continue
        x, n_a = tail_profile(w, a)
        assert n_a == n, (n_a, n)
        delta = np.diff(x[: n_a + 1])
        for K in sorted({1, 2, 3, max(1, n // 3), max(1, n - 1)}):
            if K >= n:
                continue
            closed = (int(np.ceil(n / K)) - 1) / n
            beta, _ = cyclic_bottleneck(delta, K)
            beta_c, _ = cyclic_bottleneck(delta, K, conservative=True)
            qv, qw = tail_block_rule(xi, w, a, K)
            Dp, Dm = discrepancies(xi, w, qv, qw, a)
            out["equal_weights_gap_opt"] = max(out["equal_weights_gap_opt"], abs(beta - closed))
            out["equal_weights_gap_cons"] = max(out["equal_weights_gap_cons"], abs(beta_c - closed))
            out["equal_weights_gap_tb"] = max(out["equal_weights_gap_tb"], abs(Dp + Dm - closed))
    for _ in range(n_inst):  # equal weights, any a: conservatism costs nothing
        N = int(rng.integers(5, 400))
        a = float(rng.uniform(0.005, 0.9))
        x, n_a = tail_profile(np.full(N, 1.0 / N), a)
        delta = np.diff(x[: n_a + 1])
        for K in sorted({1, 2, 3, 5, max(1, n_a // 3), max(1, n_a - 1)}):
            if K >= n_a:
                continue
            gap = cyclic_bottleneck(delta, K, conservative=True)[0] - cyclic_bottleneck(delta, K)[0]
            out["equal_weights_cons_minus_opt_any_a"] = max(out["equal_weights_cons_minus_opt_any_a"], gap)
    return out


def check_proposition4(rng, n_inst=200, n_mc=4000):
    """Closed forms of Proposition 4(a)-(b) and the two expectation bounds of 4(c).

    (b) is checked for every block position h (the h-th largest scenario of each block), with odd
    and even block sizes m: D+ = (m - h)/n and D- = (h - 1)/n.
    (c) upper bound E[D+ + D-] <= 4 / sqrt((1 - a) a K) (any weights; checked with equal and with
    random weights), lower bound E[D+ + D-] >= E[D-] >= 1 / (8 sqrt(a K)) for n even and K >= 4/a, and
    the high-probability DKW bound of the online supplement (Remark S1) (failure frequency at most delta).
    """
    out = {"kw_gap": 0.0, "qm_gap": 0.0, "block_position_gap": 0.0, "sampling_mean_lb_min_ratio": np.inf,
           "sampling_mean_ub_max_ratio": 0.0, "sampling_dkw_failure_rate_max": 0.0}
    for _ in range(n_inst):
        m = int(rng.integers(1, 30))  # block size (odd and even)
        K = int(rng.integers(2, 12))
        N = m * K
        n = int(rng.integers(max(m, K + 1), N + 1)) if max(m, K + 1) <= N else N  # a N = n integer
        a = n / N
        xi = np.sort(rng.normal(size=N))[::-1] + 1e-9 * np.arange(N)[::-1]
        xi, w = normalize_model(xi, np.full(N, 1.0 / N))
        if len(w) != N:
            continue
        if K < n:
            qv, qw = keep_k_worst(xi, w, a, K)
            Dp, Dm = discrepancies(xi, w, qv, qw, a)
            out["kw_gap"] = max(out["kw_gap"], abs(Dp - (1 - K / n)) + Dm)
        if m <= n:
            qv, qw = quantile_medians(xi, w, a, K)
            Dp, Dm = discrepancies(xi, w, qv, qw, a)
            gap = abs(Dp + Dm - (m - 1) / n)
            if m % 2 == 1:
                gap += abs(Dp - (m - 1) / (2 * n)) + abs(Dm - (m - 1) / (2 * n))
            out["qm_gap"] = max(out["qm_gap"], gap)
            for h in range(1, m + 1):  # h-th largest scenario of every block
                idx = np.arange(K) * m + (h - 1)  # 0-based positions in decreasing order
                qv, qw = normalize_model(xi[idx], np.full(K, 1.0 / K))
                Dp, Dm = discrepancies(xi, w, qv, qw, a)
                out["block_position_gap"] = max(out["block_position_gap"],
                                                abs(Dp - (m - h) / n) + abs(Dm - (h - 1) / n))
    # random sampling: mean bounds and DKW bound
    configs = [(400, 20, 80, False), (1000, 50, 100, False), (2000, 100, 400, False), (600, 60, 40, False),
               (2000, 100, 20, False), (500, 250, 10, False), (1000, 10, 5, False), (300, 3, 2, True),
               (800, 80, 30, True)]
    for (N, n, K, random_weights) in configs:
        a = n / N
        xi = np.sort(rng.normal(size=N))[::-1] + 1e-9 * np.arange(N)[::-1]
        w0 = rng.dirichlet(np.ones(N)) if random_weights else np.full(N, 1.0 / N)
        xi, w = normalize_model(xi, w0)
        sums, dms = [], []
        for _ in range(n_mc // 4):
            qv, qw = random_sampling(xi, w, a, K, rng)
            Dp, Dm = discrepancies(xi, w, qv, qw, a)
            sums.append(Dp + Dm)
            dms.append(Dm)
        out["sampling_mean_ub_max_ratio"] = max(out["sampling_mean_ub_max_ratio"],
                                                np.mean(sums) / (4.0 / np.sqrt((1 - a) * a * K)))
        if not random_weights and n % 2 == 0 and K >= 4 / a:
            out["sampling_mean_lb_min_ratio"] = min(out["sampling_mean_lb_min_ratio"],
                                                    np.mean(dms) / (1 / (8 * np.sqrt(a * K))))
        for delta in (0.05, 0.2):
            bound = (2 / a) * np.sqrt(np.log(2 / delta) / (2 * K))
            fail = np.mean(np.array(sums) > bound)
            out["sampling_dkw_failure_rate_max"] = max(out["sampling_dkw_failure_rate_max"], fail - delta)
    return out
