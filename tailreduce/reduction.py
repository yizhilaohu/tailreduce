"""Scenario-reduction rules for CVaR with a single risk factor.

Rules implemented (all return a reduced model (values, weights) with at most K atoms):

    optimal_reduction(..., conservative=False)   minimax-optimal model (Theorem 2(a), Algorithm 1);
                                                 by default the balanced choice tau = beta/2
    optimal_reduction(..., conservative=True)    minimax-optimal model among models that never
                                                 understate risk (D- = 0), Theorem 2(b)
    tail_block_rule                              closed-form tail-block rule of Theorem 3
    tail_kmeans                                  baseline: exact K-means on the tail scenarios
                                                 (K-1 atoms) plus one atom for the rest
    keep_k_worst                                 the K largest scenarios, remaining weight on the K-th
    quantile_medians                             K equal-weight atoms at the quantiles (2k-1)/(2K)
                                                 (Kolmogorov-optimal / equal-weight W1-optimal discretization)
    kmedians_1d, kmeans_1d                       exact optimal 1-d K-medians (Wasserstein-1) and K-means
                                                 (Wasserstein-2) reductions with free weights (dynamic programming)
    random_sampling                              K i.i.d. draws, weight 1/K each

Reference models are given as (xi, w) with xi sorted strictly decreasingly (see core.normalize_model).
"""
from __future__ import annotations

import numpy as np

from .core import normalize_model, tail_profile

try:  # numba makes the 1-d clustering dynamic programs fast; a pure-numpy fallback is provided.
    import numba

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# ----------------------------------------------------------------------------------------------
# Cyclic bottleneck (Theorem 2)
# ----------------------------------------------------------------------------------------------

def cyclic_gap(delta, S):
    """beta(S): largest cyclic gap of the jumps delta_1..delta_n left uncovered by positions S.

    delta : array of length n with the profile jumps delta_j = x_j - x_{j-1}, j = 1..n (n = n_a).
    S     : iterable of 1-based positions in {1, ..., n}; must be nonempty.
    Gaps are sums of delta over maximal runs of positions not in S, read cyclically
    (the run after the last position of S continues with position 1).
    """
    d = np.asarray(delta, dtype=float)
    n = len(d)
    s = np.unique(np.asarray(list(S), dtype=int))
    if len(s) == 0:
        return float(d.sum())
    if s.min() < 1 or s.max() > n:
        raise ValueError("positions must lie in 1..n")
    X = np.r_[0.0, np.cumsum(d)]  # X[j] = delta_1 + ... + delta_j
    inner = [X[s[k + 1] - 1] - X[s[k]] for k in range(len(s) - 1)]
    wrap = X[s[0] - 1] + (X[n] - X[s[-1]])
    return float(max(inner + [wrap]))


def _next_table(X2, n, s, eps):
    """nxt[c] for 1-based positions c = 1..2n on the doubled cycle; sentinel 2n+1 maps to itself."""
    pos = np.arange(1, 2 * n + 1)
    rmax = np.searchsorted(X2, X2[pos] + s + eps, side="right") - 1
    nxt = np.empty(2 * n + 2, dtype=np.int64)
    nxt[0] = 0
    nxt[1: 2 * n + 1] = np.minimum(rmax + 1, 2 * n + 1)
    nxt[2 * n + 1] = 2 * n + 1
    return nxt


def _greedy_counts(nxt, n, starts):
    """count(p) = min{ t >= 1 : nxt^t(p) >= p + n } for each start p (binary lifting)."""
    levels = max(1, int(np.ceil(np.log2(n + 1))) + 1)
    ups = [nxt]
    for _ in range(1, levels):
        prev = ups[-1]
        ups.append(prev[prev])
    starts = np.asarray(starts, dtype=np.int64)
    cur = starts.copy()
    t = np.zeros(len(starts), dtype=np.int64)
    target = starts + n
    for k in range(levels - 1, -1, -1):
        cand = ups[k][cur]
        move = cand < target
        cur = np.where(move, cand, cur)
        t += move.astype(np.int64) << k
    return t + 1


def _greedy_support(nxt, n, p):
    """Positions chosen by the greedy rule started at p (1-based, returned in 1..n)."""
    S = [p]
    cur = p
    while True:
        nx = nxt[cur]
        if nx >= p + n:
            break
        S.append(nx)
        cur = nx
    return sorted(((q - 1) % n) + 1 for q in S)


def cyclic_bottleneck(delta, K, conservative=False, tol=1e-15, max_iter=200, balanced=False):
    """Minimize beta(S) over S subset of {1..n} with |S| <= K (and 1 in S if conservative).

    Returns (beta_K, S) with S a sorted list of 1-based positions.
    Implements the threshold search of Algorithm 1: for a threshold s, the greedy rule started at
    p places positions as far apart as allowed; binary lifting evaluates all starts at once.
    balanced=True (two-sided case only): among the starts whose greedy support is optimal, return
    the support that splits the gap across the ends of the tail most evenly, that is, minimizes
    max(T, B) with T = x_{s_1 - 1} and B = 1 - x_{s_m}; this allows the balanced model of Theorem 2(a).
    """
    d = np.asarray(delta, dtype=float)
    n = len(d)
    if K < 1:
        raise ValueError("K must be at least 1")
    if K >= n:
        return 0.0, list(range(1, n + 1))
    X2 = np.r_[0.0, np.cumsum(np.r_[d, d])]  # prefix sums on the doubled cycle, length 2n+1
    starts = np.array([1]) if conservative else np.arange(1, n + 1)
    eps = 1e-13

    def feasible(s):
        nxt = _next_table(X2, n, s, eps)
        counts = _greedy_counts(nxt, n, starts)
        i = int(np.argmin(counts))
        return counts[i] <= K, nxt, int(starts[i]), counts

    lo, hi = 0.0, float(d.sum())
    ok, nxt_hi, p_hi, counts_hi = feasible(hi)
    if not ok:  # cannot happen for K >= 1 (one position always suffices at s = total)
        raise RuntimeError("infeasible at the largest threshold")
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        ok, nxt_mid, p_mid, counts_mid = feasible(mid)
        if ok:
            hi, nxt_hi, p_hi, counts_hi = mid, nxt_mid, p_mid, counts_mid
        else:
            lo = mid
    S = _greedy_support(nxt_hi, n, p_hi)
    beta = cyclic_gap(d, S)
    if balanced and not conservative:
        X = np.r_[0.0, np.cumsum(d)]
        best = None
        for p in starts[counts_hi <= K]:
            S_p = _greedy_support(nxt_hi, n, int(p))
            if cyclic_gap(d, S_p) > beta + 1e-12:
                continue
            T, B = X[S_p[0] - 1], X[n] - X[S_p[-1]]
            key = (max(T, B), abs(T - B))
            if best is None or key < best[0]:
                best = (key, S_p)
        if best is not None:
            S = best[1]
    if conservative and 1 not in S:  # pragma: no cover - greedy from p = 1 always contains 1
        raise RuntimeError("conservative support must contain position 1")
    return cyclic_gap(d, S), S


# ----------------------------------------------------------------------------------------------
# Models from supports
# ----------------------------------------------------------------------------------------------

def block_model(xi, w, a, S):
    """Conservative model on positions S (1-based, must contain 1): each block [s_k, s_{k+1} - 1]
    of consecutive tail scenarios is merged into its largest scenario xi_{s_k}; the last block
    also receives all scenarios beyond the tail.  Has D- = 0 (Theorem 2(b))."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.unique(np.asarray(S, dtype=int))
    if s[0] != 1:
        raise ValueError("a conservative model must contain the largest scenario (position 1)")
    W = np.r_[0.0, np.cumsum(w)]
    bounds = np.r_[s, len(w) + 1]
    q = np.array([W[bounds[k + 1] - 1] - W[bounds[k] - 1] for k in range(len(s))])
    q[-1] = 1.0 - W[s[-1] - 1]
    return xi[s - 1].copy(), q


def two_sided_model(xi, w, a, S, x=None, tau=None):
    """Minimax-optimal weights on a support S (the model of Theorem 2(a)).

    With beta = beta(S), T = x_{s_1 - 1} (tail share above the first atom) and B = 1 - x_{s_m}
    (tail share below the last atom), any tau in [T, beta - B] gives the levels
        y_k = min{1, (x_{s_k} + x_{s_{k+1} - 1} + beta) / 2 - tau},  k < m,      y_m = 1,
    and the model with cumulative weights V_k = a y_k (k < m), V_m = 1 has D- <= tau and
    D+ <= beta - tau, so D+ + D- <= beta(S).  Default: the balanced choice tau = beta / 2,
    clipped to [T, beta - B].
    """
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    if x is None:
        x, n_a = tail_profile(w, a)
    else:
        n_a = int(np.argmax(np.asarray(x) >= 1.0))
    delta = np.diff(x[: n_a + 1])
    s = np.unique(np.asarray(S, dtype=int))
    m = len(s)
    beta = cyclic_gap(delta, s)
    T = x[s[0] - 1]
    B = 1.0 - x[s[-1]]
    lo, hi = T, beta - B
    if tau is None:
        tau = 0.5 * beta
    tau = float(min(max(tau, lo), max(lo, hi)))
    V = np.empty(m)
    for k in range(m - 1):
        y = min(1.0, 0.5 * (x[s[k]] + x[s[k + 1] - 1] + beta) - tau)
        V[k] = a * y
    V[m - 1] = 1.0
    V = np.maximum.accumulate(np.clip(V, 0.0, 1.0))
    q = np.diff(np.r_[0.0, V])
    keep = q > 0
    return xi[s - 1][keep].copy(), q[keep]


def optimal_reduction(xi, w, a, K, conservative=False, return_info=False, tau=None):
    """Minimax-optimal model with at most K atoms (Algorithm 1).

    conservative=False: minimizes D+ + D- over all models with at most K atoms anywhere on the line;
                        returns the model of Theorem 2(a) for the given tau (default: balanced).
    conservative=True : minimizes D+ over models with D- = 0 (never understates the tail profile);
                        returns the block model of Theorem 2(b).
    """
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    x, n_a = tail_profile(w, a)
    delta = np.diff(x[: n_a + 1])
    beta, S = cyclic_bottleneck(delta, K, conservative=conservative, balanced=not conservative)
    if conservative:
        vals, q = block_model(xi, w, a, S)
    else:
        vals, q = two_sided_model(xi, w, a, S, x=x, tau=tau)
    if return_info:
        return vals, q, {"beta": beta, "S": S, "n_a": n_a}
    return vals, q


def tail_block_rule(xi, w, a, K, return_info=False):
    """Tail-block rule (Theorem 3): tail scenario j goes to block k = ceil(K x_j), that is,
    (k-1)/K < x_j <= k/K; each block is merged into its largest scenario, and the last block also
    receives the weight of all scenarios beyond the tail.  D- = 0 and D+ < 1/K (Theorem 3(b)).
    For K >= n_a the rule keeps all tail scenarios (the exact model of Theorem 2(c))."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    x, n_a = tail_profile(w, a)
    j = np.arange(1, n_a + 1)
    if K >= n_a:
        S = j
    else:
        block = np.ceil(K * x[j] - 1e-9).astype(int)  # tolerance for exact multiples of 1/K
        block = np.clip(block, 1, K)
        first = np.r_[True, block[1:] != block[:-1]]
        S = j[first]
    vals, q = block_model(xi, w, a, S)
    if return_info:
        delta = np.diff(x[: n_a + 1])
        return vals, q, {"S": list(S), "beta": cyclic_gap(delta, S), "n_a": n_a}
    return vals, q


def keep_k_worst(xi, w, a, K):
    """The K largest scenarios with their weights; the K-th also receives all remaining weight."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    K = min(K, len(w))
    return block_model(xi, w, a, np.arange(1, K + 1))


def quantile_medians(xi, w, a, K):
    """K atoms of weight 1/K at the quantiles (2k-1)/(2K) of the reference model.

    For a continuous reference this is the Kolmogorov-optimal K-point discretization; for equal weights
    with N = mK, m odd, it is the median of each block of m consecutive scenarios, which is the
    Wasserstein-1-optimal model among models with K atoms of weight 1/K when N = mK (Proposition 4(b))."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    asc_v = xi[::-1]
    F = np.cumsum(w[::-1])
    u = (2.0 * np.arange(1, K + 1) - 1.0) / (2.0 * K)
    idx = np.searchsorted(F, u - 1e-12, side="left")  # Q(u) = min{ v : F(v) >= u }
    idx = np.minimum(idx, len(asc_v) - 1)
    vals = asc_v[idx]
    q = np.full(K, 1.0 / K)
    # merge equal atoms
    order = np.argsort(-vals, kind="mergesort")
    vals, q = vals[order], q[order]
    new = np.r_[True, vals[1:] != vals[:-1]]
    gid = np.cumsum(new) - 1
    return vals[new], np.bincount(gid, weights=q)


def random_sampling(xi, w, a, K, rng):
    """K i.i.d. draws from the reference model, each with weight 1/K (duplicates merged)."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    idx = rng.choice(len(w), size=K, replace=True, p=w / w.sum())
    cnt = np.bincount(idx, minlength=len(w))
    keep = cnt > 0
    vals = xi[keep]
    q = cnt[keep] / K
    return vals, q


# ----------------------------------------------------------------------------------------------
# Exact 1-d K-means and K-medians (Wasserstein-2 / Wasserstein-1 optimal reductions)
# ----------------------------------------------------------------------------------------------

if _HAVE_NUMBA:
    _jit = numba.njit(cache=True)
else:  # pragma: no cover
    def _jit(f):
        return f


@_jit
def _cost(i, j, xs, PW, PX, PXX, kind):
    """Within-cluster cost of the sorted points xs[i..j] (inclusive)."""
    wt = PW[j + 1] - PW[i]
    if wt <= 0.0:
        return 0.0
    if kind == 0:  # k-means: weighted sum of squared deviations from the mean
        sx = PX[j + 1] - PX[i]
        c = (PXX[j + 1] - PXX[i]) - sx * sx / wt
        return c if c > 0.0 else 0.0
    # k-medians: weighted absolute deviation from the weighted median
    target = PW[i] + 0.5 * wt
    r = np.searchsorted(PW, target)
    t = r - 1
    if t < i:
        t = i
    if t > j:
        t = j
    left_w = PW[t + 1] - PW[i]
    left_x = PX[t + 1] - PX[i]
    right_w = PW[j + 1] - PW[t + 1]
    right_x = PX[j + 1] - PX[t + 1]
    return xs[t] * left_w - left_x + right_x - xs[t] * right_w


@_jit
def _dp_cluster(xs, ws, K, kind):
    """Optimal partition of sorted points into K contiguous clusters (divide-and-conquer DP).

    Returns the start index of each cluster (length K, increasing, first = 0)."""
    n = xs.shape[0]
    PW = np.zeros(n + 1)
    PX = np.zeros(n + 1)
    PXX = np.zeros(n + 1)
    for i in range(n):
        PW[i + 1] = PW[i] + ws[i]
        PX[i + 1] = PX[i] + ws[i] * xs[i]
        PXX[i + 1] = PXX[i] + ws[i] * xs[i] * xs[i]
    D = np.full((K, n), np.inf)
    OPT = np.zeros((K, n), dtype=np.int64)
    for j in range(n):
        D[0, j] = _cost(0, j, xs, PW, PX, PXX, kind)
        OPT[0, j] = 0
    stack = np.zeros((4 * n + 16, 4), dtype=np.int64)
    for k in range(1, K):
        # D[k, j] = min_{i in [k, j]} D[k-1, i-1] + cost(i, j); opt(j) is nondecreasing in j
        top = 0
        stack[0, 0] = k
        stack[0, 1] = n - 1
        stack[0, 2] = k
        stack[0, 3] = n - 1
        top = 1
        while top > 0:
            top -= 1
            jlo = stack[top, 0]
            jhi = stack[top, 1]
            ilo = stack[top, 2]
            ihi = stack[top, 3]
            if jlo > jhi:
                continue
            mid = (jlo + jhi) // 2
            best = np.inf
            besti = ilo
            hi_i = ihi if ihi < mid else mid
            lo_i = ilo if ilo > k else k
            for i in range(lo_i, hi_i + 1):
                val = D[k - 1, i - 1] + _cost(i, mid, xs, PW, PX, PXX, kind)
                if val < best:
                    best = val
                    besti = i
            D[k, mid] = best
            OPT[k, mid] = besti
            if top + 2 >= stack.shape[0]:
                raise RuntimeError("stack overflow")
            stack[top, 0] = jlo
            stack[top, 1] = mid - 1
            stack[top, 2] = ilo
            stack[top, 3] = besti
            top += 1
            stack[top, 0] = mid + 1
            stack[top, 1] = jhi
            stack[top, 2] = besti
            stack[top, 3] = ihi
            top += 1
    starts = np.zeros(K, dtype=np.int64)
    j = n - 1
    for k in range(K - 1, -1, -1):
        i = OPT[k, j]
        starts[k] = i
        j = i - 1
    return starts


def _cluster_reduction(xi, w, K, kind):
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    xs = xi[::-1].copy()  # increasing
    ws = w[::-1].copy()
    n = len(xs)
    K = min(K, n)
    starts = _dp_cluster(xs, ws, K, 0 if kind == "kmeans" else 1)
    ends = np.r_[starts[1:], n]
    vals, q = [], []
    PW = np.r_[0.0, np.cumsum(ws)]
    for s, e in zip(starts, ends):
        cw = ws[s:e]
        cx = xs[s:e]
        wt = cw.sum()
        if kind == "kmeans":
            # a cluster of one value is represented by that value exactly (a floating-point mean can be off by
            # one ulp, which would move the atom to the other side of its scenario)
            c = float(cx[0]) if cx[0] == cx[-1] else float(np.dot(cw, cx) / wt)
        else:
            t = int(np.searchsorted(PW, PW[s] + 0.5 * wt, side="left")) - 1
            t = min(max(t, s), e - 1)
            c = float(xs[t])
        vals.append(c)
        q.append(wt)
    vals = np.array(vals)
    q = np.array(q)
    order = np.argsort(-vals, kind="mergesort")
    return vals[order], q[order] / q.sum()


def kmeans_1d(xi, w, a, K):
    """Exact weighted K-means of the scenarios (Wasserstein-2-optimal K-point reduction)."""
    return _cluster_reduction(xi, w, K, "kmeans")


def kmedians_1d(xi, w, a, K):
    """Exact weighted K-medians with centers at scenarios and nearest-center redistribution
    (the optimal Kantorovich/Wasserstein-1 reduction of Dupacova, Growe-Kuska and Romisch, in 1-d)."""
    return _cluster_reduction(xi, w, K, "kmedians")


def tail_kmeans(xi, w, a, K):
    """Tail-restricted K-means (baseline TW2): exact weighted K-means with K - 1 clusters on the
    n_a tail scenarios, plus one atom at the weighted mean of the other scenarios."""
    xi = np.asarray(xi, dtype=float)
    w = np.asarray(w, dtype=float)
    x, n_a = tail_profile(w, a)
    if K < 2 or n_a >= len(w):
        return kmeans_1d(xi, w, a, K)
    tv, tw = _cluster_reduction(xi[:n_a], w[:n_a], K - 1, "kmeans")
    tw = tw * w[:n_a].sum()
    bw = w[n_a:].sum()
    bv = float(np.dot(w[n_a:], xi[n_a:]) / bw)
    return normalize_model(np.r_[tv, bv], np.r_[tw, bw])


def tail_atoms(xi, w, a, values):
    """Number of atoms of a model that lie above xi_{n_a + 1} (the Corollary of Theorem 3)."""
    xi = np.asarray(xi, dtype=float)
    _, n_a = tail_profile(np.asarray(w, dtype=float), a)
    cut = xi[n_a] if n_a < len(xi) else -np.inf
    return int(np.sum(np.asarray(values, dtype=float) > cut))


METHODS = {
    "OPT": lambda xi, w, a, K, rng=None: optimal_reduction(xi, w, a, K, conservative=False),
    "OPT-C": lambda xi, w, a, K, rng=None: optimal_reduction(xi, w, a, K, conservative=True),
    "TB": lambda xi, w, a, K, rng=None: tail_block_rule(xi, w, a, K),
    "KW": lambda xi, w, a, K, rng=None: keep_k_worst(xi, w, a, K),
    "QM": lambda xi, w, a, K, rng=None: quantile_medians(xi, w, a, K),
    "W1": lambda xi, w, a, K, rng=None: kmedians_1d(xi, w, a, K),
    "W2": lambda xi, w, a, K, rng=None: kmeans_1d(xi, w, a, K),
    "RS": lambda xi, w, a, K, rng=None: random_sampling(xi, w, a, K, rng),
    "TW2": lambda xi, w, a, K, rng=None: tail_kmeans(xi, w, a, K),
}
