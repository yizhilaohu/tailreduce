"""Core objects for CVaR scenario reduction with a single risk factor.

Notation (matches the paper):
    xi_1 > xi_2 > ... > xi_N      scenario values of the scalar risk factor (sorted decreasingly)
    w_1, ..., w_N > 0              scenario weights, summing to one
    W_j = w_1 + ... + w_j          cumulative weight of the j largest scenarios, W_0 = 0
    a in (0, 1]                    CVaR tail mass (a = 1 - alpha)
    phi(v) = min(1, v / a)         CVaR distortion
    x_j = phi(W_j)                 tail profile of the reference model
    n_a = min{ j : W_j >= a }      number of tail scenarios (= K_min, Corollary 1)

A (reduced) scenario model is a pair (values, weights) of 1-d arrays.  Values may be
anywhere on the real line; weights are nonnegative and sum to one.
"""
from __future__ import annotations

import numpy as np

# Relative tolerance used when deciding whether a cumulative weight has reached the tail mass a.
# Cumulative sums of floating-point weights may miss a by a few ulps; values within this
# tolerance of 1 in the profile scale are treated as 1.
PROFILE_TOL = 1e-12


def phi(v, a: float):
    """CVaR distortion phi(v) = min(1, v / a)."""
    return np.minimum(1.0, np.asarray(v, dtype=float) / a)


def normalize_model(values, weights=None, merge_ties: bool = True):
    """Return (values, weights) sorted decreasingly, with equal values merged and weights normalized.

    Atoms with zero weight are dropped.
    """
    v = np.asarray(values, dtype=float).ravel()
    if weights is None:
        w = np.full(v.shape, 1.0 / len(v))
    else:
        w = np.asarray(weights, dtype=float).ravel()
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    if np.any(w < -1e-15):
        raise ValueError("weights must be nonnegative")
    keep = w > 0
    v, w = v[keep], w[keep]
    order = np.argsort(-v, kind="mergesort")
    v, w = v[order], w[order]
    if merge_ties and len(v) > 1:
        new_group = np.r_[True, v[1:] != v[:-1]]
        idx = np.cumsum(new_group) - 1
        w = np.bincount(idx, weights=w)
        v = v[new_group]
    w = w / w.sum()
    return v, w


def tail_profile(w_sorted, a: float):
    """Tail profile of the reference model.

    Parameters
    ----------
    w_sorted : weights of the reference scenarios sorted by decreasing value.
    a : tail mass.

    Returns
    -------
    x : array of length N + 1 with x[j] = phi(W_j), x[0] = 0 (values within PROFILE_TOL of 1 set to 1).
    n_a : number of tail scenarios, n_a = min{ j : W_j >= a }.
    """
    w = np.asarray(w_sorted, dtype=float)
    W = np.r_[0.0, np.cumsum(w)]
    x = np.minimum(1.0, W / a)
    x[x >= 1.0 - PROFILE_TOL] = 1.0
    n_a = int(np.argmax(x >= 1.0))  # first index with x == 1 (exists because W_N = 1 >= a)
    return x, n_a


def kmin_threshold(w_sorted, a: float) -> int:
    """Minimum size of an exact reduced model (Corollary 1): K_min = n_a = #{ j : W_{j-1} < a }."""
    return tail_profile(w_sorted, a)[1]


def path_pairs(ref_values, ref_weights, mod_values, mod_weights):
    """All pairs (P(H), Q(H)) over half-lines H = (t, inf) and [t, inf), t real, plus H = empty set.

    Both models must have weights summing to one.  The pairs are the cumulative weights of the
    two models over the prefixes of the merged, decreasingly sorted, distinct support points.
    """
    pv = np.asarray(ref_values, dtype=float)
    qv = np.asarray(mod_values, dtype=float)
    pts = np.concatenate([pv, qv])
    pw = np.concatenate([np.asarray(ref_weights, dtype=float), np.zeros(len(qv))])
    qw = np.concatenate([np.zeros(len(pv)), np.asarray(mod_weights, dtype=float)])
    order = np.argsort(-pts, kind="mergesort")
    pts, pw, qw = pts[order], pw[order], qw[order]
    P = np.cumsum(pw)
    Q = np.cumsum(qw)
    last = np.r_[pts[1:] != pts[:-1], True]
    P = np.r_[0.0, P[last]]
    Q = np.r_[0.0, Q[last]]
    return P, Q


def discrepancies(ref_values, ref_weights, mod_values, mod_weights, a: float):
    """Exact overstatement D+ and understatement D- of the tail profile (equation (3) of the paper).

    D+ = sup_H [phi(Q(H)) - phi(P(H))],  D- = sup_H [phi(P(H)) - phi(Q(H))],
    where H ranges over all half-lines (t, inf), [t, inf) and the empty set.
    By Theorem 1 the worst-case regret over all losses with range at most M is M * (D+ + D-).
    """
    P, Q = path_pairs(ref_values, ref_weights, mod_values, mod_weights)
    xp = phi(P, a)
    xq = phi(Q, a)
    xp[xp >= 1.0 - PROFILE_TOL] = 1.0
    xq[xq >= 1.0 - PROFILE_TOL] = 1.0
    d = xq - xp
    return float(max(d.max(), 0.0)), float(max((-d).max(), 0.0))


def cvar(values, weights, a: float) -> float:
    """CVaR (average of the worst a-fraction of probability mass) of a discrete loss distribution.

    Uses the tail-average form: fill a bucket of size a with the largest losses first.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-v, kind="mergesort")
    v, w = v[order], w[order]
    W = np.cumsum(w)
    W_prev = np.r_[0.0, W[:-1]]
    take = np.clip(np.minimum(W, a) - W_prev, 0.0, None)
    return float(np.dot(take, v) / a)


def var_level(values, weights, a: float) -> float:
    """Value-at-Risk eta* = inf{ t : P(L > t) <= a } of a discrete loss distribution."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-v, kind="mergesort")
    v, w = v[order], w[order]
    W = np.cumsum(w)
    j = int(np.argmax(W >= a - PROFILE_TOL * a))
    return float(v[j])
