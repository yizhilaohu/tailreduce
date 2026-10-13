"""Unit tests: every statement of the paper on small random instances (fast subset of scripts/verify_theory.py)."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tailreduce import (  # noqa: E402
    cvar,
    cyclic_bottleneck,
    cyclic_gap,
    discrepancies,
    kmeans_1d,
    kmedians_1d,
    normalize_model,
    tail_profile,
)
from tailreduce import verify  # noqa: E402

TOL = 1e-9


def test_cvar_three_ways():
    # Example: losses 100, 60, 40, 10 with probabilities .05, .10, .25, .60 and a = .2 -> CVaR = 65
    assert abs(cvar([100, 60, 40, 10], [0.05, 0.10, 0.25, 0.60], 0.2) - 65.0) < 1e-12


def test_theorem1_identity():
    viol, gap = verify.check_theorem1(np.random.default_rng(1), n_inst=80)
    assert viol <= 1e-9
    assert gap <= 1e-7


def test_grid_lemma():
    assert verify.check_grid_lemma(np.random.default_rng(2), n_inst=200) <= 1e-12


def test_theorem2_bruteforce():
    out = verify.check_theorem2(np.random.default_rng(3), n_inst=25, n_random=400)
    assert out["lp_gap"] <= 1e-7
    assert out["lp_gap_conservative"] <= 1e-7
    assert out["random_below_beta"] <= 1e-12
    assert out["constructed_gap"] <= 1e-9
    assert out["constructed_gap_conservative"] <= 1e-9
    assert out["too_many_atoms"] == 0
    assert out["conservative_understates"] <= 1e-12
    assert out["tau_family_violation"] <= 1e-9


def test_threshold():
    worst_exact, worst_missing = verify.check_threshold(np.random.default_rng(4), n_inst=200)
    assert worst_exact <= 1e-12
    assert worst_missing <= 1e-12


def test_theorem3():
    out = verify.check_theorem3(np.random.default_rng(5), n_inst=100)
    assert out["tb_understates"] <= 1e-12
    assert out["tb_above_1_over_K"] < 1e-9
    assert out["tb_minus_opt_minus_J"] < 1e-9
    assert out["lower_bound_violation"] <= 1e-12
    assert out["equal_weights_gap_opt"] <= 1e-9
    assert out["equal_weights_gap_cons"] <= 1e-9
    assert out["equal_weights_gap_tb"] <= 1e-9
    assert out["equal_weights_cons_minus_opt_any_a"] <= 1e-9
    assert out["corollary_violation"] <= 1e-12
    assert out["tb_too_many_atoms"] == 0


def test_proposition4_closed_forms():
    out = verify.check_proposition4(np.random.default_rng(6), n_inst=100, n_mc=800)
    assert out["kw_gap"] <= 1e-9
    assert out["qm_gap"] <= 1e-9
    assert out["block_position_gap"] <= 1e-9
    assert out["sampling_mean_lb_min_ratio"] >= 1.0
    assert out["sampling_mean_ub_max_ratio"] <= 1.0
    assert out["sampling_dkw_failure_rate_max"] <= 0.0


def test_cyclic_gap_small():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(cyclic_gap(d, [1]) - 0.9) < 1e-15
    assert abs(cyclic_gap(d, [2, 4]) - 0.3) < 1e-15  # gaps: {3} -> .3 ; {1} -> .1
    beta, S = cyclic_bottleneck(d, 2)
    best = min(cyclic_gap(d, S2) for m in (1, 2) for S2 in itertools.combinations(range(1, 5), m))
    assert abs(beta - best) < 1e-12


def _bruteforce_cluster(xs, ws, K, kind):
    n = len(xs)
    best = np.inf
    for cuts in itertools.combinations(range(1, n), K - 1):
        b = (0,) + cuts + (n,)
        tot = 0.0
        for s, e in zip(b[:-1], b[1:]):
            cx, cw = xs[s:e], ws[s:e]
            if kind == "kmeans":
                mu = np.dot(cw, cx) / cw.sum()
                tot += np.dot(cw, (cx - mu) ** 2)
            else:
                tot += min(np.dot(cw, np.abs(cx - c)) for c in cx)
        best = min(best, tot)
    return best


def test_clustering_dp_optimal():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(3, 9))
        xs = np.sort(rng.normal(size=n))
        ws = rng.dirichlet(np.ones(n))
        xi, w = normalize_model(xs, ws)
        for K in range(1, n + 1):
            for kind, fn in (("kmeans", kmeans_1d), ("kmedians", kmedians_1d)):
                qv, qw = fn(xi, w, 0.5, K)
                asc_x, asc_w = xi[::-1], w[::-1]
                # cost of the returned model = transport cost of assigning each point to the nearest center
                if kind == "kmeans":
                    cost = sum(wi * min((xv - c) ** 2 for c in qv) for xv, wi in zip(asc_x, asc_w))
                else:
                    cost = sum(wi * min(abs(xv - c) for c in qv) for xv, wi in zip(asc_x, asc_w))
                assert cost <= _bruteforce_cluster(asc_x, asc_w, K, kind) + 1e-10
                assert abs(qw.sum() - 1) < 1e-12
