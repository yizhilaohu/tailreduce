"""CVaR-optimal tail hedge of an index exposure (family A of the decision experiment, Section 5.4).

The decision maker holds one unit of an index and buys z_k in [0, 1] units of K_L 'layers'
(put spreads on the daily loss): layer k pays p_k(xi) = min((xi - l_{k-1})^+, l_k - l_{k-1})
when the daily loss is xi (in percent).  Layer prices include a loading that grows with depth:
pi_k = (1 + theta_k) E[p_k(xi)] under the training data.  The hedged loss is
        g(z, xi) = xi - sum_k z_k (p_k(xi) - pi_k),
which is nondecreasing in xi for every z in [0, 1]^m (slope 1 - z_k on layer k).  The premium
budget is sum_k pi_k z_k <= B.  The CVaR at tail mass a is minimized by the linear program of
Rockafellar and Uryasev over any scenario model (values, weights).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .core import cvar


def layer_payoffs(xi, attach):
    """Matrix of layer payoffs, shape (len(xi), len(attach) - 1)."""
    xi = np.asarray(xi, dtype=float)[:, None]
    lo = np.asarray(attach[:-1], dtype=float)[None, :]
    hi = np.asarray(attach[1:], dtype=float)[None, :]
    return np.clip(xi - lo, 0.0, hi - lo)


class HedgeProblem:
    def __init__(self, train_losses, attach, loadings, budget_fraction, a):
        self.attach = np.asarray(attach, dtype=float)
        self.m = len(attach) - 1
        self.loadings = np.asarray(loadings, dtype=float)
        pay = layer_payoffs(train_losses, self.attach)
        self.fair = pay.mean(axis=0)
        self.price = (1.0 + self.loadings) * self.fair
        self.budget = budget_fraction * self.price.sum()
        self.a = a

    def losses(self, z, xi):
        """g(z, xi) for a vector of scenarios."""
        pay = layer_payoffs(xi, self.attach)
        return np.asarray(xi, dtype=float) - pay @ z + float(self.price @ z)

    def objective(self, z, values, weights):
        return cvar(self.losses(z, values), weights, self.a)

    def oscillation_bound(self, lo, hi):
        """M with sup_xi g - inf_xi g <= M for every feasible z, for scenarios in [lo, hi]."""
        return float(hi - lo)  # g has slope in [0, 1] in xi

    def solve(self, values, weights):
        """Minimize CVaR_a over z in [0,1]^m, sum pi_k z_k <= B.  Returns (z, objective, seconds)."""
        import time

        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        n, m, a = len(v), self.m, self.a
        pay = layer_payoffs(v, self.attach)
        # variables: z (m), eta (1), u (n);  minimize eta + (1/a) sum w_i u_i
        c = np.r_[np.zeros(m), 1.0, w / a]
        # u_i >= xi_i - sum_k z_k (pay_ik - pi_k) - eta   <=>   sum_k (pi_k - pay_ik) z_k - eta - u_i <= -xi_i
        Az = sp.csr_matrix(self.price[None, :] - pay)
        A1 = sp.hstack([Az, sp.csr_matrix(-np.ones((n, 1))), -sp.identity(n, format="csr")], format="csr")
        A2 = sp.hstack([sp.csr_matrix(self.price[None, :]), sp.csr_matrix((1, 1 + n))], format="csr")
        A = sp.vstack([A1, A2], format="csr")
        b = np.r_[-v, self.budget]
        bounds = [(0.0, 1.0)] * m + [(None, None)] + [(0.0, None)] * n
        t0 = time.perf_counter()
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        dt = time.perf_counter() - t0
        if res.status != 0:
            raise RuntimeError(res.message)
        z = np.clip(res.x[:m], 0.0, 1.0)
        return z, float(res.fun), dt
