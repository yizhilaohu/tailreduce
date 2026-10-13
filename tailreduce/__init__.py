"""tailreduce: minimax-optimal scenario reduction for CVaR with a single risk factor.

Companion code for the paper "Minimax-optimal scenario reduction for CVaR with a single risk factor".
"""
from .core import (  # noqa: F401
    PROFILE_TOL,
    cvar,
    discrepancies,
    kmin_threshold,
    normalize_model,
    path_pairs,
    phi,
    tail_profile,
    var_level,
)
from .reduction import (  # noqa: F401
    METHODS,
    block_model,
    cyclic_bottleneck,
    cyclic_gap,
    keep_k_worst,
    kmeans_1d,
    kmedians_1d,
    optimal_reduction,
    quantile_medians,
    random_sampling,
    tail_atoms,
    tail_block_rule,
    tail_kmeans,
    two_sided_model,
)

__version__ = "1.1.0"
