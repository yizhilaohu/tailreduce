# tailreduce: code for "Minimax-optimal scenario reduction for CVaR with a single risk factor"

This folder reproduces every number, table and figure of the paper and of its online supplement. It contains:

- the reduction algorithms of the paper (Algorithm 1 and the tail-block rule),
- six baselines,
- a numerical check of every theorem, corollary and proposition,
- the three experiments on public stock-index data,
- the scripts that write the LaTeX tables, the LaTeX macros for the numbers quoted in the text, and the figures.

## Contents

```
code/
├── README.md                 this file
├── LICENSE                   MIT (template; the authors fill in their names)
├── requirements.txt          exact package versions used for the paper
├── run_all.sh                one command for everything (download → tests → checks → experiments → tables/figures → PDFs)
├── tailreduce/               the library
│   ├── core.py               CVaR, tail profile, exact discrepancies D+ and D- (Theorem 1), model normalization
│   ├── reduction.py          Algorithm 1 (OPT, OPT-C), tail-block rule (TB), baselines KW, QM, W1, W2, TW2, RS
│   ├── hedge.py              CVaR tail-hedge linear program (decision family A)
│   ├── data.py               data download (pinned commit, SHA-256 checks) and loss series
│   └── verify.py             numerical checks of Theorems 1–3, Lemma 2, Corollary 1 and Proposition 4
├── scripts/
│   ├── download_data.py      fetch and preprocess the index data
│   ├── verify_theory.py      numerical check of all results          → results/verification.json
│   ├── run_e1_worst_case.py  worst-case regret, equal weights        → results/e1_worst_case.csv
│   ├── run_e2_online_weights.py  time-decayed weights               → results/e2_online.csv, e2_kmin_curve.csv, e2_kmin_table.csv
│   ├── run_e3_decisions.py   decisions from reduced scenario sets    → results/e3_decisions.csv
│   ├── make_tables.py        LaTeX tables and macros → ../paper/tables/*.tex (incl. facts.tex), results/facts.json
│   └── make_figures.py       Figures 2 and 3 → ../paper/figures/*.pdf|png
├── tests/test_theory.py      fast unit tests (pytest, about 30 s)
└── results/                  outputs of the full run used in the paper (CSV/JSON/logs)
```

`data/` is created by the download script. The data are not shipped (see "Data").

## Installation

Python 3.13 was used. Any recent Python 3 should work.

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` may not exist for your Python version. If so, install the latest
versions instead:

```bash
pip install numpy scipy pandas matplotlib numba rdata pytest
```

Numba is optional. Without it, the K-means and K-medians baselines run in pure Python and are much slower.
The PDFs need a standard TeX distribution with `pdflatex` and `bibtex` (TeX Live 2023 was used). The
Elsevier class `elsarticle.cls` and the style `model1-num-names.bst` are included in `../paper`.

## Reproduce everything

```bash
./run_all.sh            # full run: about 25 minutes on two cores; then rebuilds ../paper/*.pdf
./run_all.sh --quick    # smoke test on reduced grids (about 2 minutes); writes *_quick files only
```

The quick run does not touch the full results or the paper's tables and figures. The individual steps are:

```bash
python3 scripts/download_data.py          # data/raw, data/processed (checksums are verified)
python3 -m pytest -q tests                # unit tests, about 30 s
python3 scripts/verify_theory.py          # about 8 min, results/verification.json (Table S1)
python3 scripts/run_e1_worst_case.py      # about 7 min, results/e1_worst_case.csv (51,840 rows)
python3 scripts/run_e2_online_weights.py  # about 1.5 min, results/e2_online.csv (9,360 rows) and K_min files
python3 scripts/run_e3_decisions.py       # about 5.5 min, results/e3_decisions.csv (103,680 rows)
python3 scripts/make_tables.py            # ../paper/tables/*.tex, ../paper/tables/facts.tex, results/facts.json
python3 scripts/make_figures.py           # ../paper/figures/*.pdf|png
../paper/build.sh                         # main.pdf, main_journal.pdf, appendix.pdf
```

Times are for two cores of a 2.1 GHz Intel Xeon processor. All random choices use fixed seeds. These are
CRC32 hashes of strings such as `E1-window|SP500|2000|7`, so every window and every random problem can be
regenerated on its own. A second full run on the same machine reproduced all result files exactly, except
the measured run times. On other machines, results agree up to floating-point and linear-programming solver
tolerances. All quoted numbers are rounded.

`make_tables.py` prints a warning if the results no longer support a qualitative statement of the text
(for example, "the tail-block rule stayed below 1/K in every window").

## Where each number in the paper comes from

| Paper item | Produced by | From |
|---|---|---|
| Figure 1 (Example 1) | TikZ in `../paper/main.tex` | — |
| Figure 2; Tables S2–S6 | `make_figures.py`, `make_tables.py` | `results/e1_worst_case.csv` |
| Table 1, Figure 3; Tables S7–S10 | `make_tables.py`, `make_figures.py` | `results/e2_online.csv`, `results/e2_kmin_curve.csv`, `results/e2_kmin_table.csv` |
| Table 2; Tables S11–S13 | `make_tables.py` | `results/e3_decisions.csv` |
| Table S1 (numerical check) | `make_tables.py` | `results/verification.json` |
| Numbers quoted in the text (ratios, run times, ...) | `make_tables.py` → `../paper/tables/facts.tex` | `results/facts.json` |

## The reduction rules

All rules take sorted scenarios `xi` (largest first), weights `w`, the tail mass `a` and the budget `K`, and
return the values and weights of the reduced model.

| Name in the paper | Function | What it does |
|---|---|---|
| OPT | `optimal_reduction(xi, w, a, K)` | balanced minimax-optimal model (Theorem 2(a), Algorithm 1) |
| OPT-C | `optimal_reduction(xi, w, a, K, conservative=True)` | best model that never understates the tail (Theorem 2(b)) |
| TB | `tail_block_rule(xi, w, a, K)` | closed-form tail-block rule (Theorem 3(b)) |
| KW | `keep_k_worst` | the K largest scenarios (Proposition 4(a)) |
| QM | `quantile_medians` | weight 1/K at the quantiles (2k-1)/(2K) (Proposition 4(b)) |
| W1 | `kmedians_1d` | globally optimal weighted K-medians (Wasserstein-1-optimal reduction) |
| W2 | `kmeans_1d` | globally optimal weighted K-means (Wasserstein-2-optimal reduction) |
| TW2 | `tail_kmeans` | K-means with K-1 atoms on the tail scenarios, plus one atom for the rest |
| RS | `random_sampling(xi, w, a, K, rng)` | K independent draws of weight 1/K |

## Data

| Index | Source (pinned) | Daily returns |
|---|---|---|
| S&P 500 (`^GSPC`) | R package `quarks` 1.1.6, data set `SP500` | 6,036 (2000-01-04 to 2023-12-29) |
| FTSE 100 (`^FTSE`) | R package `quarks` 1.1.6, data set `FTSE100` | 6,059 (2000-01-05 to 2023-12-29) |
| Dow Jones Industrial Average (`^DJI`) | R package `quarks` 1.1.6, data set `DJI` | 6,036 (2000-01-04 to 2023-12-29) |

The `.rda` files are read from the read-only CRAN mirror on GitHub at commit
`15b9b176cd6e9b4d328fba0567b52bddc8447af2` (`https://raw.githubusercontent.com/cran/quarks/<commit>/data/`)
and checked against the SHA-256 values in `tailreduce/data.py`. The same files are in the CRAN source package
`quarks_1.1.6.tar.gz` (directory `data/`). The data are distributed with `quarks` under GPL-3 and are not
redistributed here. The series are price indices (no dividends). Scenarios are daily losses in percent,
`loss_t = -100 * log(P_t / P_{t-1})`, computed from closing prices.

## Using the library

```python
import numpy as np
from tailreduce import normalize_model, optimal_reduction, tail_block_rule, discrepancies

losses = np.random.default_rng(0).standard_t(4, size=2000)   # any scenario values
xi, w = normalize_model(losses)          # sorted decreasing, equal weights, ties merged
a, K = 0.05, 20

qv, qw = optimal_reduction(xi, w, a, K)                          # minimax-optimal model (Theorem 2)
qv_c, qw_c = optimal_reduction(xi, w, a, K, conservative=True)   # best model that never understates risk
qv_tb, qw_tb = tail_block_rule(xi, w, a, K)                      # closed-form rule (Theorem 3)

Dp, Dm = discrepancies(xi, w, qv, qw, a)
print(Dp + Dm)   # worst-case regret / M over all losses nondecreasing in the scenario value (Theorem 1)
```

Any weights can be passed to `normalize_model(values, weights)`, for example time-decayed weights.

## Tests

`python3 -m pytest -q tests` runs small versions of all checks (about 30 s). It also compares the
dynamic-programming K-means and K-medians with brute-force enumeration.

## License

The code is under the MIT license (`LICENSE`; the authors add their names before release). The data keep
their original license (GPL-3, R package `quarks`) and are not included.
