"""LaTeX tables (main text and online supplement), LaTeX macros with the numbers quoted in the text
(paper/tables/facts.tex), and a JSON file with the same numbers (results/facts.json).

Usage: python scripts/make_tables.py [--outdir ../paper/tables]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["OPT", "OPT-C", "TB", "TW2", "W2", "W1", "QM", "RS", "KW"]
E1_ORDER = ["OPT", "TW2", "W2", "W1", "QM", "RS", "KW"]  # OPT = OPT-C = TB for equal weights
LAB = {"OPT": "OPT", "OPT-C": "OPT-C", "TB": "TB", "TW2": "TW2 (tail $K$-means)", "W2": "W2 ($K$-means)",
       "W1": "W1 ($K$-medians)", "QM": "QM (quantile medians)", "RS": "RS (random sampling)",
       "KW": "KW (keep $K$ worst)"}
HL_NAME = {"uniform": r"$\infty$ (equal)", "1000": "1000 days", "250": "250 days", "63": "63 days"}
HL_ORDER = ["uniform", "1000", "250", "63"]


def f3(x):
    """Three decimals; tiny values print as 0.000 and negatives get a math minus sign."""
    if abs(x) < 5e-4:
        return "0.000"
    return f"{x:.3f}" if x >= 0 else f"$-${-x:.3f}"


def sgn(x, digits=1):
    """Signed number with a math minus or plus sign, e.g. $-$5.8 or $+$0.7."""
    v = round(float(x), digits)
    if v == 0:
        return f"{0:.{digits}f}"
    return f"$+${v:.{digits}f}" if v > 0 else f"$-${-v:.{digits}f}"


def bp(x):
    """Regret in basis points: two decimals below 10, one decimal below 100, none above."""
    x = float(x)
    if abs(x) < 0.005:
        return "0.00"
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:.0f}"


def bp_max(x):
    x = float(x)
    if abs(x) < 0.05:
        return "0.0"
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def table(lines_head, rows, lines_tail):
    return "\n".join(lines_head + rows + lines_tail) + "\n"


# ------------------------------------------------------------------------------------------------ E1
def tab_e1(out, facts):
    df = pd.read_csv(ROOT / "results" / "e1_worst_case.csv")
    d5000 = df[df.N == 5000].groupby(["a", "K", "method"]).D.mean().unstack()
    # ratios to the optimum (N = 5000) for the free-weight rules, budgets 10..200
    ratios = {}
    for m in ("W1", "W2", "TW2", "QM"):
        r = (d5000[m] / d5000["OPT"]).replace([np.inf], np.nan)
        r = r[(r.index.get_level_values("K") >= 10)]
        ratios[m] = r.dropna()
    facts["e1_ratio_W_range_K10_200"] = [float(min(ratios["W1"].min(), ratios["W2"].min())),
                                         float(max(ratios["W1"].max(), ratios["W2"].max()))]
    facts["e1_ratio_TW2_range_K10_200"] = [float(ratios["TW2"].min()), float(ratios["TW2"].max())]
    facts["e1_ratio_QM_a05_K20"] = float(d5000.loc[(0.05, 20), "QM"] / d5000.loc[(0.05, 20), "OPT"])
    facts["e1_ratio_W2_OPT_a05_K20"] = float(d5000.loc[(0.05, 20), "W2"] / d5000.loc[(0.05, 20), "OPT"])
    facts["e1_ratio_W1_OPT_a05_K20"] = float(d5000.loc[(0.05, 20), "W1"] / d5000.loc[(0.05, 20), "OPT"])
    facts["e1_ratio_W2_OPT_a05_K50"] = float(d5000.loc[(0.05, 50), "W2"] / d5000.loc[(0.05, 50), "OPT"])
    facts["e1_ratio_W1_OPT_a05_K50"] = float(d5000.loc[(0.05, 50), "W1"] / d5000.loc[(0.05, 50), "OPT"])
    # tail atoms (Corollary 1), N = 5000
    kt = df[df.N == 5000].groupby(["a", "K", "method"]).tail_atoms.mean().unstack()
    facts["e1_tail_atoms_a05_K50"] = {m: float(kt.loc[(0.05, 50), m]) for m in ("W1", "W2", "QM", "TW2", "OPT")}
    facts["e1_tail_atoms_a05_K20"] = {m: float(kt.loc[(0.05, 20), m]) for m in ("W1", "W2", "QM", "TW2", "OPT")}
    # fitted slopes of log D in log K (K = 10..200, N = 5000, K < n_a) for each a
    slopes = {}
    for m in ("OPT", "W1", "W2", "QM", "RS", "TW2"):
        sl = []
        for a in (0.01, 0.05, 0.10):
            s = d5000.loc[a][m]
            s = s[(s.index >= 10) & (s > 1e-9) & (s < 0.999)]
            if len(s) >= 3:
                sl.append(float(np.polyfit(np.log(s.index.values), np.log(s.values), 1)[0]))
        slopes[m] = sl
    facts["e1_slopes_logK"] = slopes
    time5000 = df[df.N == 5000].groupby("method").seconds.median() * 1e3
    facts["e1_time_ms_N5000"] = {k: float(v) for k, v in time5000.items()}
    t20 = df[(df.N == 5000) & (df.K == 20)].groupby("method").seconds.median() * 1e3
    facts["e1_time_ms_N5000_K20"] = {k: float(v) for k, v in t20.items()}
    facts["e1_rows"] = int(len(df))
    sens = df[(df.a == 0.05) & (df.K == 20)].groupby(["method", "N"]).D.mean().unstack()
    facts["e1_K20_a05_by_N"] = {m: {int(n): float(v) for n, v in sens.loc[m].items()} for m in sens.index}
    sd = df[(df.N == 5000)].groupby(["method", "a", "K"]).D.std()
    facts["e1_sd_max_free_N5000"] = {m: float(sd.loc[m].max()) for m in ("W1", "W2", "TW2", "RS")}
    facts["e1_sd_max_fixed_N5000"] = float(max(sd.loc[m].max() for m in ("OPT", "OPT-C", "TB", "QM", "KW")))

    # supplement: full grid, one table per tail mass
    for a in (0.01, 0.05, 0.10):
        s = df[df.a == a].groupby(["N", "K", "method"]).D.mean().unstack()[E1_ORDER]
        head = [r"\begin{table}[ht]", r"\centering", r"\small",
                rf"\caption{{Worst-case regret divided by $M$ ($\Dp+\Dm$), tail mass $a={a:g}$, equal weights: mean over "
                r"3 indices $\times$ 20 windows. OPT, OPT-C, and TB coincide; OPT is exact for $K\ge aN$.}",
                rf"\label{{tab:app_e1_{int(round(a * 100)):02d}}}",
                r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}rrrrrrrrr@{}}", r"\toprule",
                r"$N$ & $K$ & OPT & TW2 & W2 & W1 & QM & RS & KW \\", r"\midrule"]
        rows, last_N = [], None
        for (N_, K_), r in s.iterrows():
            rows.append(f"{N_ if N_ != last_N else ''} & {K_} & " + " & ".join(f3(v) for v in r.to_numpy()) + r" \\")
            last_N = N_
            if K_ == 200 and N_ != 5000:
                rows.append(r"\addlinespace")
        (out / f"tab_app_e1_a{int(round(a * 100)):02d}.tex").write_text(
            table(head, rows, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    # supplement: by index, N = 2000, a = 0.05 (mean and standard deviation)
    s = df[(df.N == 2000) & (df.a == 0.05)].groupby(["dataset", "K", "method"]).D.agg(["mean", "std"])
    head = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{$\Dp+\Dm$ by index, $N=2000$, $a=0.05$, equal weights: mean (standard deviation over "
            r"20 windows). OPT, QM, and KW do not depend on the data.}",
            r"\label{tab:app_e1_ds}", r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
            r"Index & $K$ & OPT & TW2 & W2 & W1 & QM & RS & KW \\", r"\midrule"]
    rows = []
    for ds in ("SP500", "FTSE100", "DJIA"):
        for K in (5, 10, 20, 50):
            cells = []
            for m in E1_ORDER:
                mu, sdv = s.loc[(ds, K, m), "mean"], s.loc[(ds, K, m), "std"]
                if m in ("OPT", "QM", "KW"):
                    cells.append(f"{mu:.3f}")
                else:
                    cells.append(f"{mu:.3f} (<0.01)" if sdv < 5e-3 else f"{mu:.3f} ({sdv:.2f})")
            name = {"SP500": "S\\&P 500", "FTSE100": "FTSE 100", "DJIA": "DJIA"}[ds] if K == 5 else ""
            rows.append(f"{name} & {K} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    rows[-1] = r"\bottomrule"
    (out / "tab_app_e1_ds.tex").write_text(table(head, rows, [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    # supplement: atoms in the tail and ratios to OPT (N = 5000)
    head = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Equal weights, $N=5000$: mean number of atoms above $\xi_{n_a+1}$ (tail atoms, "
            r"Corollary~\ref{cor:tail}) and ratio of $\Dp+\Dm$ to that of OPT. QM places about $aK$ atoms in the "
            r"tail; the free-weight rules W1 and W2 place more, and TW2 places $K-1$.}",
            r"\label{tab:app_tailatoms}", r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{@{}rrrrrrrrrr@{}}", r"\toprule",
            r" & & \multicolumn{4}{c}{Tail atoms} & \multicolumn{4}{c}{Ratio to OPT} \\",
            r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
            r"$a$ & $K$ & QM & W1 & W2 & TW2 & QM & W1 & W2 & TW2 \\", r"\midrule"]
    rows = []
    for a in (0.01, 0.05, 0.10):
        for K in (10, 20, 50, 100, 200):
            if K >= int(round(a * 5000)):
                continue
            cells = [f"{kt.loc[(a, K), m]:.1f}" for m in ("QM", "W1", "W2", "TW2")]
            cells += [f"{d5000.loc[(a, K), m] / d5000.loc[(a, K), 'OPT']:.1f}" for m in ("QM", "W1", "W2", "TW2")]
            rows.append(f"{a:g} & {K} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    rows[-1] = r"\bottomrule"
    (out / "tab_app_tailatoms.tex").write_text(table(head, rows, [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))


# ------------------------------------------------------------------------------------------------ E2
def tab_e2(out, facts):
    df = pd.read_csv(ROOT / "results" / "e2_online.csv")
    a = 0.05
    sub = df[df.a == a]
    head = [r"\begin{table}[t]", r"\centering",
            r"\caption{Time-decayed weights ($N=2000$, $a=0.05$, windows every 63 trading days, 3 indices). $K_{\min}$: "
            r"median [10\%, 90\% quantile]. $J$: mean largest tail share of one scenario. Other columns: mean "
            r"$\Dp+\Dm$ of OPT and TB.}",
            r"\label{tab:e2}", r"\small", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
            r" & & & \multicolumn{2}{c}{$K=10$} & \multicolumn{2}{c}{$K=20$} & \multicolumn{2}{c}{$K=50$} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
            r"Half-life & $K_{\min}$ & $J$ & OPT & TB & OPT & TB & OPT & TB \\", r"\midrule"]
    rows = []
    for hl in HL_ORDER:
        s = sub[sub.half_life.astype(str) == hl]
        km = s.drop_duplicates(["dataset", "end"]).n_a
        q10, q90 = np.quantile(km, 0.1, method="nearest"), np.quantile(km, 0.9, method="nearest")
        cells = [f"{int(np.median(km))} [{int(q10)}, {int(q90)}]", f"{s.J.mean():.3f}"]
        for K in (10, 20, 50):
            r = s[s.K == K]
            cells += [f3(r.D_OPT.mean()), f3(r.D_TB.mean())]
        rows.append(f"{HL_NAME[hl]} & " + " & ".join(cells) + r" \\")
    (out / "tab_e2.tex").write_text(table(head, rows, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    for hl in ("250", "63"):
        for K in (10, 20, 50):
            s = sub[(sub.half_life.astype(str) == hl) & (sub.K == K)]
            facts[f"e2_ratio_TB_OPT_hl{hl}_K{K}"] = float(s.D_TB.mean() / s.D_OPT.mean())
            facts[f"e2_gap_OPTC_OPT_hl{hl}_K{K}"] = float((s["D_OPT-C"] - s.D_OPT).mean())
            facts[f"e2_gapmax_OPTC_OPT_hl{hl}_K{K}"] = float((s["D_OPT-C"] - s.D_OPT).max())
    s63 = sub[sub.half_life.astype(str) == "63"]
    km = s63.drop_duplicates(["dataset", "end"]).n_a
    facts["e2_kmin_hl63_a05"] = [int(np.median(km)), int(np.quantile(km, 0.1, method="nearest")),
                                 int(np.quantile(km, 0.9, method="nearest"))]
    facts["e2_windows_per_dataset"] = int(df.drop_duplicates(["dataset", "end"]).groupby("dataset").size().min())
    facts["e2_time_ms_median"] = {k: float(df[f"sec_{k}"].median() * 1e3) for k in ("OPT", "OPT-C", "TB")}
    facts["e2_time_ms_max"] = {k: float(df[f"sec_{k}"].max() * 1e3) for k in ("OPT", "OPT-C", "TB")}
    facts["e2_max_Dminus_OPTC"] = float(df["Dminus_OPT-C"].max())
    facts["e2_max_Dminus_TB"] = float(df["Dminus_TB"].max())
    facts["e2_lb_violations"] = int((df.D_OPT < df.lb - 1e-12).sum())
    facts["e2_TB_at_least_1_over_K"] = int((df.D_TB >= df.ub - 1e-12).sum())
    facts["e2_max_n_a"] = int(df.n_a.max())
    # supplement: all a, K and half-lives (one table per tail mass)
    blocks = []
    for a_ in (0.01, 0.05, 0.10):
        tag = f"{int(round(a_ * 100)):02d}"
        head = [r"\begin{table}[ht]", r"\centering", r"\small",
                rf"\caption{{Time-decayed weights, tail mass $a={a_:g}$ ($N=2000$; means over windows): $\Dp+\Dm$ of "
                r"OPT, OPT-C, and TB, the largest jump $J$, and the bounds $1/K-J\le\beta_K\le\mathrm{TB}<1/K$ of "
                r"Theorem~\ref{thm:rates}.}",
                rf"\label{{tab:app_e2_{tag}}}", r"\setlength{\tabcolsep}{4pt}",
                r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
                r"Half-life & $K$ & $J$ & $1/K-J$ & OPT & OPT-C & TB & $1/K$ \\", r"\midrule"]
        rows = []
        for hl in HL_ORDER:
            for K in (5, 10, 20, 50):
                s_ = df[(df.a == a_) & (df.half_life.astype(str) == hl) & (df.K == K)]
                rows.append(f"{HL_NAME[hl] if K == 5 else ''} & {K} & {f3(s_.J.mean())} & {f3(s_.lb.mean())} & "
                            f"{f3(s_.D_OPT.mean())} & {f3(s_['D_OPT-C'].mean())} & {f3(s_.D_TB.mean())} & "
                            f"{f3(1.0 / K)} \\\\")
            rows.append(r"\addlinespace")
        rows[-1] = r"\bottomrule"
        blocks.append(table(head, rows, [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    (out / "tab_app_e2.tex").write_text("\n".join(blocks))
    # supplement: K_min by index, N, a and half-life
    kt = pd.read_csv(ROOT / "results" / "e2_kmin_table.csv")
    head = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
            r"\caption{Minimum exact size $K_{\min}=n_a$ under time-decayed weights: median [10\%, 90\% "
            r"quantile] over windows ending every 63 trading days, by index, half-life, sample size $N$, and tail mass "
            r"$a$. For equal weights $K_{\min}=\lceil aN\rceil$ (up to merged ties).}",
            r"\label{tab:app_kmin}", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}llrlll@{}}", r"\toprule",
            r"Index & Half-life & $N$ & $a=0.01$ & $a=0.05$ & $a=0.10$ \\", r"\midrule"]
    rows = []
    for ds in ("SP500", "FTSE100", "DJIA"):
        for hl in ("63", "250", "1000"):
            for N_ in (500, 1000, 2000, 5000):
                cells = []
                for a_ in (0.01, 0.05, 0.10):
                    v = kt[(kt.dataset == ds) & (kt.half_life.astype(str) == hl) & (kt.N == N_) & (kt.a == a_)].n_a
                    cells.append(f"{int(np.median(v))} [{int(np.quantile(v, 0.1, method='nearest'))}, "
                                 f"{int(np.quantile(v, 0.9, method='nearest'))}]")
                name = {"SP500": "S\\&P 500", "FTSE100": "FTSE 100", "DJIA": "DJIA"}[ds] if (hl == "63" and N_ == 500) else ""
                hln = f"{hl} days" if N_ == 500 else ""
                rows.append(f"{name} & {hln} & {N_} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    rows[-1] = r"\bottomrule"
    (out / "tab_app_kmin.tex").write_text(table(head, rows, [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))


# ------------------------------------------------------------------------------------------------ E3
N_TEST = 250  # test days after each training window (as in scripts/run_e3_decisions.py)


def cluster_se(g, col, key="end"):
    c = g.groupby(key)[col].mean()
    return float(c.std(ddof=1) / np.sqrt(len(c)))


def crash_periods(ends, k=2):
    """Window ends whose test period contains one of the k largest daily S&P 500 losses
    (the test periods with the crashes of 2008 and 2020)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from tailreduce.data import load_losses
    losses = load_losses(ROOT / "data" / "processed", "SP500").to_numpy()
    worst = {int(e): float(losses[e:e + N_TEST].max()) for e in ends}
    return sorted(sorted(worst, key=lambda e: -worst[e])[:k])


def tab_e3(out, facts):
    df = pd.read_csv(ROOT / "results" / "e3_decisions.csv")
    df["regret_bp"] = 100 * df.regret
    df["d_oos_bp"] = 100 * (df.oos_cvar - df.oos_cvar_full)
    g = df.groupby(["weights", "family", "K", "method"]).regret_bp
    mean, mx, p95 = g.mean(), g.max(), g.quantile(0.95)
    head = [r"\begin{table*}[t]", r"\centering",
            r"\caption{Decisions from reduced scenario sets ($N=2000$ training days, $a=0.05$; 3 indices $\times$ 16 "
            r"windows $\times$ 20 random problems per family). In-sample regret in basis points of notional: mean "
            r"(maximum over the 960 problems). Family A: tail hedge, loss slope at most one; family B: capital buffer "
            r"with a breach cost. With equal weights, OPT-C and TB coincide. Median optimal CVaR (equal weights): "
            rf"{100 * df[(df.family == 'A') & (df.weights == 'equal')].f_star.median():.0f}~bp (A) and "
            rf"{100 * df[(df.family == 'B') & (df.weights == 'equal')].f_star.median():.0f}~bp (B).}}",
            r"\label{tab:e3}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
            r" & \multicolumn{4}{c}{Equal weights} & \multicolumn{4}{c}{Half-life 63 days} \\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
            r" & \multicolumn{2}{c}{A: tail hedge} & \multicolumn{2}{c}{B: capital buffer} & "
            r"\multicolumn{2}{c}{A: tail hedge} & \multicolumn{2}{c}{B: capital buffer} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
            r"Method & $K=10$ & $K=20$ & $K=10$ & $K=20$ & $K=10$ & $K=20$ & $K=10$ & $K=20$ \\", r"\midrule"]
    rows = []
    for m in ORDER:
        cells = []
        for w in ("equal", "h63"):
            for fam in ("A", "B"):
                for K in (10, 20):
                    cells.append(f"{bp(mean[(w, fam, K, m)])} ({bp_max(mx[(w, fam, K, m)])})")
        rows.append(f"{LAB[m]} & " + " & ".join(cells) + r" \\")
    (out / "tab_e3.tex").write_text(table(head, rows, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]))
    # facts
    facts["e3_rows"] = int(len(df))
    facts["e3_max_regret_minus_bound"] = float((df.regret - df.bound).max())
    facts["e3_regret_over_bound_K20"] = {f"{w}-{fam}-{m}": float((s.regret / s.bound.replace(0, np.nan)).mean())
                                         for (w, fam, m), s in df[df.K == 20].groupby(["weights", "family", "method"])}
    facts["e3_mean_bp"] = {f"{w}-{fam}-{K}-{m}": float(v) for (w, fam, K, m), v in mean.items()}
    facts["e3_max_bp"] = {f"{w}-{fam}-{K}-{m}": float(v) for (w, fam, K, m), v in mx.items()}
    facts["e3_fstar_median_bp"] = {f"{w}-{fam}": float(100 * s.f_star.median())
                                   for (w, fam), s in df.groupby(["weights", "family"])}
    A = df[df.family == "A"]
    full = A.drop_duplicates(["weights", "dataset", "end", "problem"])
    facts["e3_lp_ms_full_median"] = float(full.lp_seconds_full.median() * 1e3)
    facts["e3_lp_ms_reduced_median"] = {int(k): float(v * 1e3) for k, v in A.groupby("K").lp_seconds.median().items()}
    binds = (full.budget - full.premium_full).abs() <= 1e-6 * np.maximum(1.0, full.budget)
    facts["e3_budget_binds_full_share"] = float(binds.mean())
    k20 = df[df.K == 20]
    oos = {}
    for (w, fam, m), s in k20.groupby(["weights", "family", "method"]):
        oos[f"{w}-{fam}-{m}"] = [float(s.d_oos_bp.mean()), cluster_se(s, "d_oos_bp")]
    facts["e3_doos_K20_mean_clse"] = oos
    # out of sample: role of the two test periods with crashes (supplement, Section S4)
    crash = crash_periods(sorted(df.end.unique()))
    per = k20.groupby(["weights", "family", "method", "end"]).d_oos_bp.mean()
    combos = [(w, fam) for w in ("equal", "h63") for fam in ("A", "B")]
    cons = [float(per[(w, fam, m, e)]) for w, fam in combos for m in ("OPT-C", "TB") for e in crash]
    opt_w1 = [float(per[(w, fam, m, e)]) for w, fam in combos for m in ("OPT", "W1") for e in crash]
    ex = [float(per.loc[(w, fam, m)].drop(crash).mean()) for w, fam in combos for m in ("OPT", "OPT-C", "TB")]
    facts["e3_crash_period_ends"] = crash
    facts["e3_crash_doos_cons_range_bp"] = [min(cons), max(cons)]
    facts["e3_crash_doos_opt_w1_range_bp"] = [min(opt_w1), max(opt_w1)]
    facts["e3_excrash_doos_opt_cons_maxabs_bp"] = max(abs(x) for x in ex)
    ta = k20.drop_duplicates(["weights", "family", "dataset", "end", "method"])
    facts["e3_tail_atoms_K20_mean"] = {f"{w}-{m}": float(v) for (w, m), v in
                                       ta[ta.family == "A"].groupby(["weights", "method"]).tail_atoms.mean().items()}
    # supplement: all budgets, mean / p95 / max, both weightings
    blocks = []
    for w, wname in (("equal", "equal weights"), ("h63", "time-decayed weights (half-life 63 days)")):
        head = [r"\begin{table}[ht]", r"\centering", r"\small",
                rf"\caption{{Decision experiment, {wname}: in-sample regret in bp, mean / 95th percentile / maximum "
                r"over 960 problems per family and budget.}",
                rf"\label{{tab:app_e3_{w}}}", r"\setlength{\tabcolsep}{3pt}",
                r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
                r" & \multicolumn{3}{c}{A: tail hedge} & \multicolumn{3}{c}{B: capital buffer} \\",
                r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
                r"Method & $K=10$ & $K=20$ & $K=50$ & $K=10$ & $K=20$ & $K=50$ \\", r"\midrule"]
        rows = []
        for m in ORDER:
            cells = [f"{bp(mean[(w, fam, K, m)])} / {bp_max(p95[(w, fam, K, m)])} / {bp_max(mx[(w, fam, K, m)])}"
                     for fam in ("A", "B") for K in (10, 20, 50)]
            rows.append(f"{LAB[m]} & " + " & ".join(cells) + r" \\")
        blocks.append(table(head, rows, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    (out / "tab_app_e3.tex").write_text("\n".join(blocks))
    # supplement: out-of-sample results (K = 20), clustered standard errors, premium and Sharpe ratios
    head = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Decision experiment, out of sample ($K=20$, next 250 trading days). $\Delta$CVaR: CVaR of the "
            r"decision minus that of the full-sample decision, in bp, mean with standard error clustered by test "
            r"period (16 periods; the 3 indices and the 20 problems of a period share the test days). Family A: premium "
            r"spent (bp of notional per day) and change of the out-of-sample annualized Sharpe ratio relative to the "
            r"full-sample decision.}",
            r"\label{tab:app_e3_oos}", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}lllllrrrr@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{$\Delta$CVaR, equal} & \multicolumn{2}{c}{$\Delta$CVaR, $h=63$} & "
            r"\multicolumn{2}{c}{Premium A} & \multicolumn{2}{c}{$\Delta$Sharpe A} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
            r"Method & A & B & A & B & equal & $h=63$ & equal & $h=63$ \\", r"\midrule"]
    rows = []
    for m in ORDER:
        cells = []
        for w in ("equal", "h63"):
            for fam in ("A", "B"):
                mu, se = oos[f"{w}-{fam}-{m}"]
                cells.append(f"{sgn(mu)} ({se:.1f})")
        for w in ("equal", "h63"):
            s = k20[(k20.family == "A") & (k20.weights == w) & (k20.method == m)]
            cells.append(f"{100 * s.premium.mean():.2f}")
        for w in ("equal", "h63"):
            s = k20[(k20.family == "A") & (k20.weights == w) & (k20.method == m)]
            cells.append(sgn((s.oos_sharpe - s.oos_sharpe_full).mean(), 3))
        rows.append(f"{LAB[m]} & " + " & ".join(cells) + r" \\")
    pf = [f"{100 * full[full.weights == w].premium_full.mean():.2f}" for w in ("equal", "h63")]
    rows.append(r"\midrule")
    rows.append(f"Full sample & 0 & 0 & 0 & 0 & {pf[0]} & {pf[1]} & 0 & 0 \\\\")
    (out / "tab_app_e3_oos.tex").write_text(table(head, rows, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))
    facts["e3_premium_bp_K20"] = {f"{w}-{m}": float(100 * k20[(k20.family == "A") & (k20.weights == w) & (k20.method == m)].premium.mean())
                                  for w in ("equal", "h63") for m in ORDER}
    facts["e3_premium_bp_full"] = {w: float(100 * full[full.weights == w].premium_full.mean()) for w in ("equal", "h63")}


# ------------------------------------------------------------------------------------------------ checks
def sci(x):
    """Largest error in LaTeX scientific notation (0 stays 0)."""
    x = float(x)
    if x <= 0:
        return "0" if x == 0 else r"$<0$"
    e = int(np.floor(np.log10(x)))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


def tab_verify(out, ver):
    """Supplement table summarizing results/verification.json."""
    t1, t2, g, th, t3, p4 = (ver[k] for k in ("theorem1", "theorem2", "grid_lemma", "threshold", "theorem3",
                                               "proposition4"))
    rand_models = 200 * t2["instances"]
    rows = [
        (r"Thm.~\ref{thm:identity}(i)--(ii)", "bounds on random monotone problems", f"{t1['instances']:,}",
         sci(t1["max_violation_of_upper_bounds"])),
        (r"Thm.~\ref{thm:identity}(iii)", r"two-decision problem attains $M(\Dp+\Dm)$", f"{t1['instances']:,}",
         sci(t1["max_shortfall_of_tight_instance"])),
        (r"Lemma~\ref{lem:grid}", "moving atoms to scenarios", f"{g['instances']:,}", sci(g["max_increase"])),
        (r"Thm.~\ref{thm:minimax}(a)", r"$\beta_K$ vs.\ LP over all supports", f"{t2['instances']:,}", sci(t2["lp_gap"])),
        (r"Thm.~\ref{thm:minimax}(a)", r"models with $\tau\in[T,\beta-B]$ attain $\beta_K$", f"{t2['instances']:,}",
         sci(max(t2["constructed_gap"], t2.get("tau_family_violation", 0.0)))),
        (r"Thm.~\ref{thm:minimax}(a)", r"random models never below $\beta_K$", f"{rand_models:,}",
         sci(t2["random_below_beta"])),
        (r"Thm.~\ref{thm:minimax}(b)", r"$\beta^{\mathrm c}_K$ vs.\ LP; block model, $\Dm=0$", f"{t2['instances']:,}",
         sci(max(t2["lp_gap_conservative"], t2["constructed_gap_conservative"], t2["conservative_understates"]))),
        (r"Thm.~\ref{thm:minimax}(c)", r"exact size $n_a$; a missing tail scenario costs $\delta_j$",
         f"{th['instances']:,}",
         sci(max(th["max_discrepancy_of_exact_model"], th["max_shortfall_when_tail_scenario_missing"]))),
        (r"Thm.~\ref{thm:rates}(a)--(b)", r"lower bound; tail-block rule ($\Dm=0$, $\Dp<1/K$)",
         f"{t3['instances_per_part']:,}",
         sci(max(t3["lower_bound_violation"], t3["tb_understates"], t3["tb_above_1_over_K"],
                 t3["tb_minus_opt_minus_J"]))),
        (r"Thm.~\ref{thm:rates}(c)", r"equal weights: $\beta^{\mathrm c}_K=\beta_K$; closed form",
         f"{2 * t3['instances_per_part']:,}",
         sci(max(t3["equal_weights_gap_opt"], t3["equal_weights_gap_cons"], t3["equal_weights_gap_tb"],
                 t3["equal_weights_cons_minus_opt_any_a"]))),
        (r"Cor.~\ref{cor:tail}", "tail atoms bound random models", f"{t3['instances_per_part']:,}",
         sci(t3["corollary_violation"])),
        (r"Prop.~\ref{prop:price}(a)--(b)", r"closed forms (all $h$, odd and even $m$)", f"{p4['instances']:,}",
         sci(max(p4["kw_gap"], p4["qm_gap"], p4.get("block_position_gap", 0.0)))),
    ]
    head = [r"\begin{table}[ht]", r"\centering", r"\small",
            r"\caption{Numerical check of the results on random instances (\texttt{scripts/verify\_theory.py}, which "
            rf"calls \texttt{{tailreduce/verify.py}}; seed {ver['seed']}). Last column: largest violation over all "
            r"instances, zero in exact arithmetic (negative values print as $<0$). The LP check is limited by the "
            r"solver tolerance.}",
            r"\label{tab:app_verify}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{@{}llrr@{}}",
            r"\toprule", r"Result & Check & Cases & Largest error \\", r"\midrule"]
    body = [" & ".join(r) + r" \\" for r in rows]
    body += [r"\midrule",
             r"Prop.~\ref{prop:price}(c) & \multicolumn{3}{l}{simulated mean / upper bound $\le$ " +
             f"{p4['sampling_mean_ub_max_ratio']:.2f}; " + r"simulated $\mathbb{E}[\Dm]$ / lower bound $\ge$ " +
             f"{p4['sampling_mean_lb_min_ratio']:.2f}" + r"} \\"]
    (out / "tab_app_verify.tex").write_text(table(head, body, [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]))


# ------------------------------------------------------------------------------------------------ macros
def write_macros(out, facts):
    """\\newcommand macros for the numbers quoted in the text (Section 5 and the introduction)."""
    f = facts
    mb = f["e3_mean_bp"]
    lo_w, hi_w = f["e1_ratio_W_range_K10_200"]
    lo_t, hi_t = f["e1_ratio_TW2_range_K10_200"]
    km = f["e2_kmin_hl63_a05"]
    oos = f["e3_doos_K20_mean_clse"]
    mid = [m for m in ORDER if m not in ("QM", "RS")]
    oos_mid_max = max(abs(oos[f"{w}-{fam}-{m}"][0]) for w in ("equal", "h63") for fam in ("A", "B") for m in mid)
    oos_mid_se_max = max(oos[f"{w}-{fam}-{m}"][1] for w in ("equal", "h63") for fam in ("A", "B") for m in mid)
    oos_qmrs = [oos[f"{w}-{fam}-{m}"][0] for w in ("equal", "h63") for fam in ("A", "B") for m in ("QM", "RS")]
    ratio_tb = [f["e2_ratio_TB_OPT_hl63_K10"], f["e2_ratio_TB_OPT_hl63_K50"]]
    dec_ratio = [mb["h63-A-10-TB"] / mb["h63-A-10-OPT"], mb["h63-B-10-TB"] / mb["h63-B-10-OPT"],
                 mb["h63-A-20-TB"] / mb["h63-A-20-OPT"], mb["h63-B-20-TB"] / mb["h63-B-20-OPT"]]
    rb = [v for v in f["e3_regret_over_bound_K20"].values() if not np.isnan(v)]
    m = {
        "FactWratioLo": f"{lo_w:.0f}", "FactWratioHi": f"{hi_w:.0f}",
        "FactTWratioLo": f"{lo_t:.1f}", "FactTWratioHi": f"{hi_t:.1f}",
        "FactQMratio": f"{f['e1_ratio_QM_a05_K20']:.0f}",
        "FactWtworatioKtwenty": f"{f['e1_ratio_W2_OPT_a05_K20']:.0f}",
        "FactWtworatioKfifty": f"{f['e1_ratio_W2_OPT_a05_K50']:.0f}",
        "FactTailWone": f"{f['e1_tail_atoms_a05_K50']['W1']:.1f}",
        "FactTailWtwo": f"{f['e1_tail_atoms_a05_K50']['W2']:.1f}",
        "FactTimeOPT": f"{f['e1_time_ms_N5000_K20']['OPT']:.0f}",
        "FactTimeTB": f"{f['e1_time_ms_N5000_K20']['TB']:.1f}",
        "FactTimeWtwo": f"{f['e1_time_ms_N5000_K20']['W2']:.0f}",
        "FactTimeWone": f"{f['e1_time_ms_N5000_K20']['W1']:.0f}",
        "FactKminMed": f"{km[0]}", "FactKminLo": f"{km[1]}", "FactKminHi": f"{km[2]}",
        "FactTBratioLo": f"{ratio_tb[0]:.0f}", "FactTBratioHi": f"{ratio_tb[1]:.0f}",
        "FactTBratioKtwenty": f"{f['e2_ratio_TB_OPT_hl63_K20']:.1f}",
        "FactOPTCgapTen": f"{f['e2_gap_OPTC_OPT_hl63_K10']:.3f}",
        "FactOPTCgapMaxTen": f"{f['e2_gapmax_OPTC_OPT_hl63_K10']:.2f}",
        "FactDecRatioLo": f"{min(dec_ratio):.0f}", "FactDecRatioHi": f"{max(dec_ratio):.0f}",
        "FactRegretBoundMax": f"{100 * max(rb):.0f}",
        "FactOOSmid": f"{oos_mid_max:.0f}", "FactOOSse": f"{oos_mid_se_max:.0f}",
        "FactOOSqmrsLo": f"{min(oos_qmrs):.0f}", "FactOOSqmrsHi": f"{max(oos_qmrs):.0f}",
        "FactLPfull": f"{f['e3_lp_ms_full_median']:.0f}",
        "FactLPred": f"{f['e3_lp_ms_reduced_median'][20]:.0f}",
        "FactBinds": f"{100 * f['e3_budget_binds_full_share']:.0f}",
        # supplement (Section S4)
        "FactConsDminusMax": sci(max(f["e2_max_Dminus_OPTC"], f["e2_max_Dminus_TB"])),
        "FactCrashConsMin": f"${f['e3_crash_doos_cons_range_bp'][0]:.0f}$",
        "FactCrashOptMax": f"{f['e3_crash_doos_opt_w1_range_bp'][1]:.0f}",
        "FactExCrashMax": f"{f['e3_excrash_doos_opt_cons_maxabs_bp']:.1f}",
    }
    for key in m:
        assert key.isalpha(), key
    # the wording of the text relies on these facts; warn if a rerun no longer supports it
    checks = {
        "1/K - J <= beta_K in every window (Section S4)": f["e2_lb_violations"] == 0,
        "tail-block rule below 1/K in every window (Section S4)": f["e2_TB_at_least_1_over_K"] == 0,
        "OPT-C and TB never positive in the crash periods (Section S4)": f["e3_crash_doos_cons_range_bp"][1] <= 1e-9,
        "OPT and W1 never negative in the crash periods (Section S4)": f["e3_crash_doos_opt_w1_range_bp"][0] >= -1e-9,
        "regret never above the bound of Theorem 1 (Section 5.4)": f["e3_max_regret_minus_bound"] <= 1e-9,
        "QM and RS place on average fewer than two tail atoms at K = 20 (Section S4)":
            max(f["e3_tail_atoms_K20_mean"][f"{w}-{m}"] for w in ("equal", "h63") for m in ("QM", "RS")) < 2,
    }
    for text, ok in checks.items():
        if not ok:
            print(f"WARNING: the text states '{text}', but the results do not support it")
    lines = ["% Numbers quoted in the text; generated by code/scripts/make_tables.py from code/results/*.csv"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (out / "facts.tex").write_text("\n".join(lines) + "\n")
    facts["macros"] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT.parent / "paper" / "tables"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    facts = {}
    tab_e1(out, facts)
    tab_e2(out, facts)
    tab_e3(out, facts)
    ver = ROOT / "results" / "verification.json"
    if ver.exists():
        facts["verification"] = json.loads(ver.read_text())
        tab_verify(out, facts["verification"])
    write_macros(out, facts)
    (ROOT / "results" / "facts.json").write_text(json.dumps(facts, indent=2, default=float))
    print(json.dumps(facts["macros"], indent=1))


if __name__ == "__main__":
    main()
