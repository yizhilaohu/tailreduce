"""Download and preprocess the public index data used in the experiments.

Sources (pinned to fixed commits for reproducibility):
  * S&P 500 (^GSPC), FTSE 100 (^FTSE), Dow Jones Industrial Average (^DJI):
    daily OHLC data 2000-01-03 to 2023-12-29 shipped with the CRAN package 'quarks' (GPL-3),
    read from the read-only CRAN mirror on GitHub (github.com/cran/quarks).
Scenarios are daily losses in percent: loss_t = -100 * log(P_t / P_{t-1}).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

QUARKS = "https://raw.githubusercontent.com/cran/quarks/15b9b176cd6e9b4d328fba0567b52bddc8447af2/data/{}.rda"
SOURCES = {
    "SP500": {"url": QUARKS.format("SP500"), "file": "quarks_SP500.rda", "label": "S&P 500"},
    "FTSE100": {"url": QUARKS.format("FTSE100"), "file": "quarks_FTSE100.rda", "label": "FTSE 100"},
    "DJIA": {"url": QUARKS.format("DJI"), "file": "quarks_DJI.rda", "label": "DJIA"},
}
# sha256 of the pinned files (checked after download)
SHA256 = {
    "quarks_SP500.rda": "a64af52e3528221472f9df4f566b1fd75b03ab58df3e47eb2fa98d343f40f143",
    "quarks_FTSE100.rda": "6d5c5710f0caa14e84440963e957066c10ea09e54ccad40c351e28adf8684bc3",
    "quarks_DJI.rda": "e6a8e7c9bb8f00e0514edd148f7fac511049aae29a52e4fb00847b3e2886c027",
}
MAIN_DATASETS = ("SP500", "FTSE100", "DJIA")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path, names=MAIN_DATASETS, force=False):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = SOURCES[name]
        dest = raw_dir / src["file"]
        if force or not dest.exists():
            urllib.request.urlretrieve(src["url"], dest)
        digest = sha256(dest)
        expected = SHA256.get(src["file"])
        if expected is not None and digest != expected:
            raise RuntimeError(f"checksum mismatch for {dest}: {digest}")
        print(f"{name:8s} {dest.name:22s} sha256={digest}")


def load_prices(raw_dir: Path, name: str) -> pd.Series:
    """Daily closing prices from the R data file of the 'quarks' package (column price_close)."""
    import warnings

    import rdata  # pure-python reader for R data files

    path = raw_dir / SOURCES[name]["file"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(str(path))
    df = pd.DataFrame(next(iter(obj.values())))
    dcol = "ref_date" if "ref_date" in df.columns else [c for c in df.columns if "date" in c.lower()][0]
    dates = df[dcol]
    if not np.issubdtype(np.asarray(dates).dtype, np.datetime64):
        dates = pd.to_datetime(np.asarray(dates, dtype=float), unit="D", origin="unix")
    s = pd.Series(np.asarray(df["price_close"], dtype=float), index=pd.DatetimeIndex(dates), name=name)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s[np.isfinite(s) & (s > 0)]
    return s


def daily_losses(prices: pd.Series) -> pd.Series:
    """Daily losses in percent, -100 * log(P_t / P_{t-1}); zero-return days are kept."""
    lr = np.log(prices).diff().dropna()
    return (-100.0 * lr).rename(prices.name)


def build(raw_dir: Path, out_dir: Path, names=MAIN_DATASETS):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        p = load_prices(raw_dir, name)
        L = daily_losses(p)
        L.to_frame("loss_pct").to_csv(out_dir / f"{name}_losses.csv", index_label="date")
        rows.append({"dataset": name, "label": SOURCES[name]["label"], "first": L.index.min().date(),
                     "last": L.index.max().date(), "T": len(L), "mean": L.mean(), "std": L.std(),
                     "min": L.min(), "max": L.max(), "q95": L.quantile(0.95), "q99": L.quantile(0.99)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def load_losses(out_dir: Path, name: str) -> pd.Series:
    df = pd.read_csv(out_dir / f"{name}_losses.csv", parse_dates=["date"], index_col="date")
    return df["loss_pct"].rename(name)
