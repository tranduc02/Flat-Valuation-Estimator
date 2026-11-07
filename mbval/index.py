"""City-wide flat price index from repeat sales (Bailey-Muth-Nourse).

Each flat that sold twice gives log(p2/p1) = idx[t2] - idx[t1] + noise. We solve
for idx by least squares with a penalty on period-to-period jumps, so thin
periods borrow strength from their neighbours. The index lets us move any past
sale to today's prices: lp_norm = log(price) - idx[sale period].
"""
import numpy as np
import pandas as pd

import config

MIN_HOLD_DAYS = 180      # quick flips are often renovations or non-market
MAX_ABS_LOG_CHANGE = np.log(3)
MIN_PAIRS = 100


def repeat_pairs(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.sort_values(["gnaf_id", "sale_date"])
    prev = s.groupby("gnaf_id")[["sale_date", "price"]].shift(1)
    pairs = pd.DataFrame({
        "t1": prev["sale_date"], "t2": s["sale_date"],
        "dlog": np.log(s["price"]) - np.log(prev["price"]),
    }).dropna()
    held = (pairs["t2"] - pairs["t1"]).dt.days
    return pairs[(held >= MIN_HOLD_DAYS) & (pairs["dlog"].abs() <= MAX_ABS_LOG_CHANGE)]


def build_index(sales: pd.DataFrame) -> pd.Series:
    """Log price index by period (pd.Period index), first period = 0."""
    freq = config.INDEX_FREQ
    periods = pd.period_range(sales["sale_date"].min(), sales["sale_date"].max(), freq=freq)
    pairs = repeat_pairs(sales)
    if len(pairs) < MIN_PAIRS:
        print(f"index: only {len(pairs)} repeat-sale pairs; falling back to median log price per period")
        return _median_index(sales, periods)

    pos = {p: i for i, p in enumerate(periods)}
    p1 = pairs["t1"].dt.to_period(freq).map(pos).to_numpy()
    p2 = pairs["t2"].dt.to_period(freq).map(pos).to_numpy()
    keep = p1 != p2
    p1, p2, y = p1[keep], p2[keep], pairs["dlog"].to_numpy()[keep]

    n_periods = len(periods)
    x = np.zeros((len(y), n_periods))
    x[np.arange(len(y)), p2] += 1
    x[np.arange(len(y)), p1] -= 1
    # Smoothness rows: sqrt(lambda) * (idx[k] - idx[k-1]) = 0
    lam = np.sqrt(config.INDEX_SMOOTHING)
    smooth = np.zeros((n_periods - 1, n_periods))
    smooth[np.arange(n_periods - 1), np.arange(1, n_periods)] = lam
    smooth[np.arange(n_periods - 1), np.arange(n_periods - 1)] = -lam
    a = np.vstack([x, smooth])[:, 1:]  # drop the base period (fixed at 0)
    b = np.concatenate([y, np.zeros(n_periods - 1)])
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    print(f"index: {len(y):,} repeat-sale pairs over {n_periods} periods ({freq})")
    return pd.Series(np.concatenate([[0.0], coef]), index=periods, name="log_index")


def _median_index(sales: pd.DataFrame, periods: pd.PeriodIndex) -> pd.Series:
    lp = np.log(sales["price"]).groupby(sales["sale_date"].dt.to_period(periods.freqstr)).median()
    idx = lp.reindex(periods).interpolate(limit_direction="both").rolling(3, min_periods=1).mean()
    return (idx - idx.iloc[0]).rename("log_index")


def index_at(index: pd.Series, dates) -> np.ndarray:
    """Index value for each date; dates outside the index range take the nearest end."""
    dates = pd.to_datetime(pd.Series(dates))
    periods = dates.dt.to_period(index.index.freqstr)
    first, last = index.index[0], index.index[-1]
    periods = periods.where(periods >= first, first).where(periods <= last, last)
    return index.reindex(periods).to_numpy()
