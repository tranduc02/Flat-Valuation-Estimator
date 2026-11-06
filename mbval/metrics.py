"""The judges' metric (share within 10%, exactly 10% counts) plus the tie-break and R squared."""
import numpy as np

import config

BANDS = [(0, 0.05, "within 5%"), (0.05, 0.10, "5% to 10%"), (0.10, 0.20, "10% to 20%"), (0.20, np.inf, "over 20%")]


def summarize(actual, pred) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    missing = np.isnan(pred)
    rel = np.abs(pred - actual) / actual
    within = (rel <= config.WITHIN_PCT + 1e-9) & ~missing  # a blank counts as wrong
    ok = ~missing
    abs_err = np.abs(pred[ok] - actual[ok])
    ss_res = np.sum((actual[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((actual[ok] - actual[ok].mean()) ** 2)

    eps = 1e-9
    rel_ok = rel[ok]
    bands = {}
    for lo, hi, label in BANDS:
        above_lo = rel_ok > lo + eps if lo > 0 else np.ones(len(rel_ok), dtype=bool)
        bands[label] = int((above_lo & (rel_ok <= hi + eps)).sum())
    return {
        "n": int(len(actual)),
        "n_missing": int(missing.sum()),
        "within_10_n": int(within.sum()),
        "within_10_share": round(float(within.mean()), 4) if len(actual) else None,
        "median_abs_error": round(float(np.median(abs_err)), 0) if ok.any() else None,
        "median_abs_pct_error": round(float(np.median(rel[ok])), 4) if ok.any() else None,
        "r2": round(float(1 - ss_res / ss_tot), 4) if ss_tot > 0 else None,
        "bands": bands,
    }


def fmt(summary: dict) -> str:
    return (f"n={summary['n']:,}  within 10%: {summary['within_10_n']:,} ({summary['within_10_share']:.1%})  "
            f"median abs error ${summary['median_abs_error']:,.0f}  R2={summary['r2']}")
