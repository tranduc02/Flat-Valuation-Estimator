"""Feature building.

Three kinds of features:
- static: what the flats file (and optional API extras) says about the flat and its building;
- history: index-adjusted prices of earlier sales of the same flat, the same building,
  the same building and bedroom count, the same mesh block, and the same suburb and bedroom
  count. They only use sales strictly before the query date, so a training row (a past sale)
  sees exactly what a prediction on 31 August would see: the past, never its own price.
- nearby buildings: index-adjusted prices in the closest other buildings that had sales
  before the query's month, which helps flats whose own building has little or no history.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config

TOWN_HALL = (-33.8731, 151.2065)
# Rough points along the harbour foreshore, for a "distance to water" (view) proxy.
HARBOUR = [
    (-33.8610, 151.2110), (-33.8570, 151.2150), (-33.8550, 151.2060), (-33.8560, 151.2010),
    (-33.8640, 151.1940), (-33.8725, 151.2000), (-33.8740, 151.1860), (-33.8690, 151.2220),
    (-33.8720, 151.2300), (-33.8750, 151.2310),
]

STATIC_NUMERIC = [
    "latitude", "longitude", "bedrooms", "bathrooms", "car_spaces", "flats_in_building",
    "unit_num", "floor", "unit_rank", "dist_cbd_km", "dist_harbour_km",
    "is_shop", "is_office", "is_carspace", "is_storage", "is_penthouse",
]
STATIC_CATEGORICAL = ["suburb", "block_type"]

# (feature prefix, grouping columns) for history features.
HISTORY_GROUPS = [
    ("own", ["gnaf_id"]),
    ("bld", ["building"]),
    ("bldbed", ["building", "bed_key"]),
    ("mb", ["mb"]),
    ("subbed", ["suburb", "bed_key"]),
]
NEARBY_BUILDINGS = 8

# Cumulative steps for `run.py ablation`: each adds the history features with these prefixes.
ABLATION_STEPS = [
    ("Flat details only", []),
    ("+ the flat's own earlier sales", ["own_"]),
    ("+ earlier sales in the same building", ["bld_", "bldbed_"]),
    ("+ mesh block and suburb sales", ["mb_", "subbed_"]),
    ("+ nearby buildings", ["nbr_"]),
    ("+ mesh block homes and people (supplied)", ["blk_"]),
    ("+ Microburbs API property and dwelling type", ["api_"]),
    ("+ OpenStreetMap buildings and stations", ["osm_"]),
    ("+ Valuer General lot area", ["vg_"]),
]
GROUP_PREFIXES = tuple(p for _, prefixes in ABLATION_STEPS for p in prefixes)
EXTRA_PREFIXES = ("api_", "blk_", "osm_", "vg_")  # static inputs from outside the flats file


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def static_features(flats: pd.DataFrame, api_extras: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per gnaf_id with the static features and the grouping keys."""
    f = flats.copy()
    f["dist_cbd_km"] = haversine_km(f["latitude"], f["longitude"], *TOWN_HALL)
    f["dist_harbour_km"] = np.min(
        [haversine_km(f["latitude"], f["longitude"], la, lo) for la, lo in HARBOUR], axis=0)
    f["bed_key"] = f["bedrooms"].fillna(-1).clip(upper=5).astype(int)
    for col in STATIC_NUMERIC:
        if col not in f:
            f[col] = np.nan
        f[col] = f[col].astype(float)
    for col in STATIC_CATEGORICAL:
        f[col] = f[col].fillna("unknown").astype(str) if col in f else "unknown"
    f["mb"] = f["mb"].fillna("unknown").astype(str) if "mb" in f else "unknown"

    keep = ["gnaf_id", "building", "bed_key", "mb", "is_nonres"] + STATIC_NUMERIC + STATIC_CATEGORICAL
    out = f[keep]
    if api_extras is not None:
        out = out.merge(api_extras, on="gnaf_id", how="left")
        if "api_floor" in out:
            out["floor"] = out["floor"].fillna(out["api_floor"])
    return out.set_index("gnaf_id")


def static_feature_names(static: pd.DataFrame) -> list[str]:
    extras = [c for c in static.columns if c.startswith(EXTRA_PREFIXES) and c != "api_floor"]
    return STATIC_NUMERIC + STATIC_CATEGORICAL + extras


def history_features(query: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """For each query row (keys + query_date): count, mean and latest index-adjusted log
    price (lp_norm) of earlier sales in each group, plus the age of that latest sale."""
    query = query.reset_index(drop=True).copy()
    query["_row"] = np.arange(len(query))
    q_sorted = query.sort_values("query_date")
    h_all = hist.sort_values("sale_date")
    out = pd.DataFrame(index=query.index)

    for prefix, keys in HISTORY_GROUPS:
        h = h_all[keys + ["sale_date", "lp_norm"]].copy()
        g = h.groupby(keys, sort=False)["lp_norm"]
        h[f"{prefix}_n"] = g.cumcount() + 1
        h[f"{prefix}_mean"] = g.cumsum() / h[f"{prefix}_n"]
        h = h.rename(columns={"lp_norm": f"{prefix}_last", "sale_date": f"{prefix}_last_date"})
        m = pd.merge_asof(
            q_sorted[keys + ["query_date", "_row"]], h,
            left_on="query_date", right_on=f"{prefix}_last_date", by=keys,
            direction="backward", allow_exact_matches=False,
        ).set_index("_row").sort_index()
        out[f"{prefix}_n"] = m[f"{prefix}_n"].fillna(0).to_numpy()
        out[f"{prefix}_mean"] = m[f"{prefix}_mean"].to_numpy()
        out[f"{prefix}_last"] = m[f"{prefix}_last"].to_numpy()
        out[f"{prefix}_last_age_yrs"] = ((m["query_date"] - m[f"{prefix}_last_date"]).dt.days / 365.25).to_numpy()
    return out


def _km_xy(lat, lon) -> np.ndarray:
    """Flat-earth km coordinates; accurate enough across one council area."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return np.column_stack([lon * 111.32 * np.cos(np.radians(-33.88)), lat * 110.57])


def neighbour_features(q: pd.DataFrame, hist: pd.DataFrame, static: pd.DataFrame,
                       k: int = NEARBY_BUILDINGS) -> pd.DataFrame:
    """For each query row: sales-weighted mean lp_norm of the k nearest *other* buildings with
    sales before the first day of the query's month, their mean distance and their sale count.
    Month-level cutoffs keep this time-honest while building one search tree per month."""
    out = np.full((len(q), 3), np.nan)
    bld_xy = static.groupby("building")[["latitude", "longitude"]].mean().dropna()
    bld_xy = pd.DataFrame(_km_xy(bld_xy["latitude"], bld_xy["longitude"]), index=bld_xy.index, columns=["x", "y"])
    q_xy = _km_xy(q["latitude"], q["longitude"])
    q_ok = ~np.isnan(q_xy).any(axis=1)
    q_bld = q["building"].to_numpy()
    h = hist[["sale_date", "building", "lp_norm"]].sort_values("sale_date")
    h_dates = h["sale_date"].to_numpy()

    months = q["query_date"].dt.to_period("M")
    for month, rows in pd.Series(np.arange(len(q))).groupby(months.to_numpy()).groups.items():
        rows = np.asarray(rows)
        rows = rows[q_ok[rows]]
        cutoff = np.datetime64(month.start_time, "ns")
        prior = h.iloc[: np.searchsorted(h_dates, cutoff, side="left")]
        agg = prior.groupby("building")["lp_norm"].agg(["sum", "count"]).join(bld_xy, how="inner")
        if len(agg) < 2 or len(rows) == 0:
            continue
        kk = min(k + 1, len(agg))
        dist, idx = cKDTree(agg[["x", "y"]].to_numpy()).query(q_xy[rows], k=kk)
        dist, idx = dist.reshape(len(rows), kk), idx.reshape(len(rows), kk)
        use = agg.index.to_numpy()[idx] != q_bld[rows, None]   # drop the flat's own building
        use &= np.cumsum(use, axis=1) <= k
        sums, counts = agg["sum"].to_numpy()[idx] * use, agg["count"].to_numpy()[idx] * use
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows, 0] = sums.sum(1) / counts.sum(1)
            out[rows, 1] = (dist * use).sum(1) / use.sum(1)
        out[rows, 2] = counts.sum(1)
    return pd.DataFrame(out, columns=["nbr_mean", "nbr_dist_km", "nbr_n"])


def feature_frame(query: pd.DataFrame, hist: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """query: gnaf_id + query_date. hist: past sales with gnaf_id, sale_date, lp_norm.
    Returns the model's feature matrix, row-aligned with query."""
    names = static_feature_names(static)
    q = query[["gnaf_id", "query_date"]].reset_index(drop=True)
    q["query_date"] = q["query_date"].astype("datetime64[ns]")  # merge_asof needs matching units
    q = q.join(static, on="gnaf_id")
    hist = hist[["gnaf_id", "sale_date", "lp_norm"]].copy()
    hist["sale_date"] = hist["sale_date"].astype("datetime64[ns]")
    hist = hist.join(static[["building", "bed_key", "mb", "suburb"]], on="gnaf_id", how="inner")
    x = pd.concat([q[names], history_features(q, hist), neighbour_features(q, hist, static)], axis=1)
    x["t_years"] = q["query_date"].dt.year + q["query_date"].dt.dayofyear / 366
    for col in STATIC_CATEGORICAL + [c for c in names if c not in STATIC_CATEGORICAL
                                     and not pd.api.types.is_numeric_dtype(x[c])]:
        x[col] = x[col].fillna("unknown").astype(str).astype("category")
    if config.DROP_FEATURE_PREFIXES:
        x = x[[c for c in x.columns if not c.startswith(tuple(config.DROP_FEATURE_PREFIXES))]]
    return x


def recency_weights(dates: pd.Series, as_of) -> np.ndarray:
    age_years = (pd.Timestamp(as_of) - dates).dt.days.to_numpy() / 365.25
    return 0.5 ** (age_years / config.RECENCY_HALF_LIFE_YEARS)
