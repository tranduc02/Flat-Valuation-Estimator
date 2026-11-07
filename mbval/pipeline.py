"""Backtest, ablation and final prediction.

All of them run the same steps for an as_of date: use sales dated before as_of only,
build the price index, train, and value flats as at as_of. The backtest uses
as_of = SPLIT_DATE and checks against later sales; the final run uses the day
after VALUATION_DATE.
"""
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from mbval import data, external, features, index, metrics
from mbval.model import Model


@dataclass
class Prepared:
    as_of: pd.Timestamp
    index: pd.Series
    hist: pd.DataFrame   # all usable sales before as_of, with lp_norm
    train: pd.DataFrame  # the subset used as training rows
    x: pd.DataFrame
    y: np.ndarray
    w: np.ndarray


@dataclass
class Fitted:
    as_of: pd.Timestamp
    model: Model
    index: pd.Series
    hist: pd.DataFrame
    n_train: int


def load_inputs(data_dir: Path):
    flats = data.load_flats(data_dir)
    sales = data.load_sales(data_dir).assign(source="supplied", available_date=pd.NaT)
    vg = external.load_vg_sales(sales)
    if vg is not None:
        vg = vg[vg["gnaf_id"].isin(flats["gnaf_id"])]
    if vg is not None and len(vg):
        sales = (pd.concat([sales, vg.assign(source="vg")], ignore_index=True)
                 .sort_values("sale_date").reset_index(drop=True))
        print(f"added {len(vg):,} Valuer General sales that the supplied file doesn't have")

    extras = []
    api_file = config.CACHE_DIR / "api_properties.csv"
    if api_file.exists():
        extras.append(pd.read_csv(api_file, dtype={"gnaf_id": str}))
    blocks = block_features(flats, data_dir)
    if blocks is not None:
        extras.append(blocks)
    osm = external.osm_features(flats)
    if osm is not None:
        extras.append(osm)
        print(f"OpenStreetMap inputs: building height known for {osm['osm_own_height_m'].notna().mean():.0%} of flats, "
              f"flat height for {osm['osm_taller_100m'].notna().mean():.0%}")
    area = external.vg_area(flats)
    if area is not None:
        extras.append(area)
        print(f"Valuer General lot area for {len(area):,} flats")
    merged = None
    for e in extras:
        merged = e if merged is None else merged.merge(e, on="gnaf_id", how="outer")
    static = features.static_features(flats, merged)
    return flats, sales, static


def block_features(flats: pd.DataFrame, data_dir: Path) -> pd.DataFrame | None:
    """Supplied homes/people per mesh block. The 2026 figures are Microburbs estimates, not Census
    results, and 'new homes since 2021' is a slight look-ahead for sales before 2026."""
    path = Path(data_dir) / config.BLOCKS_FILE
    if not path.exists():
        return None
    b = data.read_csv(path)
    b["mb"] = b["mb"].astype(str)
    for col in ("homes_2021", "homes_2026", "people_2021", "people_2026", "new_homes"):
        b[col] = pd.to_numeric(b[col], errors="coerce")
    b = b.drop_duplicates("mb").set_index("mb")
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = pd.DataFrame({
            "blk_homes_2026": b["homes_2026"],
            "blk_new_homes_share": (b["new_homes"] / b["homes_2026"]).where(b["homes_2026"] > 0),
            "blk_people_per_home": (b["people_2026"] / b["homes_2026"]).where(b["homes_2026"] > 0),
        })
    out = flats[["gnaf_id", "mb"]].join(feats, on="mb").drop(columns="mb")
    print(f"mesh block homes/people for {out['blk_homes_2026'].notna().mean():.0%} of flats")
    return out


def prepare(sales: pd.DataFrame, static: pd.DataFrame, as_of) -> Prepared:
    as_of = pd.Timestamp(as_of)
    known = sales["available_date"].isna() | (sales["available_date"] < as_of)  # VG rows: registered by as_of
    hist = sales[(sales["sale_date"] < as_of) & known
                 & sales["price"].between(config.MIN_PRICE, config.MAX_PRICE)].copy()
    idx = index.build_index(hist)
    hist["lp_norm"] = np.log(hist["price"]) - index.index_at(idx, hist["sale_date"])

    train = hist[(hist["sale_date"] >= pd.Timestamp(config.TRAIN_SINCE))
                 & hist["gnaf_id"].isin(static.index)].reset_index(drop=True)
    x = features.feature_frame(train.assign(query_date=train["sale_date"]), hist, static)
    w = features.recency_weights(train["sale_date"], as_of)
    return Prepared(as_of, idx, hist, train, x, train["lp_norm"].to_numpy(), w)


def fit(sales: pd.DataFrame, static: pd.DataFrame, as_of) -> Fitted:
    p = prepare(sales, static, as_of)
    model = Model().fit(p.x, p.y, p.w, p.train["sale_date"])
    print(f"trained on {len(p.train):,} sales ({p.train['sale_date'].min().date()} to "
          f"{p.train['sale_date'].max().date()}), {len(p.x.columns)} features, {model.rounds} rounds")
    return Fitted(p.as_of, model, p.index, p.hist, len(p.train))


def value_features(hist: pd.DataFrame, idx: pd.Series, as_of, static: pd.DataFrame, gnaf_ids):
    """Features for valuing gnaf_ids at as_of, and the index level to add back to the model output."""
    q = pd.DataFrame({"gnaf_id": list(gnaf_ids), "query_date": pd.Timestamp(as_of)})
    level = index.index_at(idx, [pd.Timestamp(as_of) - timedelta(days=1)])[0]
    return features.feature_frame(q, hist, static), level


def predict(fitted: Fitted, static: pd.DataFrame, gnaf_ids, adjust: bool = True) -> np.ndarray:
    """Estimated price at fitted.as_of for each gnaf_id (times PRICE_ADJUST unless adjust=False)."""
    x, level = value_features(fitted.hist, fitted.index, fitted.as_of, static, gnaf_ids)
    return np.exp(fitted.model.predict(x) + level) * (config.PRICE_ADJUST if adjust else 1.0)


def test_sales(sales: pd.DataFrame, static: pd.DataFrame, cap: bool = True, start=None, end=None) -> pd.DataFrame:
    """Sales from SPLIT_DATE to VALUATION_DATE (or start..end), filtered like the judges' set: flats
    under $1.2m. cap=False keeps every price, like the full-set result the judges also publish."""
    split = pd.Timestamp(start or config.SPLIT_DATE)
    end = pd.Timestamp(end or config.VALUATION_DATE)
    top = config.SCORED_PRICE_CAP - 1 if cap else config.MAX_PRICE
    sales = sales[sales["source"] == "supplied"]  # score on the supplied file only, like the judges
    test = sales[(sales["sale_date"] >= split) & (sales["sale_date"] <= end)
                 & sales["price"].between(config.MIN_PRICE, top)
                 & sales["gnaf_id"].isin(static.index)].copy()
    test = test[~static.loc[test["gnaf_id"], "is_nonres"].to_numpy()].reset_index(drop=True)
    if test.empty:
        raise SystemExit("No test sales between SPLIT_DATE and VALUATION_DATE. Move SPLIT_DATE earlier.")
    return test


def baselines(fitted: Fitted, static: pd.DataFrame, gnaf_ids) -> dict[str, np.ndarray]:
    """The task page's two simple benchmarks, rebuilt on our own split for comparison."""
    hist = fitted.hist.join(static[["suburb", "bed_key"]], on="gnaf_id", how="inner")
    recent = hist[hist["sale_date"] >= fitted.as_of - pd.DateOffset(months=12)]
    q = static.loc[list(gnaf_ids), ["suburb", "bed_key"]].reset_index()

    by_sub_bed = recent.groupby(["suburb", "bed_key"])["price"].agg(["median", "size"])
    by_sub_bed = by_sub_bed[by_sub_bed["size"] >= 5]["median"].rename("m1")
    by_sub = recent.groupby("suburb")["price"].median().rename("m2")
    median = (q.join(by_sub_bed, on=["suburb", "bed_key"]).join(by_sub, on="suburb")
              .pipe(lambda d: d["m1"].fillna(d["m2"]).fillna(recent["price"].median())).to_numpy())

    last = fitted.hist.sort_values("sale_date").groupby("gnaf_id").last()
    level = index.index_at(fitted.index, [fitted.as_of - timedelta(days=1)])[0]
    last_indexed = np.exp(last["lp_norm"] + level).reindex(q["gnaf_id"]).to_numpy()
    return {
        "suburb_bedroom_median": median,
        "last_sale_indexed": np.where(np.isnan(last_indexed), median, last_indexed),
    }


def floor_premium(hist: pd.DataFrame, static: pd.DataFrame) -> dict | None:
    """How much higher a flat sells per floor, compared only with flats in the same building.

    Within each building we subtract the building's average (index-adjusted log price and
    floor) and fit one slope, so location, building quality and sale date cancel out.
    """
    h = hist[hist["sale_date"] >= pd.Timestamp(config.TRAIN_SINCE)].join(
        static[["building", "floor", "bed_key", "is_nonres"]], on="gnaf_id", how="inner")
    h = h[~h["is_nonres"] & h["floor"].between(0, 90)]
    h = h[h.groupby("building")["floor"].transform("nunique") >= 2]
    if len(h) < 30:
        return None

    def slope(df, keys):
        y = df["lp_norm"] - df.groupby(keys)["lp_norm"].transform("mean")
        x = df["floor"] - df.groupby(keys)["floor"].transform("mean")
        return float((x * y).sum() / (x * x).sum()), x, y

    b, rel_floor, rel_price = slope(h, "building")
    result = {"per_floor_pct": round(float(np.expm1(b)), 4), "n_sales": len(h),
              "n_buildings": int(h["building"].nunique())}

    hb = h[h["bed_key"] >= 0]
    hb = hb[hb.groupby(["building", "bed_key"])["floor"].transform("nunique") >= 2]
    if len(hb) >= 30:
        result["per_floor_pct_same_bedrooms"] = round(float(np.expm1(slope(hb, ["building", "bed_key"])[0])), 4)
        result["n_sales_same_bedrooms"] = len(hb)

    # For the chart: average % above the building's average, by floors above/below its average floor.
    curve = rel_price.groupby(rel_floor.round().astype(int)).agg(["mean", "size"])
    curve = curve[(curve["size"] >= 20) & (curve.index.to_series().abs() <= 20)]
    result["curve"] = {int(f): [round(float(np.expm1(m)), 4), int(n)] for f, (m, n) in curve.iterrows()}
    return result


def backtest(data_dir: Path) -> dict:
    flats, sales, static = load_inputs(data_dir)
    split = pd.Timestamp(config.SPLIT_DATE)
    fitted = fit(sales, static, split)
    test = test_sales(sales, static)

    test["pred"] = predict(fitted, static, test["gnaf_id"])
    for name, values in baselines(fitted, static, test["gnaf_id"]).items():
        test[name] = values
    had_sale = test["gnaf_id"].isin(fitted.hist["gnaf_id"])
    bld_sales = test["gnaf_id"].map(static["building"]).isin(
        fitted.hist["gnaf_id"].map(static["building"]).dropna())
    bed_known = static.loc[test["gnaf_id"], "bed_key"].to_numpy() >= 0

    def seg(mask):
        return metrics.summarize(test.loc[mask, "price"], test.loc[mask, "pred"])

    full = test_sales(sales, static, cap=False)
    full["pred"] = predict(fitted, static, full["gnaf_id"])

    result = {
        "split_date": str(split.date()),
        "test_end": str(config.VALUATION_DATE),
        "price_adjust": config.PRICE_ADJUST,
        "n_train": fitted.n_train,
        "n_test": len(test),
        "model": metrics.summarize(test["price"], test["pred"]),
        "model_all_prices": metrics.summarize(full["price"], full["pred"]),
        "baselines": {name: metrics.summarize(test["price"], test[name])
                      for name in ("suburb_bedroom_median", "last_sale_indexed")},
        "segments": {
            "sold_before": seg(had_sale),
            "never_sold_before": seg(~had_sale),
            "building_had_sales": seg(bld_sales),
            "building_no_sales": seg(~bld_sales),
            "bedrooms_known": seg(bed_known),
            "bedrooms_unknown": seg(~bed_known),
        },
        "floor_premium": floor_premium(fitted.hist, static),
        "external": {
            "vg_sales_added": int((sales["source"] == "vg").sum()),
            "osm_height_share": (round(float(static["osm_own_height_m"].notna().mean()), 4)
                                 if "osm_own_height_m" in static else None),
            "vg_area_n": int(static["vg_lot_area_m2"].notna().sum()) if "vg_lot_area_m2" in static else 0,
        },
        "importance": fitted.model.importance(),
        "index": {str(p): round(float(v), 4) for p, v in fitted.index.items()},
    }
    config.OUT_DIR.mkdir(exist_ok=True)
    test.to_csv(config.OUT_DIR / "backtest_predictions.csv", index=False)
    (config.OUT_DIR / "backtest_metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\nbacktest: train before {split.date()}, test {split.date()} to {config.VALUATION_DATE} "
          f"(under ${config.SCORED_PRICE_CAP:,}, flats only)")
    print(f"  model                  {metrics.fmt(result['model'])}")
    for name, s in result["baselines"].items():
        print(f"  {name:<22} {metrics.fmt(s)}")
    print(f"  model, all prices      {metrics.fmt(result['model_all_prices'])}")
    print(f"  task page, judges' {config.PRIVATE_TEST_N} private sales: " + ", ".join(
        f"{name} {n / config.PRIVATE_TEST_N:.0%}" for name, n in config.PUBLISHED_BENCHMARKS.items()))
    for name in ("sold_before", "never_sold_before", "building_no_sales"):
        s = result["segments"][name]
        if s["n"]:
            print(f"  segment {name:<14} {metrics.fmt(s)}")
    fp = result["floor_premium"]
    if fp:
        print(f"  floor premium: {fp['per_floor_pct']:+.2%} per floor within a building "
              f"({fp['n_sales']:,} sales, {fp['n_buildings']:,} buildings)")
    print("  top features:", ", ".join(f"{n} {s:.0%}" for n, s in result["importance"][:8]))
    return result


def ablation(data_dir: Path) -> list[dict]:
    """Backtest score as each group of history features is added (for the report's
    'what worked' table). Features are built once; only the model is refit per step."""
    config.N_MODELS = 1  # one model per step keeps this affordable; the steps compare like with like
    flats, sales, static = load_inputs(data_dir)
    test = test_sales(sales, static)
    p = prepare(sales[sales["source"] == "supplied"], static, config.SPLIT_DATE)
    x_test, level = value_features(p.hist, p.index, p.as_of, static, test["gnaf_id"])
    rows, enabled = [], []

    def score(label, model, x, lvl, n_features):
        s = metrics.summarize(test["price"], np.exp(model.predict(x) + lvl))
        rows.append({"step": label, "n_features": n_features, "n": s["n"], "within_10_n": s["within_10_n"],
                     "within_10_share": s["within_10_share"], "median_abs_error": s["median_abs_error"], "r2": s["r2"]})
        print(f"  {label:<40} {metrics.fmt(s)}")

    print(f"\nablation: train before {config.SPLIT_DATE}, {len(test):,} test sales")
    for label, prefixes in features.ABLATION_STEPS:
        if prefixes and not any(c.startswith(tuple(prefixes)) for c in p.x.columns):
            continue  # that source isn't there (not downloaded, or dropped in config)
        enabled += prefixes
        cols = [c for c in p.x.columns
                if not c.startswith(features.GROUP_PREFIXES) or c.startswith(tuple(enabled))]
        score(label, Model().fit(p.x[cols], p.y, p.w, p.train["sale_date"]), x_test, level, len(cols))
    if (sales["source"] == "vg").any():
        p2 = prepare(sales, static, config.SPLIT_DATE)
        x2, level2 = value_features(p2.hist, p2.index, p2.as_of, static, test["gnaf_id"])
        score("+ Valuer General extra sales", Model().fit(p2.x, p2.y, p2.w, p2.train["sale_date"]),
              x2, level2, len(p2.x.columns))
    config.OUT_DIR.mkdir(exist_ok=True)
    (config.OUT_DIR / "ablation.json").write_text(json.dumps(rows, indent=2))
    return rows


def calibration(data_dir: Path) -> dict:
    """Do estimates run high or low? Measured only on earlier windows (train before each split, test
    the next three months), then checked on the saved July-August backtest, which played no part."""
    flats, sales, static = load_inputs(data_dir)
    rows = []
    print("\ncalibration windows (median estimate / sale price, flats under $1.2m):")
    for split in config.CALIBRATION_SPLITS:
        start = pd.Timestamp(split)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        fitted = fit(sales, static, start)
        test = test_sales(sales, static, start=start, end=end)
        pred = predict(fitted, static, test["gnaf_id"], adjust=False)
        s = metrics.summarize(test["price"], pred)
        ratio = float(np.exp(np.median(np.log(pred / test["price"].to_numpy()))))
        rows.append({"split": str(start.date()), "n": len(test), "median_ratio": round(ratio, 4),
                     "within_10_share": s["within_10_share"]})
        print(f"  {start.date()} to {end.date()}: n={len(test)}  median ratio {ratio:.3f}  "
              f"within 10% {s['within_10_share']:.1%}")
    factor = float(np.exp(-np.mean([np.log(r["median_ratio"]) for r in rows])))
    result = {"windows": rows, "suggested_factor": round(factor, 4)}

    bt_file, m_file = config.OUT_DIR / "backtest_predictions.csv", config.OUT_DIR / "backtest_metrics.json"
    if bt_file.exists() and m_file.exists():
        bt = pd.read_csv(bt_file)
        raw = bt["pred"] / json.loads(m_file.read_text()).get("price_adjust", 1.0)
        before, after = metrics.summarize(bt["price"], raw), metrics.summarize(bt["price"], raw * factor)
        result["check"] = {"without": before["within_10_share"], "with": after["within_10_share"],
                           "median_abs_error_without": before["median_abs_error"],
                           "median_abs_error_with": after["median_abs_error"], "n": before["n"]}
        print(f"\nsuggested factor {factor:.3f}. Check on the July-August backtest (n={before['n']}): "
              f"within 10% {before['within_10_share']:.1%} -> {after['within_10_share']:.1%}, "
              f"median miss ${before['median_abs_error']:,.0f} -> ${after['median_abs_error']:,.0f}")
    config.OUT_DIR.mkdir(exist_ok=True)
    (config.OUT_DIR / "calibration.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate(data_dir: Path) -> list[dict]:
    """Within 10% on four separate periods (the three calibration windows and the July-August
    backtest), for one model and for the average of N_MODELS. Use it to judge a change: one
    283-sale test is too noisy on its own."""
    flats, sales, static = load_inputs(data_dir)
    windows = [(pd.Timestamp(s), pd.Timestamp(s) + pd.DateOffset(months=3) - pd.Timedelta(days=1))
               for s in config.CALIBRATION_SPLITS]
    windows.append((pd.Timestamp(config.SPLIT_DATE), pd.Timestamp(config.VALUATION_DATE)))
    rows = []
    for start, end in windows:
        fitted = fit(sales, static, start)
        test = test_sales(sales, static, start=start, end=end)
        x, level = value_features(fitted.hist, fitted.index, fitted.as_of, static, test["gnaf_id"])
        for label, members in (("one model", 1), (f"average of {len(fitted.model.boosters)}", None)):
            pred = np.exp(fitted.model.predict(x, members) + level) * config.PRICE_ADJUST
            s = metrics.summarize(test["price"], pred)
            rows.append({"window": f"{start.date()} to {end.date()}", "model": label, "n": s["n"],
                         "within_10_n": s["within_10_n"], "within_10_share": s["within_10_share"],
                         "median_abs_error": s["median_abs_error"]})
            print(f"  {start.date()} to {end.date()}  {label:<14} {metrics.fmt(s)}", flush=True)
    df = pd.DataFrame(rows)
    print("\nall four periods together:")
    for label, g in df.groupby("model", sort=False):
        print(f"  {label:<14} within 10%: {g['within_10_n'].sum():,} of {g['n'].sum():,} "
              f"({g['within_10_n'].sum() / g['n'].sum():.1%}), mean of period shares "
              f"{g['within_10_share'].mean():.1%}, mean median miss ${g['median_abs_error'].mean():,.0f}")
    config.OUT_DIR.mkdir(exist_ok=True)
    (config.OUT_DIR / "evaluate.json").write_text(json.dumps(rows, indent=2))
    return rows


def team_slug(name: str) -> str:
    slug = re.sub(r"\s+", "-", name.strip().lower())
    return re.sub(r"[^a-z0-9-]", "", slug)


def run_predict(data_dir: Path, team: str) -> Path:
    flats, sales, static = load_inputs(data_dir)
    as_of = pd.Timestamp(config.VALUATION_DATE) + timedelta(days=1)  # sales on 31 Aug count
    fitted = fit(sales, static, as_of)
    values = predict(fitted, static, flats["gnaf_id"])
    out = pd.DataFrame({"gnaf_id": flats["gnaf_id"], "value": np.round(values).astype("int64")})

    path = config.ROOT / f"valuation_{team_slug(team)}.csv"
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    config.OUT_DIR.mkdir(exist_ok=True)
    info = {
        "valuation_date": str(config.VALUATION_DATE),
        "n_train": fitted.n_train,
        "rounds": fitted.model.rounds,
        "n_flats": len(out),
        "value_median": int(out["value"].median()),
        "importance": fitted.model.importance(),
        "index": {str(p): round(float(v), 4) for p, v in fitted.index.items()},
    }
    (config.OUT_DIR / "predict_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {path.name}: {len(out):,} rows, median value ${info['value_median']:,}")
    return path
