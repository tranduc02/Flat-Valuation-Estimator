"""Backtest and final prediction.

Both run the same fit(as_of): use sales dated before as_of only, build the price
index, train, and value flats as at as_of. The backtest uses as_of = SPLIT_DATE
and checks against later sales; the final run uses the day after VALUATION_DATE.
"""
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from mbval import data, features, index, metrics
from mbval.model import Model


@dataclass
class Fitted:
    as_of: pd.Timestamp
    model: Model
    index: pd.Series
    hist: pd.DataFrame
    n_train: int


def load_inputs(data_dir: Path):
    flats = data.load_flats(data_dir)
    sales = data.load_sales(data_dir)
    extras_file = config.CACHE_DIR / "api_properties.csv"
    extras = pd.read_csv(extras_file, dtype={"gnaf_id": str}) if extras_file.exists() else None
    if extras is not None:
        print(f"using API extras from {extras_file.name}: {list(extras.columns[1:])}")
    static = features.static_features(flats, extras)
    return flats, sales, static


def fit(sales: pd.DataFrame, static: pd.DataFrame, as_of) -> Fitted:
    as_of = pd.Timestamp(as_of)
    hist = sales[(sales["sale_date"] < as_of)
                 & sales["price"].between(config.MIN_PRICE, config.MAX_PRICE)].copy()
    idx = index.build_index(hist)
    hist["lp_norm"] = np.log(hist["price"]) - index.index_at(idx, hist["sale_date"])

    train = hist[(hist["sale_date"] >= pd.Timestamp(config.TRAIN_SINCE))
                 & hist["gnaf_id"].isin(static.index)].reset_index(drop=True)
    x = features.feature_frame(train.assign(query_date=train["sale_date"]), hist, static)
    w = features.recency_weights(train["sale_date"], as_of)
    model = Model().fit(x, train["lp_norm"].to_numpy(), w, train["sale_date"])
    print(f"trained on {len(train):,} sales ({train['sale_date'].min().date()} to "
          f"{train['sale_date'].max().date()}), {model.rounds} rounds")
    return Fitted(as_of, model, idx, hist, len(train))


def predict(fitted: Fitted, static: pd.DataFrame, gnaf_ids) -> np.ndarray:
    """Estimated price at fitted.as_of for each gnaf_id."""
    q = pd.DataFrame({"gnaf_id": list(gnaf_ids), "query_date": fitted.as_of})
    x = features.feature_frame(q, fitted.hist, static)
    level = index.index_at(fitted.index, [fitted.as_of - timedelta(days=1)])[0]
    return np.exp(fitted.model.predict(x) + level)


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


def backtest(data_dir: Path) -> dict:
    flats, sales, static = load_inputs(data_dir)
    split, end = pd.Timestamp(config.SPLIT_DATE), pd.Timestamp(config.VALUATION_DATE)
    fitted = fit(sales, static, split)

    test = sales[(sales["sale_date"] >= split) & (sales["sale_date"] <= end)
                 & sales["price"].between(config.MIN_PRICE, config.SCORED_PRICE_CAP - 1)
                 & sales["gnaf_id"].isin(static.index)].copy()
    test = test[~static.loc[test["gnaf_id"], "is_nonres"].to_numpy()].reset_index(drop=True)
    if test.empty:
        raise SystemExit("No test sales between SPLIT_DATE and VALUATION_DATE. Move SPLIT_DATE earlier.")

    test["pred"] = predict(fitted, static, test["gnaf_id"])
    for name, values in baselines(fitted, static, test["gnaf_id"]).items():
        test[name] = values
    had_sale = test["gnaf_id"].isin(fitted.hist["gnaf_id"])
    bed_known = static.loc[test["gnaf_id"], "bed_key"].to_numpy() >= 0

    result = {
        "split_date": str(split.date()),
        "test_end": str(end.date()),
        "n_train": fitted.n_train,
        "n_test": len(test),
        "model": metrics.summarize(test["price"], test["pred"]),
        "baselines": {name: metrics.summarize(test["price"], test[name])
                      for name in ("suburb_bedroom_median", "last_sale_indexed")},
        "segments": {
            "sold_before": metrics.summarize(test.loc[had_sale, "price"], test.loc[had_sale, "pred"]),
            "never_sold_before": metrics.summarize(test.loc[~had_sale, "price"], test.loc[~had_sale, "pred"]),
            "bedrooms_known": metrics.summarize(test.loc[bed_known, "price"], test.loc[bed_known, "pred"]),
            "bedrooms_unknown": metrics.summarize(test.loc[~bed_known, "price"], test.loc[~bed_known, "pred"]),
        },
        "importance": fitted.model.importance(),
        "index": {str(p): round(float(v), 4) for p, v in fitted.index.items()},
    }
    config.OUT_DIR.mkdir(exist_ok=True)
    test.to_csv(config.OUT_DIR / "backtest_predictions.csv", index=False)
    (config.OUT_DIR / "backtest_metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\nbacktest: train before {split.date()}, test {split.date()} to {end.date()} "
          f"(under ${config.SCORED_PRICE_CAP:,}, flats only)")
    print(f"  model                  {metrics.fmt(result['model'])}")
    for name, s in result["baselines"].items():
        print(f"  {name:<22} {metrics.fmt(s)}")
    print("  top features:", ", ".join(f"{n} {s:.0%}" for n, s in result["importance"][:8]))
    return result


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
