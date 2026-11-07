"""Checks around the pipeline: API smoke test, data inspection, API extras, submission validation."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

import config
from mbval import data
from mbval.client import DEFAULT_BASE_URL, Microburbs, MicroburbsError

EXAMPLE_GNAF = "GANSW705970502"  # 26 Tudor Street, Surry Hills (the docs' worked example)


def _show(label: str, value, limit: int = 600):
    text = json.dumps(value, default=str)
    print(f"\n{label}\n  {text[:limit]}{' ...' if len(text) > limit else ''}")


def check_api():
    ping = requests.get(f"{DEFAULT_BASE_URL}/v1/ping", timeout=15)
    print(f"ping (no key): HTTP {ping.status_code}")
    mb = Microburbs()  # no cache: we want live answers here
    checks = [
        ("me", "/v1/me", {}),
        ("geocode address", "/v1/geocode/address", {"address": "26 Tudor Street, Surry Hills"}),
        ("AVM (check prediction_date vs 31 Aug cutoff)", f"/v1/properties/{EXAMPLE_GNAF}/valuation/avm", {}),
        ("sale history", f"/v1/properties/{EXAMPLE_GNAF}/sale-history", {}),
        ("suburb properties, first rows (floor/level fields?)", "/v1/suburbs/Surry Hills/properties",
         {"limit": 3, "offset": 0}),
    ]
    for label, path, params in checks:
        try:
            _show(label, mb.get_body(path, **params))
        except MicroburbsError as e:
            print(f"\n{label}\n  {e}")


def inspect(data_dir: Path):
    data_dir = Path(data_dir)
    for name in (config.FLATS_FILE, config.SALES_FILE, config.REPEAT_SALES_FILE):
        path = data_dir / name
        if not path.exists():
            print(f"\n== {name}: not found in {data_dir}")
            continue
        df = data.read_csv(path)
        print(f"\n== {name}: {len(df):,} rows x {df.shape[1]} columns")
        nulls = df.isna().sum()
        for col in df.columns:
            print(f"  {col:<28} {str(df[col].dtype):<10} blank {nulls[col]:>9,}   e.g. {df[col].dropna().head(2).tolist()}")

    flats_path = data_dir / config.FLATS_FILE
    if flats_path.exists():
        flats = data.load_flats(data_dir)
        print(f"\nflats parsing: {flats['building'].nunique():,} buildings, "
              f"floor known for {flats['floor'].notna().mean():.0%}, bedrooms known for {flats['bedrooms'].notna().mean():.0%}, "
              f"{flats['is_nonres'].sum():,} shop/office/car space/storage addresses")
        print("  suburbs:", flats["suburb"].value_counts().head(40).to_dict())
        print("  sample:", flats[["address", "building", "unit_num", "floor", "unit_rank"]].head(5).to_dict("records"))

    if (data_dir / config.SALES_FILE).exists():
        sales = data.load_sales(data_dir)
        print(f"  date range {sales['sale_date'].min().date()} to {sales['sale_date'].max().date()}")
        print("  price quantiles:", sales["price"].quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(-3).to_dict())
        print("  sales per year:", sales["sale_date"].dt.year.value_counts().sort_index().to_dict())
        if flats_path.exists():
            print(f"  {sales['gnaf_id'].isin(flats['gnaf_id']).mean():.1%} of sales match a flats gnaf_id; "
                  f"{flats['gnaf_id'].isin(sales['gnaf_id']).mean():.1%} of flats have at least one sale")


def fetch_api_extras(data_dir: Path):
    """Per-address property and dwelling type (and floor/level, if the API ever adds them) from
    /v1/suburbs/{suburb}/properties, saved to cache/api_properties.csv.

    We keep only the type fields: they describe what the place is (apartment, older-style unit,
    commercial suite...) and don't change over time. We deliberately skip the API's bedrooms,
    bathrooms and car spaces, which carry no as-at date (the task page warns against them)."""
    flats = data.load_flats(data_dir)
    mb = Microburbs(cache_dir=config.CACHE_DIR)
    rows = []
    # The API names suburbs exactly as the flats file does ("Waterloo (NSW)"). Suburbs outside its
    # 29-suburb area (e.g. Newtown, Paddington) answer 403 and are skipped.

    def one(suburb):
        try:
            got = mb.get_all_pages(f"/v1/suburbs/{suburb}/properties", page_size=200)
            print(f"  {suburb}: {len(got):,} properties", flush=True)
            return got
        except MicroburbsError as e:
            print(f"  {suburb}: {str(e)[:90]}", flush=True)
            return []

    with ThreadPoolExecutor(max_workers=4) as pool:  # the client's throttle keeps the total under 8 rps
        for got in pool.map(one, sorted(flats["suburb"].unique())):
            rows.extend(got)
    if not rows:
        print("no rows fetched")
        return
    df = pd.DataFrame(rows)
    id_col = next(c for c in ("gnaf_pid", "gnaf_id", "id") if c in df)
    out = pd.DataFrame({"gnaf_id": df[id_col].astype(str)})
    for col in ("property_type", "dwelling_type"):
        if col in df:
            out[f"api_{col}"] = df[col].fillna("unknown").astype(str).str.lower()
    for col in ("floor", "level"):
        if col in df:
            out[f"api_{col}"] = pd.to_numeric(df[col].astype(str).str.extract(r"(-?\d+)")[0], errors="coerce")
    out = out.drop_duplicates("gnaf_id")
    config.CACHE_DIR.mkdir(exist_ok=True)
    out.to_csv(config.CACHE_DIR / "api_properties.csv", index=False)
    matched = flats["gnaf_id"].isin(out["gnaf_id"]).mean()
    print(f"saved {len(out):,} rows to cache/api_properties.csv; columns {list(out.columns[1:])}; "
          f"{matched:.0%} of flats matched")


def validate(path: Path, data_dir: Path) -> bool:
    path = Path(path)
    flats = data.read_csv(Path(data_dir) / config.FLATS_FILE)
    problems = []
    if not re.fullmatch(r"valuation_[a-z0-9-]+\.csv", path.name):
        problems.append(f"file name {path.name!r} should be valuation_<team>.csv (lower case, letters, digits, hyphens)")
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM; write plain UTF-8")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append("file is not valid UTF-8")
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(sub.columns) != ["gnaf_id", "value"]:
        problems.append(f"header is {list(sub.columns)}; must be exactly gnaf_id,value")
    else:
        if sub["gnaf_id"].duplicated().any():
            problems.append(f"{sub['gnaf_id'].duplicated().sum():,} duplicate gnaf_id rows")
        want, got = set(flats["gnaf_id"]), set(sub["gnaf_id"])
        if want - got:
            problems.append(f"{len(want - got):,} input IDs missing")
        if got - want:
            problems.append(f"{len(got - want):,} IDs not in the input")
        bad = ~sub["value"].str.fullmatch(r"\d+")
        if bad.any():
            problems.append(f"{bad.sum():,} values are blank or not whole dollars, e.g. {sub.loc[bad, 'value'].head(3).tolist()}")
        elif (sub["value"].astype("int64") <= 0).any():
            problems.append("some values are zero")
    print(f"{path.name}: {len(sub):,} rows (input has {len(flats):,})")
    for p in problems:
        print(f"  FAIL {p}")
    if not problems:
        print("  OK: header, IDs, whole-dollar values and encoding all pass")
    return not problems
