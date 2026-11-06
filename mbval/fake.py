"""Synthetic data in the supplied files' shape, to test the pipeline before 11 am.

The flats file follows the columns on the task page. The real sales file's
columns are unknown, so this uses sold_date / sold_price to exercise auto-detection.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config

SUBURBS = {  # name: (lat, lon, price multiplier, postcode)
    "Surry Hills": (-33.886, 151.211, 1.05, "2010"),
    "Waterloo (NSW)": (-33.900, 151.207, 0.85, "2017"),
    "Pyrmont": (-33.870, 151.194, 1.10, "2009"),
    "Glebe (NSW)": (-33.879, 151.185, 0.95, "2037"),
    "Zetland": (-33.906, 151.210, 0.90, "2017"),
    "Haymarket": (-33.880, 151.204, 0.80, "2000"),
}
STREETS = ["George", "Crown", "Harris", "Bourke", "Elizabeth", "Powell", "Wattle", "Bay"]


def log_index(t: pd.Series) -> np.ndarray:
    """A made-up flat price path: rises, a 2017-19 dip, a 2020-21 boom, a 2022 fall, then growth."""
    y = (t - pd.Timestamp("2008-01-01")).dt.days.to_numpy() / 365.25 + 2008
    knots_x = [2008, 2017, 2019, 2022, 2023, 2027]
    knots_y = [0.0, 0.55, 0.45, 0.70, 0.64, 0.84]
    return np.interp(y, knots_x, knots_y)


def make_fake(out_dir: Path, n_buildings: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    flats, values = [], []
    i = 0
    for b in range(n_buildings):
        suburb = rng.choice(list(SUBURBS))
        lat0, lon0, mult, postcode = SUBURBS[suburb]
        lat, lon = lat0 + rng.normal(0, 0.004), lon0 + rng.normal(0, 0.004)
        street = f"{rng.integers(1, 400)} {rng.choice(STREETS)} Street"
        floors, per_floor = int(rng.integers(2, 25)), int(rng.integers(2, 8))
        quality = rng.normal(0, 0.15)
        mb = str(11_200_000_000 + b * 7)
        units = [(fl, fl * 100 + k) for fl in range(1, floors + 1) for k in range(1, per_floor + 1)]
        extras = [("Shop", 1), ("Car Space", 1)] if rng.random() < 0.2 else []
        for fl, unit in units:
            beds = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.45, 0.35, 0.1]))
            value = (420_000 + 300_000 * beds) * mult * np.exp(quality + rng.normal(0, 0.05)) * (1 + 0.012 * fl)
            flats.append(dict(
                gnaf_id=f"GANSW7{i:08d}", address=f"Unit {unit}, {street}", suburb=suburb, postcode=postcode,
                mb=mb, block_type="Residential", latitude=round(lat, 8), longitude=round(lon, 8),
                unit_number=unit,
                bedrooms=beds if rng.random() < 0.3 else None,  # mostly blank, like the real file
                bathrooms=(1 + (beds >= 2)) if rng.random() < 0.3 else None,
                car_spaces=int(rng.random() < 0.5) if rng.random() < 0.3 else None,
                flats_in_building=len(units)))
            values.append(value)
            i += 1
        for kind, n in extras:
            flats.append(dict(
                gnaf_id=f"GANSW7{i:08d}", address=f"{kind} {n}, {street}", suburb=suburb, postcode=postcode,
                mb=mb, block_type="Commercial", latitude=round(lat, 8), longitude=round(lon, 8),
                unit_number=None, bedrooms=None, bathrooms=None, car_spaces=None, flats_in_building=len(units)))
            values.append(60_000 if kind == "Car Space" else 900_000)
            i += 1
    flats = pd.DataFrame(flats)
    flats["unit_number"] = flats["unit_number"].astype("Int64")

    # Sales from 2008 to 4 Nov 2026 (later ones exercise the after-cutoff filter).
    start, end = pd.Timestamp("2008-01-01"), pd.Timestamp("2026-11-04")
    n_sales = rng.poisson(1.4, len(flats))
    idx = np.repeat(np.arange(len(flats)), n_sales)
    dates = start + pd.to_timedelta(rng.integers(0, (end - start).days, len(idx)), unit="D")
    dates = pd.Series(dates)
    level_now = log_index(pd.Series([pd.Timestamp(config.VALUATION_DATE)]))[0]
    price = np.asarray(values)[idx] * np.exp(log_index(dates) - level_now + rng.normal(0, 0.06, len(idx)))
    sales = pd.DataFrame({"gnaf_id": flats["gnaf_id"].to_numpy()[idx],
                          "sold_date": dates.dt.strftime("%Y-%m-%d"),
                          "sold_price": np.round(price, -3).astype("int64")})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flats.to_csv(out_dir / config.FLATS_FILE, index=False)
    sales.sort_values("sold_date").to_csv(out_dir / config.SALES_FILE, index=False)
    print(f"fake data in {out_dir}: {len(flats):,} flats, {len(sales):,} sales")
