"""Load and clean the supplied CSVs.

The flats file schema is documented on the task page. The sales file's columns
are not, so the id/date/price columns are auto-detected (override in config.py).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

ID_HINTS = ("gnaf_id", "gnaf_pid", "gnaf", "id")
DATE_HINTS = ("sold_date", "sale_date", "date_sold", "contract_date", "settlement_date", "date")
PRICE_HINTS = ("sold_price", "sale_price", "price", "purchase_price", "amount", "value")

TEXT_COLS = {"gnaf_id": str, "gnaf_pid": str, "mb": str, "postcode": str, "unit_number": str}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=TEXT_COLS, low_memory=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_flats(data_dir: Path) -> pd.DataFrame:
    flats = read_csv(Path(data_dir) / config.FLATS_FILE)
    if flats["gnaf_id"].duplicated().any():
        print(f"warning: {flats['gnaf_id'].duplicated().sum()} duplicate gnaf_id rows in flats; keeping first")
        flats = flats.drop_duplicates("gnaf_id")
    for col in ("latitude", "longitude", "bedrooms", "bathrooms", "car_spaces", "flats_in_building"):
        if col in flats:
            flats[col] = pd.to_numeric(flats[col], errors="coerce")
    flats["address"] = flats["address"].fillna("").astype(str)
    flats["suburb"] = flats["suburb"].fillna("unknown").astype(str)
    return add_address_parts(flats)


def add_address_parts(flats: pd.DataFrame) -> pd.DataFrame:
    """Split "Unit 315, 2-4 Powell Street" into a building key, a unit number and a floor guess."""
    parts = flats["address"].map(_split_address)
    unit_text = parts.str[0]
    street = parts.str[1]
    flats["building"] = street.str.upper().str.replace(r"\s+", " ", regex=True).str.strip() + "|" + flats["suburb"]

    unit_from_col = pd.to_numeric(flats.get("unit_number"), errors="coerce") if "unit_number" in flats else np.nan
    unit_from_text = pd.to_numeric(unit_text.str.extract(r"(\d+)\s*[A-Z]?\s*$", flags=re.I)[0], errors="coerce")
    flats["unit_num"] = pd.Series(unit_from_col, index=flats.index).fillna(unit_from_text)

    level = pd.to_numeric(unit_text.str.extract(r"\bLEVEL\s*(\d+)", flags=re.I)[0], errors="coerce")
    ground = unit_text.str.contains(r"\bG\d|\b(?:G|GROUND)\b", case=False, regex=True)
    numbered_by_floor = (flats["unit_num"] >= 100) & (flats["unit_num"] < 8000)  # 315 -> 3, 1203 -> 12
    floor_from_num = np.where(numbered_by_floor, flats["unit_num"] // 100, np.nan)
    flats["floor"] = level.fillna(pd.Series(floor_from_num, index=flats.index))
    flats.loc[ground & flats["floor"].isna(), "floor"] = 0

    # Where a unit sits within its building (0 = lowest number, 1 = highest): a height proxy
    # that also works for small blocks numbered 1, 2, 3...
    flats["unit_rank"] = flats.groupby("building")["unit_num"].rank(pct=True)

    upper = flats["address"].str.upper()
    flats["is_shop"] = upper.str.contains(r"\bSHOP\b", regex=True)
    flats["is_office"] = upper.str.contains(r"\b(?:OFFICE|SUITE)\b", regex=True)
    flats["is_carspace"] = upper.str.contains(r"\b(?:CAR ?SPACE|CARSPACE|PARKING|CAR ?PARK|GARAGE)\b", regex=True)
    flats["is_storage"] = upper.str.contains(r"\b(?:STORAGE|STORE ?ROOM)\b", regex=True)
    flats["is_penthouse"] = upper.str.contains(r"\bPENTHOUSE\b|\bPH\s?\d", regex=True)
    flats["is_nonres"] = flats[["is_shop", "is_office", "is_carspace", "is_storage"]].any(axis=1)
    return flats


def _split_address(address: str) -> tuple[str, str]:
    address = address.strip()
    if "," in address:
        unit, street = address.rsplit(",", 1)
        return unit.strip(), street.strip()
    m = re.match(r"^([^\s/]+)\s*/\s*(.+)$", address)  # "315/2-4 Powell Street"
    if m:
        return m.group(1), m.group(2)
    return "", address


def detect_column(df: pd.DataFrame, hints: tuple[str, ...], kind: str) -> str:
    cols = list(df.columns)
    for hint in hints:
        if hint in cols:
            return hint
    for hint in hints:
        for col in cols:
            if hint in col:
                return col
    raise KeyError(f"Could not find the {kind} column in {cols}. Set it in config.py.")


def parse_price(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def parse_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if parsed.isna().mean() > 0.2:  # maybe day-first, e.g. 06/08/2026
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    return parsed


def load_sales(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Sales as gnaf_id, sale_date, price (+ any other columns, prefixed sale_).
    Drops anything dated after the valuation date: it must never be used."""
    raw = read_csv(Path(data_dir) / config.SALES_FILE)
    id_col = config.SALES_ID_COL if config.SALES_ID_COL in raw else detect_column(raw, ID_HINTS, "id")
    date_col = config.SALES_DATE_COL or detect_column(raw, DATE_HINTS, "sale date")
    price_col = config.SALES_PRICE_COL or detect_column(raw, PRICE_HINTS, "sale price")

    sales = raw.rename(columns={c: f"sale_{c}" for c in raw.columns if c not in (id_col, date_col, price_col)})
    sales = sales.rename(columns={id_col: "gnaf_id", date_col: "sale_date", price_col: "price"})
    sales["gnaf_id"] = sales["gnaf_id"].astype(str).str.strip()
    sales["sale_date"] = parse_date(sales["sale_date"])
    sales["price"] = parse_price(sales["price"])

    n0 = len(sales)
    sales = sales.dropna(subset=["sale_date", "price"])
    cutoff = pd.Timestamp(config.VALUATION_DATE)
    late = sales["sale_date"] > cutoff
    sales = sales[~late].sort_values("sale_date").reset_index(drop=True)
    if verbose:
        print(f"sales: {n0:,} rows; using id={id_col!r} date={date_col!r} price={price_col!r}; "
              f"dropped {n0 - len(sales) - late.sum():,} unparseable and {late.sum():,} dated after {cutoff.date()}")
    return sales
