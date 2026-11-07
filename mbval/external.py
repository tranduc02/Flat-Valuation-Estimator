"""Outside data, kept under data/external/ (gitignored: never committed or republished).

- OpenStreetMap buildings and stations (c) OpenStreetMap contributors, ODbL. Building heights let
  us ask the task page's question "do nearby buildings block light or a view?", and stations
  give distance to train, metro and light rail.
- NSW Valuer General bulk property sales information, CC BY-NC-ND 4.0. Extra past sales,
  matched to gnaf_id by address. Only weekly files released on or before 31 Aug 2026 are read,
  so nothing registered after the valuation date (and none of the judges' test sales) gets in.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

import config
from mbval import data
from mbval.features import HARBOUR, _km_xy

EXTERNAL_DIR = config.DATA_DIR / "external"
OSM_BUILDINGS = EXTERNAL_DIR / "osm_buildings.csv"
OSM_STATIONS = EXTERNAL_DIR / "osm_stations.csv"
VG_DIR = EXTERNAL_DIR / "vg"
VG_SALES = EXTERNAL_DIR / "vg_sales.csv"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BBOX = (-33.935, 151.160, -33.845, 151.250)  # City of Sydney plus a few hundred metres (S, W, N, E)
STOREY_M = 3.1


# ---------------------------------------------------------------- OpenStreetMap

def _overpass(query: str) -> list[dict]:
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=400,
                      headers={"User-Agent": "flat-valuation-estimator (Microburbs hackathon)"})
    r.raise_for_status()
    return r.json()["elements"]


def _number(value) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(value)) if value is not None else None
    return float(m.group()) if m else np.nan


def _point(el: dict):
    c = el.get("center") or el
    return c.get("lat"), c.get("lon")


def fetch_osm(buildings: bool = True):
    s, w, n, e = BBOX
    bb = f"({s},{w},{n},{e})"
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if buildings:
        _fetch_osm_buildings(bb)
    _fetch_osm_stations(bb)


def _fetch_osm_buildings(bb: str):
    print("asking Overpass for buildings (can take a minute) ...")
    rows = []
    for el in _overpass(f'[out:json][timeout:300];(way["building"]{bb};relation["building"]{bb};);out tags center;'):
        lat, lon = _point(el)
        t = el.get("tags", {})
        if lat is None:
            continue
        rows.append(dict(osm_id=f"{el['type'][0]}{el['id']}", lat=lat, lon=lon, building=t.get("building"),
                         height_m=_number(t.get("height")), levels=_number(t.get("building:levels")),
                         start_date=t.get("start_date")))
    b = pd.DataFrame(rows)
    b.to_csv(OSM_BUILDINGS, index=False)
    known = (b["height_m"].notna() | b["levels"].notna()).mean()
    print(f"  {len(b):,} buildings, {known:.0%} with a height or level count -> {OSM_BUILDINGS.name}")


def _fetch_osm_stations(bb: str):
    rows = []
    q = (f'[out:json][timeout:120];(nwr["railway"~"^(station|halt|tram_stop)$"]{bb};'
         f'nwr["public_transport"="station"]{bb};);out tags center;')
    for el in _overpass(q):
        lat, lon = _point(el)
        t = el.get("tags", {})
        heavy = "yes" in (t.get("train"), t.get("subway")) or t.get("station") in ("subway", "train")
        light = not heavy and (t.get("railway") == "tram_stop" or t.get("station") == "light_rail"
                               or "yes" in (t.get("light_rail"), t.get("tram")))
        rail = heavy or t.get("railway") in ("station", "halt")
        if lat is None or not (light or rail):
            continue
        rows.append(dict(name=t.get("name"), kind="light_rail" if light else "rail", lat=lat, lon=lon))
    st = pd.DataFrame(rows).groupby(["name", "kind"], dropna=False, as_index=False)[["lat", "lon"]].mean()
    st.to_csv(OSM_STATIONS, index=False)
    print(f"  {len(st):,} stations ({(st['kind'] == 'rail').sum()} train/metro, "
          f"{(st['kind'] == 'light_rail').sum()} light rail) -> {OSM_STATIONS.name}")


def osm_features(flats: pd.DataFrame) -> pd.DataFrame | None:
    """Per gnaf_id: own building height, storeys above the flat, taller buildings close by and
    between the flat and the harbour, and distance to stations.

    A flat's height is its floor x 3.1 m; where the floor is unknown, its rank in the building
    x the building's height. Building heights come from OSM (height, else levels x 3.1 m), plus
    our own buildings' top floor where OSM has no height nearby. OSM is today's map, so a sale
    from 2015 can see a tower finished in 2020: a small, known look-ahead.
    """
    if not OSM_BUILDINGS.exists():
        return None
    b = pd.read_csv(OSM_BUILDINGS)
    b["h"] = b["height_m"].fillna(b["levels"] * STOREY_M)
    b = b[b["h"].between(2, 400)]
    bxy, bh = _km_xy(b["lat"], b["lon"]), b["h"].to_numpy()

    f = flats[["gnaf_id", "building", "latitude", "longitude", "floor", "unit_rank"]].copy()
    g = f.groupby("building").agg(lat=("latitude", "mean"), lon=("longitude", "mean"), top=("floor", "max")).dropna(
        subset=["lat", "lon"])
    gxy = _km_xy(g["lat"], g["lon"])
    d, i = cKDTree(bxy).query(gxy, k=1, distance_upper_bound=0.03)
    has_osm = np.isfinite(d)
    top_h = (g["top"].to_numpy() + 1) * STOREY_M
    own_h = np.where(has_osm, bh[np.minimum(i, len(bh) - 1)], top_h)
    add = ~has_osm & np.isfinite(top_h) & (g["top"].to_numpy() >= 1)
    all_xy, all_h = np.vstack([bxy, gxy[add]]), np.concatenate([bh, top_h[add]])
    tree = cKDTree(all_xy)

    hxy = _km_xy([p[0] for p in HARBOUR], [p[1] for p in HARBOUR])
    to_harbour = hxy[cKDTree(hxy).query(gxy)[1]] - gxy
    harbour_km = np.hypot(to_harbour[:, 0], to_harbour[:, 1])

    g = g.assign(own_h=own_h, hx=to_harbour[:, 0], hy=to_harbour[:, 1], harbour_km=harbour_km)
    if OSM_STATIONS.exists():
        st = pd.read_csv(OSM_STATIONS)
        for kind in ("rail", "light_rail"):
            s = st[st["kind"] == kind]
            if len(s):
                g[f"osm_dist_{kind}_km"] = cKDTree(_km_xy(s["lat"], s["lon"])).query(gxy)[0]

    f = f.join(g[["own_h"]], on="building")
    f["flat_h"] = ((f["floor"] + 0.5) * STOREY_M).fillna(f["unit_rank"] * f["own_h"])

    out = np.full((len(f), 4), np.nan)
    pos = pd.Series(np.arange(len(f))).groupby(f["building"].to_numpy()).groups
    neighbours = tree.query_ball_point(gxy, r=0.25)
    for j, (bld, row) in enumerate(g.iterrows()):
        rows = np.asarray(pos.get(bld, []))
        idx = np.asarray(neighbours[j], dtype=int)
        if len(rows) == 0:
            continue
        fh = f["flat_h"].to_numpy()[rows]
        if len(idx):
            v = all_xy[idx] - gxy[j]
            dist = np.hypot(v[:, 0], v[:, 1])
            keep = dist > 0.02                       # skip the flat's own building
            v, dist, h = v[keep], dist[keep], all_h[idx][keep]
        else:
            v, dist, h = np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        near = dist <= 0.1
        cos = (v @ np.array([row["hx"], row["hy"]])) / np.maximum(dist * row["harbour_km"], 1e-9)
        toward = (cos > np.cos(np.radians(30))) & (dist < row["harbour_km"])
        rise = h[None, :] - fh[:, None]
        with np.errstate(invalid="ignore"):
            out[rows, 0] = (near[None, :] & (rise > 0)).sum(1)
            out[rows, 1] = np.where(near[None, :], rise, -np.inf).max(1).clip(0) if near.any() else 0
            out[rows, 2] = (toward[None, :] & (rise > 0)).sum(1)
        out[rows, 3] = len(dist[near])
    out[np.isnan(f["flat_h"].to_numpy()), :3] = np.nan

    res = pd.DataFrame(out, columns=["osm_taller_100m", "osm_rise_100m_m", "osm_harbour_blockers", "osm_buildings_100m"])
    res.insert(0, "gnaf_id", f["gnaf_id"].to_numpy())
    res["osm_own_height_m"] = f["own_h"].to_numpy()
    res["osm_storeys_above"] = ((f["own_h"] - f["flat_h"]) / STOREY_M).clip(lower=0).to_numpy()
    for col in ("osm_dist_rail_km", "osm_dist_light_rail_km"):
        if col in g:
            res[col] = f["building"].map(g[col]).to_numpy()
    return res


# ---------------------------------------------------------------- NSW Valuer General

VG_COLS = ["rec", "district", "property_id", "sale_counter", "download", "property_name", "unit", "house",
           "street", "locality", "postcode", "area", "area_type", "contract_date", "settlement_date", "price",
           "zoning", "nature", "purpose", "strata_lot", "component", "sale_code", "interest", "dealing"]

STREET_TYPES = {
    "STREET": "ST", "ST": "ST", "ROAD": "RD", "RD": "RD", "AVENUE": "AVE", "AVE": "AVE", "AV": "AVE",
    "LANE": "LANE", "LN": "LANE", "PLACE": "PL", "PL": "PL", "PARADE": "PDE", "PDE": "PDE", "DRIVE": "DR",
    "DR": "DR", "CRESCENT": "CRES", "CRES": "CRES", "CR": "CRES", "TERRACE": "TCE", "TCE": "TCE",
    "HIGHWAY": "HWY", "HWY": "HWY", "BOULEVARD": "BVD", "BOULEVARDE": "BVD", "BVD": "BVD", "BLVD": "BVD",
    "CLOSE": "CL", "CL": "CL", "COURT": "CT", "CT": "CT", "SQUARE": "SQ", "SQ": "SQ", "WAY": "WAY",
    "CIRCUIT": "CCT", "CCT": "CCT", "ESPLANADE": "ESP", "ESP": "ESP", "GROVE": "GR", "GR": "GR",
    "PROMENADE": "PROM", "PROM": "PROM", "QUAY": "QY", "QY": "QY", "WALK": "WALK", "ROW": "ROW",
    "MEWS": "MEWS", "GARDENS": "GDNS", "GDNS": "GDNS", "GDN": "GDNS", "GARDEN": "GDNS", "STEPS": "STPS", "LOOP": "LOOP", "PATH": "PATH",
    "BROADWAY": "BROADWAY", "CIRCUS": "CIR", "CRESENT": "CRES",
}


def _dat_texts(zf: zipfile.ZipFile):
    """Every .DAT file's text inside a zip, following zips nested inside it."""
    for info in zf.infolist():
        name = info.filename.lower()
        if name.endswith(".zip"):
            yield from _dat_texts(zipfile.ZipFile(io.BytesIO(zf.read(info))))
        elif name.endswith(".dat"):
            yield zf.read(info).decode("latin-1")


def parse_vg(vg_dir: Path = VG_DIR) -> pd.DataFrame:
    cutoff = pd.Timestamp(config.VALUATION_DATE)
    rows = []
    for path in sorted(Path(vg_dir).glob("*.zip")):
        if re.fullmatch(r"\d{8}", path.stem) and pd.Timestamp(path.stem) > cutoff:
            print(f"  skipping {path.name}: released after {cutoff.date()}")
            continue
        for text in _dat_texts(zipfile.ZipFile(path)):
            for line in text.splitlines():
                if not line.startswith("B;"):
                    continue
                fld = line.split(";")
                if len(fld) < 24 or not fld[6].strip() or not fld[10].strip().isdigit():
                    continue
                if 2000 <= int(fld[10]) <= 2050:   # inner Sydney postcodes, strata (unit) sales only
                    rows.append(fld[:24])
    df = pd.DataFrame(rows, columns=VG_COLS)
    if df.empty:
        return df
    df["contract_date"] = pd.to_datetime(df["contract_date"], format="%Y%m%d", errors="coerce")
    df["available_date"] = pd.to_datetime(df["download"].str[:8], format="%Y%m%d", errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    area = pd.to_numeric(df["area"], errors="coerce")
    df["area_m2"] = np.where(df["area_type"].str.upper() == "H", area * 10_000, area)
    df = df.dropna(subset=["contract_date", "price"])
    df = df[(df["contract_date"] <= cutoff) & (df["available_date"] <= cutoff)
            & (df["contract_date"] >= pd.Timestamp("1990-01-01"))]  # drops typos like year 1026
    df = df.sort_values("available_date").drop_duplicates(["district", "property_id", "sale_counter"], keep="last")
    return df.reset_index(drop=True)


def _suburb_key(s) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(s)).upper().strip()


def _street_key(s) -> tuple[str, str]:
    w = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper()).split()
    if len(w) >= 2 and w[-1] in STREET_TYPES:
        return " ".join(w[:-1]), STREET_TYPES[w[-1]]
    return " ".join(w), ""


def _unit_key(u) -> str:
    u = re.sub(r"[^A-Z0-9]", "", str(u).upper())
    return re.sub(r"^0+(?=\d)", "", u)


def _house_keys(h) -> tuple[str, str]:
    h = re.sub(r"\s", "", str(h).upper())
    m = re.match(r"\d+", h)
    return h, (m.group() if m else h)


def _keys(suburb, street, house, unit):
    base, typ = _street_key(street)
    full, first = _house_keys(house)
    s, u = _suburb_key(suburb), _unit_key(unit)
    return f"{s}|{base}|{typ}|{full}|{u}", f"{s}|{base}|{first}|{u}"


def _flat_keys(flats: pd.DataFrame) -> pd.DataFrame:
    parts = flats["address"].map(data._split_address)
    unit_text, street = parts.str[0], parts.str[1]
    m = street.str.extract(r"^(\d+[A-Z]?(?:\s*-\s*\d+[A-Z]?)?)\s+(.+)$", flags=re.I)
    unit = flats["unit_number"] if "unit_number" in flats else pd.Series(np.nan, index=flats.index)
    unit = unit.fillna(unit_text.str.extract(r"([A-Z]?\d+[A-Z]?)\s*$", flags=re.I)[0])
    # "APARTMENT 101", "SUITE 9", "Unit 5" -> "101", "9", "5" (the VG file has the bare number)
    unit = unit.str.upper().str.replace(
        r"^(?:APARTMENT|APT|UNIT|FLAT|SUITE|SHOP|STUDIO|TOWNHOUSE|PENTHOUSE|CARSPACE|STORE|OFFICE|KIOSK|ROOM)\s+",
        "", regex=True)
    keys = [_keys(s, st, h, u) if isinstance(h, str) and isinstance(u, str) else (None, None)
            for s, st, h, u in zip(flats["suburb"], m[1], m[0], unit)]
    return pd.DataFrame(keys, columns=["k1", "k2"], index=flats.index).assign(gnaf_id=flats["gnaf_id"])


def build_vg(data_dir: Path):
    """Parse the zips in data/external/vg/, match unit sales to gnaf_id, save vg_sales.csv."""
    flats = data.load_flats(data_dir)
    vg = parse_vg()
    if vg.empty:
        print(f"no Valuer General unit sales found in {VG_DIR}; download the zips there first")
        return
    fk = _flat_keys(flats)
    maps = []
    for col in ("k1", "k2"):
        k = fk.dropna(subset=[col])
        k = k[~k[col].duplicated(keep=False)]         # only keys that name exactly one flat
        maps.append(k.set_index(col)["gnaf_id"])
    keys = [_keys(*r) for r in vg[["locality", "street", "house", "unit"]].itertuples(index=False)]
    vg["gnaf_id"] = pd.Series([k[0] for k in keys]).map(maps[0]).fillna(pd.Series([k[1] for k in keys]).map(maps[1]))

    suburbs = set(flats["suburb"].map(_suburb_key))
    in_area = vg["locality"].str.upper().str.strip().isin(suburbs)
    matched = vg[vg["gnaf_id"].notna()]
    out = matched[["gnaf_id", "contract_date", "price", "available_date", "area_m2", "purpose", "nature"]].rename(
        columns={"contract_date": "sale_date"})
    out.to_csv(VG_SALES, index=False)
    print(f"Valuer General: {len(vg):,} unit sales read ({vg['contract_date'].min().date()} to "
          f"{vg['contract_date'].max().date()}), {in_area.sum():,} in our suburbs, "
          f"{len(out):,} matched to a gnaf_id ({len(out) / max(in_area.sum(), 1):.0%})")
    print(f"  area filled on {out['area_m2'].notna().mean():.0%} of matched sales -> {VG_SALES.name}")
    unmatched = vg[in_area & vg["gnaf_id"].isna()]
    if len(unmatched):
        print("  unmatched examples:", unmatched[["unit", "house", "street", "locality"]].head(5).to_dict("records"))


def load_vg_sales(supplied: pd.DataFrame) -> pd.DataFrame | None:
    """Matched VG sales that the supplied file doesn't already have (same flat within 120 days)."""
    if not (config.USE_VG_SALES and VG_SALES.exists()):
        return None
    vg = pd.read_csv(VG_SALES, dtype={"gnaf_id": str}, parse_dates=["sale_date", "available_date"])
    vg = vg[vg["sale_date"] <= pd.Timestamp(config.VALUATION_DATE)]
    both = vg.reset_index().merge(supplied[["gnaf_id", "sale_date"]], on="gnaf_id", how="inner",
                                  suffixes=("", "_s"))
    dup = both.loc[(both["sale_date"] - both["sale_date_s"]).abs() <= pd.Timedelta(days=120), "index"].unique()
    return vg.drop(index=dup)[["gnaf_id", "sale_date", "price", "available_date"]]


def vg_area(flats: pd.DataFrame) -> pd.DataFrame | None:
    """Strata lot area from VG sales, where it looks like one flat's area (under 500 m2)."""
    if not VG_SALES.exists():
        return None
    vg = pd.read_csv(VG_SALES, dtype={"gnaf_id": str})
    a = vg[vg["area_m2"].between(15, 500)].groupby("gnaf_id")["area_m2"].median()
    if a.empty:
        return None
    return a.rename("vg_lot_area_m2").reset_index()
