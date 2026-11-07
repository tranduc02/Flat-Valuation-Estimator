"""Settings for the Microburbs valuation task. Edit TEAM_NAME first."""
from datetime import date
from pathlib import Path

# Exactly as registered. The CSV name is derived from it ("Data Wranglers" -> data-wranglers).
TEAM_NAME = "can't find a name"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"    # put the supplied CSVs here (gitignored)
CACHE_DIR = ROOT / "cache"  # API responses (gitignored: API output must not be committed)
OUT_DIR = ROOT / "out"      # backtest predictions, metrics, charts (gitignored)

FLATS_FILE = "flats-city-of-sydney.csv"
SALES_FILE = "flat-sales-city-of-sydney.csv"
REPEAT_SALES_FILE = "repeat-sales-city-of-sydney.csv"
BLOCKS_FILE = "blocks-homes-and-people-2021-2026.csv"  # supplied: homes and people per mesh block

# We estimate what each flat would have sold for on this date, using only information dated by it.
VALUATION_DATE = date(2026, 8, 31)

# Backtest: build from sales before SPLIT_DATE, test on sales from SPLIT_DATE to VALUATION_DATE.
# Two months mirrors the real gap (31 Aug -> private sales 1 Sep to 4 Nov).
SPLIT_DATE = date(2026, 7, 1)

# Multiply every estimate by this. Set from `python run.py calibrate`, which measures whether estimates
# run high or low on earlier three-month windows (never on the July-August backtest it is checked on).
PRICE_ADJUST = 1.0
CALIBRATION_SPLITS = (date(2025, 10, 1), date(2026, 1, 1), date(2026, 4, 1))

# Judges score confirmed flats sold under this price, by share within 10%.
SCORED_PRICE_CAP = 1_200_000
WITHIN_PCT = 0.10

# Task page: how the simple methods did on the judges' private set (227 flat sales, 1 Sep - 4 Nov 2026).
PRIVATE_TEST_N = 227
PUBLISHED_BENCHMARKS = {
    "Microburbs' own valuation (covers 13% of addresses)": 34,
    "suburb median by bedrooms": 86,
    "last sale, index-adjusted": 109,
}

# Sales file columns. None = auto-detect; run `python run.py inspect` and set these if it guesses wrong.
SALES_ID_COL = "gnaf_id"
SALES_DATE_COL = None
SALES_PRICE_COL = None

# Training window and cleaning.
TRAIN_SINCE = date(2010, 1, 1)
MIN_PRICE = 100_000          # below this is mostly car spaces, storage and part-shares
MAX_PRICE = 10_000_000
RECENCY_HALF_LIFE_YEARS = 4  # older sales count less in training

# Input groups to leave out, by feature-name prefix: "own_", "bld_", "bldbed_", "mb_", "subbed_", "nbr_",
# "osm_" (OpenStreetMap buildings/stations), "vg_" (Valuer General lot area).
# Run `python run.py ablation` first and drop a group only if adding it made the backtest worse.
DROP_FEATURE_PREFIXES = ()

# Add matched NSW Valuer General sales (data/external/vg_sales.csv, from `run.py build-vg`) to the
# supplied sales history. The backtest is still scored on supplied sales only.
USE_VG_SALES = True

# Price index: period length for the repeat-sales index ("Q" = quarterly, "M" = monthly).
INDEX_FREQ = "Q"
INDEX_SMOOTHING = 10.0  # penalty on period-to-period jumps; higher = smoother index

LGBM_PARAMS = dict(
    objective="l1",          # median regression in log space ~ minimises typical % error
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
)
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 200
SEED = 42
