# Flat Valuation Estimator

Valuing every City of Sydney flat · Microburbs Hackathon, 7 November 2026 · Task 5: *Value every flat*

> **TODO (team):** one-paragraph pitch: what we built and the headline result, e.g.
> "We valued all 181,711 City of Sydney addresses as at 31 August 2026. On N later sales we held out, X% of our
> estimates were within 10% of the sale price, against Y% for an index-adjusted last sale."

## What we built

A model that estimates the price every supplied City of Sydney address would have sold for on 31 August 2026,
using only information dated by that day. The output is `valuation_<team>.csv` (one row per `gnaf_id`,
whole dollars) and [`report.html`](report.html), which shows how accurate it was on later sales it never saw.

## Who it helps

> **TODO (team):** e.g. buyers and owners checking a flat's value before an auction, lenders checking a valuation,
> agents pricing a listing.

## The clever step

> **TODO (team):** the one non-obvious idea that moved the score, and by how much. Candidates the pipeline already has:
> - **Moving every past sale to today's prices** with a repeat-sales index, so a 2015 sale of the same flat or the
>   flat next door becomes a usable comparable.
> - **Time-honest training:** each training sale only sees sales before its own date, exactly as a 31 August
>   valuation would, so the model doesn't learn to lean on information it won't have.
> - **Floor and position in the building** parsed from the unit number (315 → floor 3).

## The data

| Source | What we used | Dated by 31 Aug 2026? |
|---|---|---|
| `flats-city-of-sydney.csv` (supplied) | location, mesh block, unit number, bedrooms/bathrooms/parking where known, flats in building | static; bedrooms come from the last advert held |
| `flat-sales-city-of-sydney.csv` (supplied) | past sale prices and dates | yes: anything later is dropped on load |
| Microburbs API `/v1/suburbs/{suburb}/properties` (optional) | floor/level per address | static |
| **TODO** | any other source | |

The supplied files, API output and the team key are not in this repo (hackathon rules); `data/`, `cache/`
and `out/` are gitignored.

## How it works

1. **Price index.** Flats that sold twice give the price change between two dates. A smoothed quarterly
   repeat-sales index (Bailey-Muth-Nourse least squares) turns every past sale into today's price level.
2. **Comparable history.** For each flat: count, mean and latest index-adjusted log price of earlier sales of the
   same flat, the same building, the same building and bedroom count, the same mesh block, and the same suburb and
   bedroom count.
3. **Flat details.** Floor, position in the building, bedrooms, bathrooms, parking, building size, distance to the
   harbour and to Town Hall, suburb, block type, shop/office/car space flags.
4. **Model.** LightGBM with an L1 loss on index-adjusted log price (targets percentage error, which is what the
   within-10% score rewards), recent sales weighted more. Value = model output moved to the 31 August index level.

## How to run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then paste the team key into .env
```

Put the supplied CSVs in `data/`, set `TEAM_NAME` in [`config.py`](config.py), then:

```bash
python run.py check-api   # key works? what do AVM / sale history / properties look like?
python run.py inspect     # columns, blanks and date range of the supplied files
python run.py fetch-api   # optional: floor/level per address from the API
python run.py backtest    # train before SPLIT_DATE, score on later sales -> out/backtest_metrics.json
python run.py predict     # final model -> valuation_<team>.csv, then validates it
python run.py report      # report_template.html -> report.html
```

`python run.py fake-data` writes synthetic files to `data/fake/`; add `--data-dir data/fake` to any step to try
the pipeline without the real data.

| File | What it is |
|---|---|
| `config.py` | team name, dates, column overrides, model settings |
| `mbval/data.py` | loading, address parsing (building, unit, floor), drops sales after 31 Aug |
| `mbval/index.py` | repeat-sales price index |
| `mbval/features.py` | static and time-honest history features |
| `mbval/model.py` | LightGBM training with time-based early stopping |
| `mbval/pipeline.py` | backtest, baselines, final prediction |
| `mbval/client.py` | Microburbs API client (retries, rate limit, disk cache) |
| `mbval/tools.py` | API check, data inspection, submission validator |
| `mbval/report.py` | fills `report_template.html` with results and charts |

## Credits

- Data © Microburbs. Estimates are automated and indicative only.
- Addresses: G-NAF © Geoscape Australia, licensed under the Open G-NAF End User Licence Agreement.
- Mesh blocks: © Australian Bureau of Statistics, CC BY 4.0.
- API client pattern adapted from the MIT-licensed [Microburbs hackathon demos](https://github.com/Microburbs/microburbs-hackathon-demos).
- **TODO:** any other source.
