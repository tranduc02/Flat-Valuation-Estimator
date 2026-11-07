# Flat Valuation Estimator

Valuing every City of Sydney flat · Microburbs Hackathon, 7 November 2026 ·
[Task 5: *Value every flat*](https://www.microburbs.com.au/p/hackathon-task-valuation)

> **TODO (team):** one-paragraph pitch: what we built and the headline result, e.g.
> "We valued all 181,711 City of Sydney addresses as at 31 August 2026. On N later sales we held out, X% of our
> estimates were within 10% of the sale price, against Y% for an index-adjusted last sale."

## What we built

A model that estimates the price every supplied City of Sydney address would have sold for on 31 August 2026,
using only information dated by that day. The output is `valuation_<team>.csv` (one row per `gnaf_id`,
whole dollars) and [`report.html`](report.html), which shows how accurate it was on later sales it never saw.

The judges score the share of estimates within 10% of the sale price on 227 private flat sales under $1.2 million
(1 September to 4 November 2026); teams within three percentage points are ranked by median dollar error. The task
page's simple methods scored 34 of 227 (Microburbs' own valuation, which covers only 13% of addresses), 86 (suburb
median by bedrooms) and 109 (each flat's last sale, index-adjusted).

## Who it helps

- **Buyers and sellers:** a price they can check before negotiating, for any flat in the city, including ones
  that have never been advertised.
- **Landlords and everyday investors:** what each flat they own or are watching might sell for.

> **TODO (team):** reword in your own voice; add a concrete example once you have one.

## The clever step

> **TODO (team):** the one non-obvious idea that moved the score, and by how much. Candidates the pipeline already has:
> - **Moving every past sale to today's prices** with a repeat-sales index, so a 2015 sale of the same flat or the
>   flat next door becomes a usable comparable.
> - **Time-honest training:** each training sale only sees sales before its own date, exactly as a 31 August
>   valuation would, so the model doesn't learn to lean on information it won't have.
> - **Floor and position in the building** parsed from the unit number (315 → floor 3), and a within-building
>   floor premium (each floor up vs. flats in the same building) reported by the backtest.

## The data

| Source | What we used | Dated by 31 Aug 2026? |
|---|---|---|
| `flats-city-of-sydney.csv` (supplied) | location, mesh block, unit number, bedrooms/bathrooms/parking where known, flats in building | static; bedrooms come from the last advert held |
| `flat-sales-city-of-sydney.csv` (supplied) | past sale prices and dates | yes: anything later is dropped on load |
| `blocks-homes-and-people-2021-2026.csv` (supplied) | homes in each mesh block, share built since 2021, people per home | 2026 figures are Microburbs estimates, not Census results |
| [NSW Valuer General bulk sales](https://valuation.property.nsw.gov.au/embed/propertySalesInformation) (CC BY-NC-ND 4.0) | 34,000+ extra unit sales the supplied file doesn't have, and strata lot area (flat size) for about 15,700 flats; matched to `gnaf_id` by unit, street number, street and suburb (82% of unit sales matched) | yes: only weekly files released by 31 Aug 2026, contract dates by 31 Aug; in the backtest a sale only counts once it was registered |
| Microburbs API `/v1/suburbs/{suburb}/properties` | property type and dwelling type (apartment, unit, studio, commercial...) for 98% of flats; floor where given (11%, and it agreed with our unit-number parsing on all 19,570 flats where both exist) | static descriptions of the place; we skip its bedrooms/bathrooms/parking, which have no as-at date |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) buildings and stations (ODbL) | building heights, how many taller buildings stand within 100 m and between the flat and the harbour, distance to train/metro and light rail | today's map: a tower finished after an old sale still counts for it (small look-ahead) |

Not used, on purpose:
- `/valuation/avm` covers only 13% of addresses and can't be backtested at our 1 July split.
- `/basics/all` gives current bedrooms and parking with no as-at date, so it might reflect changes after 31 August.

The supplied files, API output, downloaded outside data and the team key are not in this repo (hackathon rules
and the Valuer General's no-republishing licence); `data/` (including `data/external/`), `cache/` and `out/`
are gitignored. The backtest is scored on supplied sales only, so it stays comparable with the judges' test.

## How it works

1. **Price index.** Flats that sold twice give the price change between two dates. A smoothed quarterly
   repeat-sales index (Bailey-Muth-Nourse least squares) turns every past sale into today's price level.
2. **Comparable history.** For each flat: count, mean and latest index-adjusted log price of earlier sales of the
   same flat, the same building, the same building and bedroom count, the same mesh block, the same suburb and
   bedroom count, and the eight nearest other buildings with sales (one search tree per month, so it stays
   time-honest).
3. **Flat details.** Floor, position in the building, bedrooms, bathrooms, parking, building size, distance to the
   harbour and to Town Hall, suburb, block type, shop/office/car space flags, strata lot area (Valuer General),
   homes and people in the mesh block, property and dwelling type (Microburbs API).
4. **Light and view.** From OpenStreetMap heights (or our own buildings' top floor where OSM has none): the flat's
   height is its floor x 3.1 m; we count taller buildings within 100 m, how far the tallest rises above the flat,
   and taller buildings within 250 m in a 60° cone towards the nearest harbour point. Plus distance to stations.
5. **Model.** LightGBM with an L1 loss on index-adjusted log price (targets percentage error, which is what the
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
python run.py fetch-api   # property/dwelling type (and floor where given) from the API -> cache/ (~3 min)
python run.py fetch-osm   # OpenStreetMap buildings and stations -> data/external/
python run.py build-vg    # match Valuer General zips in data/external/vg/ to gnaf_id -> data/external/vg_sales.csv
python run.py backtest    # train before SPLIT_DATE, score on later sales -> out/backtest_metrics.json
python run.py ablation    # score as each group of inputs is added -> out/ablation.json (slower)
python run.py predict     # final model -> valuation_<team>.csv, then validates it
python run.py report      # report_template.html -> report.html
```

The Valuer General site blocks scripted downloads, so get its zips by hand from the
[bulk sales page](https://valuation.property.nsw.gov.au/embed/propertySalesInformation): the yearly files you want
(we used 2005 to 2025) and the 2026 weekly files **up to 31 Aug 2026 only** (later ones hold the judges' test
period). Put them in `data/external/vg/`; `build-vg` skips any weekly file dated after 31 Aug anyway.

`python run.py fake-data` writes synthetic files to `data/fake/`; add `--data-dir data/fake` to any step to try
the pipeline without the real data.

| File | What it is |
|---|---|
| `config.py` | team name, dates, column overrides, model settings |
| `mbval/data.py` | loading, address parsing (building, unit, floor), drops sales after 31 Aug |
| `mbval/index.py` | repeat-sales price index |
| `mbval/features.py` | static and time-honest history features |
| `mbval/model.py` | LightGBM training with time-based early stopping |
| `mbval/pipeline.py` | backtest, baselines, floor premium, ablation, final prediction |
| `mbval/external.py` | OpenStreetMap download and light/view features; Valuer General parsing and address matching |
| `mbval/client.py` | Microburbs API client (retries, rate limit, disk cache) |
| `mbval/tools.py` | API check, data inspection, submission validator |
| `mbval/report.py` | fills `report_template.html` with results and charts |

## Credits

- Data © Microburbs. Estimates are automated and indicative only.
- Addresses: G-NAF © Geoscape Australia, licensed under the Open G-NAF End User Licence Agreement.
- Mesh blocks: © Australian Bureau of Statistics, CC BY 4.0. Mesh block homes and people for 2026 are Microburbs
  estimates, not Census results.
- Property sales: NSW Valuer General bulk property sales information, licensed under
  [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Used for non-commercial analysis only;
  not republished. The Valuer General does not guarantee its completeness or accuracy.
- Buildings and stations: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under
  the Open Database Licence (ODbL).
- API client pattern adapted from the MIT-licensed [Microburbs hackathon demos](https://github.com/Microburbs/microburbs-hackathon-demos).
