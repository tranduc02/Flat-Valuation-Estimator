"""Microburbs hackathon, valuation task.

    python run.py check-api      # is the key working? what do AVM / sale history / properties return?
    python run.py inspect        # columns, blanks, date range of the supplied files
    python run.py fetch-api      # property/dwelling type and floor (where given) per address from the API
    python run.py fetch-osm      # OpenStreetMap buildings (heights) and stations -> data/external/
    python run.py build-vg       # parse Valuer General zips in data/external/vg/, match to gnaf_id
    python run.py backtest       # train before SPLIT_DATE, score on later sales
    python run.py ablation       # backtest score as each group of inputs is added (slower)
    python run.py calibrate      # do estimates run high or low on earlier windows? suggests PRICE_ADJUST
    python run.py predict       # final model -> valuation_<team>.csv (then validates it)
    python run.py validate       # check the CSV against the hand-in rules
    python run.py report         # report_template.html -> report.html
    python run.py fake-data      # synthetic data in data/fake to test everything before 11 am

Add --data-dir data/fake to run any step on the synthetic data.
"""
import argparse
import sys
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check-api", "inspect", "fetch-api", "fetch-osm", "build-vg", "backtest",
                                            "ablation", "calibrate", "predict", "validate", "report", "fake-data"])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--team", default=config.TEAM_NAME, help="team name (default: config.TEAM_NAME)")
    args = parser.parse_args()

    from mbval import pipeline, tools  # imported here so --help stays fast

    if args.command == "check-api":
        tools.check_api()
    elif args.command == "inspect":
        tools.inspect(args.data_dir)
    elif args.command == "fetch-api":
        tools.fetch_api_extras(args.data_dir)
    elif args.command in ("fetch-osm", "build-vg"):
        from mbval import external
        external.fetch_osm() if args.command == "fetch-osm" else external.build_vg(args.data_dir)
    elif args.command == "backtest":
        pipeline.backtest(args.data_dir)
    elif args.command == "ablation":
        pipeline.ablation(args.data_dir)
    elif args.command == "calibrate":
        pipeline.calibration(args.data_dir)
    elif args.command == "predict":
        if args.team == "your team name":
            sys.exit("Set TEAM_NAME in config.py (or pass --team) first: it names the CSV.")
        path = pipeline.run_predict(args.data_dir, args.team)
        tools.validate(path, args.data_dir)
    elif args.command == "validate":
        path = config.ROOT / f"valuation_{pipeline.team_slug(args.team)}.csv"
        sys.exit(0 if tools.validate(path, args.data_dir) else 1)
    elif args.command == "report":
        from mbval import report
        report.render()
    elif args.command == "fake-data":
        from mbval import fake
        fake.make_fake(config.DATA_DIR / "fake")


if __name__ == "__main__":
    main()
