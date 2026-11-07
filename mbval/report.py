"""Render report.html from report_template.html plus the backtest and prediction outputs.

Edit the words in report_template.html, then rerun `python run.py report`;
{{placeholders}} are filled from out/backtest_metrics.json and out/predict_info.json.
Charts are embedded as images so report.html is a single self-contained file.
"""
import base64
import html
import io
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, GREY, GREEN, AMBER, RED = "#2f6fdb", "#9aa5b1", "#2f9e6a", "#e8a33d", "#d64545"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})


def _img(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def chart_compare(m: dict) -> str:
    rows = [("Our model", m["model"], BLUE),
            ("Last sale, index-adjusted", m["baselines"]["last_sale_indexed"], GREY),
            ("Suburb median by bedrooms", m["baselines"]["suburb_bedroom_median"], GREY)]
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    for i, (label, s, color) in enumerate(rows):
        ax.barh(i, s["within_10_share"], color=color, height=0.6)
        ax.text(s["within_10_share"] + 0.01, i, f"{s['within_10_share']:.0%}  ({s['within_10_n']:,} of {s['n']:,})",
                va="center", color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of later sales estimated within 10%")
    return _img(fig, "Share within 10%: our model compared with two simple benchmarks")


def chart_bands(m: dict) -> str:
    bands = m["model"]["bands"]
    colors = [GREEN, GREEN, AMBER, RED]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(list(bands), list(bands.values()), color=colors, width=0.6)
    for i, v in enumerate(bands.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", color=INK)
    ax.set_ylabel("Later sales")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    return _img(fig, "How far our estimates were from the later sale prices")


def chart_scatter(preds: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    within = (preds["pred"] - preds["price"]).abs() / preds["price"] <= config.WITHIN_PCT
    ax.scatter(preds.loc[~within, "price"], preds.loc[~within, "pred"], s=8, color=GREY, alpha=0.6, label="Outside 10%")
    ax.scatter(preds.loc[within, "price"], preds.loc[within, "pred"], s=8, color=BLUE, alpha=0.7, label="Within 10%")
    lo, hi = preds[["price", "pred"]].min().min() * 0.9, preds[["price", "pred"]].max().max() * 1.1
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK, lw=1)
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BLUE, alpha=0.08, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    money = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1e6:.1f}m" if v >= 1e6 else f"${v / 1e3:.0f}k")
    ax.xaxis.set_major_formatter(money)
    ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Sale price")
    ax.set_ylabel("Our estimate")
    ax.legend(frameon=False, loc="upper left")
    return _img(fig, "Estimated value against sale price for the held-out sales")


def chart_index(index: dict) -> str:
    s = pd.Series(index)
    level = 100 * np.exp(s - s.iloc[0])
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(range(len(level)), level.to_numpy(), color=BLUE, lw=2)
    years = [p[:4] for p in level.index]
    ticks = [i for i, y in enumerate(years) if (i == 0 or years[i - 1] != y) and int(y) % 2 == 0]
    ax.set_xticks(ticks, [years[i] for i in ticks])
    ax.set_ylabel(f"Index ({level.index[0][:4]} = 100)")
    ax.grid(axis="y", color=GRID)
    return _img(fig, "City of Sydney flat price index from repeat sales")


def chart_floor(fp: dict) -> str:
    floors = sorted(fp["curve"], key=int)
    pct = [fp["curve"][f][0] for f in floors]
    xs = [int(f) for f in floors]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(xs, pct, color=[BLUE if p >= 0 else GREY for p in pct], width=0.7)
    line = np.array([min(xs), max(xs)])
    ax.plot(line, np.expm1(np.log1p(fp["per_floor_pct"]) * line), color=INK, lw=1,
            label=f"{fp['per_floor_pct']:+.1%} per floor")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Floors above (+) or below (-) the building's average")
    ax.set_ylabel("Price vs building average")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    return _img(fig, "Flats on higher floors sell for more than others in the same building")


def ablation_table(rows: list) -> str:
    head = "<tr><th>Inputs</th><th>Within 10%</th><th>Typical miss</th><th>R squared</th></tr>"
    body = "".join(
        f"<tr><td>{html.escape(r['step'])}</td><td>{r['within_10_share']:.1%} ({r['within_10_n']:,} of {r['n']:,})</td>"
        f"<td>{_money(r['median_abs_error'])}</td><td>{r['r2']:.3f}</td></tr>" for r in rows)
    return f"<table>{head}{body}</table>"


def chart_importance(importance: list) -> str:
    top = importance[:12][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([n for n, _ in top], [s for _, s in top], color=BLUE, height=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of the model's total gain")
    return _img(fig, "Which inputs the model relied on most")


def _money(v) -> str:
    return f"${v:,.0f}" if v is not None else "n/a"


def _pct(v) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def build_values() -> dict:
    v = {}
    mfile, pfile = config.OUT_DIR / "backtest_metrics.json", config.OUT_DIR / "predict_info.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        model, base, seg = m["model"], m["baselines"], m["segments"]
        v.update(
            split_date=m["split_date"], test_end=m["test_end"], n_train=f"{m['n_train']:,}", n_test=f"{m['n_test']:,}",
            within_n=f"{model['within_10_n']:,}", within_share=_pct(model["within_10_share"]),
            median_abs_error=_money(model["median_abs_error"]), r2=f"{model['r2']:.3f}",
            n_missing=f"{model['n_missing']:,}",
            base_median_share=_pct(base["suburb_bedroom_median"]["within_10_share"]),
            base_last_share=_pct(base["last_sale_indexed"]["within_10_share"]),
            sold_before_share=_pct(seg["sold_before"]["within_10_share"]), sold_before_n=f"{seg['sold_before']['n']:,}",
            never_sold_share=_pct(seg["never_sold_before"]["within_10_share"]),
            never_sold_n=f"{seg['never_sold_before']['n']:,}",
            chart_compare=chart_compare(m), chart_bands=chart_bands(m), chart_index=chart_index(m["index"]),
            chart_importance=chart_importance(m["importance"]),
            bands_table="".join(f"<tr><td>{html.escape(k)}</td><td>{n:,}</td></tr>" for k, n in model["bands"].items()),
        )
        fp = m.get("floor_premium")
        if fp:
            v.update(floor_premium_pct=f"{fp['per_floor_pct']:+.1%}", floor_premium_n=f"{fp['n_sales']:,}",
                     floor_premium_buildings=f"{fp['n_buildings']:,}", chart_floor=chart_floor(fp))
        full = m.get("model_all_prices")
        if full:
            v.update(all_prices_share=_pct(full["within_10_share"]), all_prices_n=f"{full['n']:,}",
                     all_prices_r2=f"{full['r2']:.3f}")
        ext = m.get("external") or {}
        v.update(vg_sales_added=f"{ext.get('vg_sales_added', 0):,}", vg_area_n=f"{ext.get('vg_area_n', 0):,}",
                 osm_height_share=_pct(ext.get("osm_height_share")))
        nb = seg.get("building_no_sales")
        if nb and nb["n"]:
            v.update(no_bld_share=_pct(nb["within_10_share"]), no_bld_n=f"{nb['n']:,}")
        preds_file = config.OUT_DIR / "backtest_predictions.csv"
        if preds_file.exists():
            v["chart_scatter"] = chart_scatter(pd.read_csv(preds_file))
    afile = config.OUT_DIR / "ablation.json"
    if afile.exists():
        v["ablation_table"] = ablation_table(json.loads(afile.read_text()))
    n = config.PRIVATE_TEST_N
    v["published_benchmarks"] = "; ".join(
        f"{html.escape(name)}: {k} of {n} ({k / n:.0%})" for name, k in config.PUBLISHED_BENCHMARKS.items())
    if pfile.exists():
        p = json.loads(pfile.read_text())
        v.update(n_flats=f"{p['n_flats']:,}", value_median=_money(p["value_median"]),
                 valuation_date=p["valuation_date"], n_train_final=f"{p['n_train']:,}")
    return v


def render():
    template = (config.ROOT / "report_template.html").read_text(encoding="utf-8")
    values = build_values()
    missing = set()

    def fill(match):
        key = match.group(1)
        if key in values:
            return values[key]
        missing.add(key)
        return '<span class="todo">TBD</span>'

    out = re.sub(r"\{\{\s*(\w+)\s*\}\}", fill, template)
    (config.ROOT / "report.html").write_text(out, encoding="utf-8")
    print(f"wrote report.html ({len(out) / 1024:,.0f} KB)")
    if missing:
        print(f"  not filled yet: {', '.join(sorted(missing))} (run backtest and predict first)")
    todos = out.count('class="todo"')
    if todos:
        print(f"  {todos} TODO/TBD markers left to write or fill")
