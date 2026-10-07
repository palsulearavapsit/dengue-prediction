from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Final_data.csv"
OUT_DIR = ROOT / "data_analysis"
OUT_DIR.mkdir(exist_ok=True)


def resolution_summary(label: str, period_alias: str, dengue: pd.DataFrame) -> dict:
    d = dengue.copy()
    d["period"] = d["date"].dt.to_period(period_alias).dt.to_timestamp()
    district_periods = d.groupby(["district", "period"]).size().reset_index(name="n_obs")

    n_obs = int(len(district_periods))
    n_districts = int(district_periods["district"].nunique())
    counts_per_district = district_periods.groupby("district")["period"].nunique()
    valid_pairs = int((counts_per_district - 1).clip(lower=0).sum())

    gaps = []
    for district, sub in district_periods.sort_values(["district", "period"]).groupby("district"):
        times = pd.to_datetime(sub["period"]).drop_duplicates().sort_values().tolist()
        if len(times) > 1:
            diffs = [(times[i + 1] - times[i]).days for i in range(len(times) - 1)]
            gaps.extend(diffs)
    gap_series = pd.Series(gaps, dtype="float64")

    raw_rows = len(dengue)
    usable_pct = (n_obs / raw_rows) * 100 if raw_rows else 0.0
    info_lost_pct = 100.0 - usable_pct

    return {
        "Resolution": label,
        "Period alias": period_alias,
        "Usable observations": n_obs,
        "Usable % of raw dengue rows": round(usable_pct, 2),
        "Information lost": round(info_lost_pct, 2),
        "Districts with >=1 period": int((counts_per_district >= 1).sum()),
        "Districts with >=2 periods": int((counts_per_district >= 2).sum()),
        "Districts with >=5 periods": int((counts_per_district >= 5).sum()),
        "Districts with >=10 periods": int((counts_per_district >= 10).sum()),
        "Valid consecutive pairs": valid_pairs,
        "Mean gap (days)": round(float(gap_series.mean()), 2) if not gap_series.empty else None,
        "Median gap (days)": round(float(gap_series.median()), 2) if not gap_series.empty else None,
        "Max gap (days)": int(gap_series.max()) if not gap_series.empty else None,
        "Share gaps > 14 days": round(float((gap_series > 14).mean() * 100), 2) if not gap_series.empty else None,
        "Share gaps == 7 days": round(float((gap_series == 7).mean() * 100), 2) if not gap_series.empty else None,
        "Share gaps == 14 days": round(float((gap_series == 14).mean() * 100), 2) if not gap_series.empty else None,
        "District count": n_districts,
    }


def write_markdown(results: list[dict], dengue: pd.DataFrame) -> Path:
    comparison = pd.DataFrame(results)
    raw_rows = len(dengue)
    markdown = []
    markdown.append("# Forecasting Resolution Analysis\n")
    markdown.append("This analysis uses the exact `Disease == 'Dengue'` subset and evaluates whether the raw EpiClim records support a defensible weekly, biweekly, or monthly forecasting target. It does not create artificial zero-case periods or fabricate missing data.\n")
    markdown.append(f"- Exact dengue records used for temporal analysis: {raw_rows}\n")
    markdown.append(f"- Unique districts in the dengue subset: {dengue['district'].nunique()}\n")
    markdown.append(f"- Date range in the dengue subset: {dengue['date'].min().date()} to {dengue['date'].max().date()}\n")
    markdown.append("\n## Comparison table: Weekly vs Biweekly vs Monthly\n")
    markdown.append(comparison.to_string(index=False))
    markdown.append("\n")

    for item in results:
        markdown.append(f"## {item['Resolution']}\n")
        markdown.append(f"- Usable observations: {item['Usable observations']}\n")
        markdown.append(f"- Usable percentage of raw dengue rows: {item['Usable % of raw dengue rows']}%\n")
        markdown.append(f"- Information lost: {item['Information lost']}%\n")
        markdown.append(f"- Districts with at least 2 periods: {item['Districts with >=2 periods']}\n")
        markdown.append(f"- Districts with at least 5 periods: {item['Districts with >=5 periods']}\n")
        markdown.append(f"- Districts with at least 10 periods: {item['Districts with >=10 periods']}\n")
        markdown.append(f"- Valid consecutive forecasting pairs: {item['Valid consecutive pairs']}\n")
        markdown.append(f"- Mean gap between periods: {item['Mean gap (days)']} days\n")
        markdown.append(f"- Median gap between periods: {item['Median gap (days)']} days\n")
        markdown.append(f"- Max gap between periods: {item['Max gap (days)']} days\n")
        markdown.append(f"- Share of gaps > 14 days: {item['Share gaps > 14 days']}%\n")
        markdown.append(f"- Share of gaps == 7 days: {item['Share gaps == 7 days']}%\n")
        markdown.append(f"- Share of gaps == 14 days: {item['Share gaps == 14 days']}%\n")
        markdown.append("\n")

    markdown.append("## Interpretation\n")
    markdown.append("- Weekly data look more granular and align more closely with the original EpiClim reporting labels, but the raw series is not continuous: the district-level observations are sparse and irregular, and only a small fraction of gaps are exactly 7 days.\n")
    markdown.append("- Biweekly aggregation reduces temporal granularity and can increase the number of districts with fewer usable periods, but it may help smooth irregularity and preserve some temporal ordering. It is a defensible secondary option only if a true weekly target is not supportable.\n")
    markdown.append("- Monthly aggregation reduces noise and produces fewer observations, but it also discards substantial within-month information and can hide short-term outbreak dynamics. It is more defensible for coarse trend forecasting than for a short-horizon early warning system.\n")
    markdown.append("\n## Recommendation\n")
    markdown.append("The current EpiClim data do not support a uniformly continuous weekly surveillance series. A scientifically defensible approach is therefore to treat the raw observations as irregular district-level time points, choose the shortest defensible interval that preserves valid consecutive history, and only build forecasting pairs when the time gap is supported by the data.\n")
    markdown.append("In practice: start with a weekly target only for districts with enough valid temporal history; otherwise fall back to a coarser interval or restrict the model to districts with sufficient continuity. Do not create fake weekly rows or impute missing dengue periods.\n")
    out = OUT_DIR / "forecasting_resolution_analysis.md"
    out.write_text("".join(markdown), encoding="utf-8")
    return out


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    dengue = raw[raw["Disease"] == "Dengue"].copy()
    dengue["date"] = pd.to_datetime(
        dengue["year"].astype(str) + "-" + dengue["mon"].astype(str).str.zfill(2) + "-" + dengue["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    dengue = dengue.dropna(subset=["date"]).copy().reset_index(drop=True)

    results = [
        resolution_summary("Weekly", "W-MON", dengue),
        resolution_summary("Biweekly", "2W-MON", dengue),
        resolution_summary("Monthly", "M", dengue),
    ]
    dest = write_markdown(results, dengue)
    print(f"Forecasting resolution analysis written to: {dest}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
