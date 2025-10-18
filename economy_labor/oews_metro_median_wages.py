# oews_metro_median_wages.py
# Usage: python oews_metro_median_wages.py --msas 35620 33100 16980 --year 2024 --out data/oews
# Example MSAs: 35620 (NY-Newark-Jersey City), 33100 (Miami-Fort Lauderdale-West Palm Beach), 16980 (Chicago-Naperville-Elgin)

import argparse
import pandas as pd
from pathlib import Path

BASE = "https://download.bls.gov/pub/time.series/oe/"

def read_tsv(name, dtype=str):
    return pd.read_csv(f"{BASE}{name}", sep="\t", dtype=dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msas", nargs="+", required=True, help="CBSA/area codes (e.g., 35620 33100 16980)")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="data/oews")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata and data (tab-delimited text)
    series = read_tsv("oe.series")
    data = read_tsv("oe.data.1.AllData")
    dt = read_tsv("oe.datatype")
    area = read_tsv("oe.area")
    occ = read_tsv("oe.occupation")

    # Keep series with desired areas + occupations (SOC), and year
    # In series, fields include: series_id, area_code, occ_code, datatype_code, industry_code, seasonal
    sr = series[series["area_code"].isin(args.msas)].copy()

    # Identify datatype codes for median wages
    # dt contains code + label; select annual & hourly medians
    med_dt = dt[dt["datatype_text"].str.contains("Median", case=False, na=False)]
    med_codes = set(med_dt["datatype_code"])

    # Join data to series
    dat = data[data["year"] == str(args.year)].merge(sr, on="series_id", how="inner")

    # Keep only median measures
    dat = dat[dat["datatype_code"].isin(med_codes)].copy()

    # Bring in SOC titles + area names
    dat = dat.merge(occ[["occupation_code","occupation_text"]], left_on="occ_code", right_on="occupation_code", how="left")
    dat = dat.merge(area[["area_code","area_text"]], on="area_code", how="left")
    dat = dat.merge(dt[["datatype_code","datatype_text"]], on="datatype_code", how="left")

    # Tidy: one row per (area, soc), keep value and whether hourly/annual median
    # value is numeric but comes as string
    dat["value"] = pd.to_numeric(dat["value"], errors="coerce")

    # Prefer annual median if available; otherwise keep hourly median
    # Create a single 'median_wage' and 'wage_unit' column per SOC in each area
    def pick_best(group):
        # If both annual and hourly exist, prefer annual
        annual = group[group["datatype_text"].str.contains("Annual", case=False, na=False)]
        if len(annual):
            row = annual.iloc[0]
            return pd.Series({
                "soc_code": row["occ_code"],
                "soc_title": row["occupation_text"],
                "area_code": row["area_code"],
                "area_title": row["area_text"],
                "year": args.year,
                "median_wage": row["value"],
                "wage_unit": "annual_dollars"
            })
        # else use hourly
        row = group.iloc[0]
        return pd.Series({
            "soc_code": row["occ_code"],
            "soc_title": row["occupation_text"],
            "area_code": row["area_code"],
            "area_title": row["area_text"],
            "year": args.year,
            "median_wage": row["value"],
            "wage_unit": "hourly_dollars"
        })

    tidy = dat.groupby(["area_code","occ_code"]).apply(pick_best).reset_index(drop=True)

    # Output
    tidy.to_csv(outdir / f"oews_metro_median_wages_{args.year}.csv", index=False)

    # Also emit a minimal crosswalk note
    with open(outdir / "README_oews.txt", "w") as f:
        f.write(
            "Source: BLS OEWS 'oe' time-series text files (oe.series, oe.data.1.AllData, oe.datatype, oe.area).\n"
            "Rows are per (MSA/CBSA x SOC). 'soc_code' is the join key for merging with occupation distributions.\n"
            "If your CPS distributions use OCC2010 or other harmonized codes, join via an external crosswalk.\n"
        )

if __name__ == "__main__":
    main()
