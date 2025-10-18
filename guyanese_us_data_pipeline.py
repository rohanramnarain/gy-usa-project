#!/usr/bin/env python3
"""
Guyanese-US-Data-Pipeline
-------------------------
One-file Python codebase to:
  1) Pull ACS 5-year summary tables for Guyanese ancestry and Guyana-born by geography
  2) Download & filter ACS 1-year PUMS microdata to isolate people with Guyanese ancestry
  3) Produce weighted summaries by state, race, and nativity

Outputs (under ./data/outputs):
  - acs5_state_ancestry_2023.csv                (state-level counts; single + multiple ancestry)
  - acs5_county_ancestry_2023_ny.csv            (county-level counts for NY example)
  - acs5_state_birthplace_guyana_2023.csv       (state-level foreign-born from Guyana)
  - pums_2023_guyanese_persons.csv              (all PUMS persons w/ Guyanese ancestry; selected columns)
  - pums_2023_state_weighted_counts.csv         (weighted totals by state)
  - pums_2023_race_weighted_counts.csv          (weighted totals by detailed race code among Guyanese)
  - pums_2023_nativity_weighted_counts.csv      (weighted native vs foreign-born among Guyanese)

Requirements:
  Python 3.10+
  pip install -r requirements.txt  (see REQUIREMENTS string below)

Env (optional but recommended for higher ACS API limits):
  export CENSUS_API_KEY=YOUR_KEY

Run examples:
  python guyanese_pipeline.py --all
  # or equivalently:
  python guyanese_pipeline.py all
  python guyanese_pipeline.py acs --geo state
  python guyanese_pipeline.py acs --geo county --state-fips 36
  python guyanese_pipeline.py pums --year 2023
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Iterable, List, Dict

import pandas as pd
import requests

# -----------------
# Config & constants
# -----------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_YEAR_SUMMARY = 2023   # ACS 5-year detailed tables
DEFAULT_YEAR_PUMS = 2023      # ACS 1-year PUMS microdata

# ACS detailed tables (5-year) variable IDs for Guyanese ancestry and Guyana birthplace
# B04004: People Reporting SINGLE Ancestry — Guyanese
VAR_B04004_GUYANESE_SINGLE = "B04004_045E"
# B04005: People Reporting MULTIPLE Ancestry — Guyanese
VAR_B04005_GUYANESE_MULTI = "B04005_045E"
# B05006: Place of Birth for the Foreign-Born Pop — Guyana
VAR_B05006_GUYANA = "B05006_171E"

# PUMS ancestry code for Guyanese (ANC1P / ANC2P)
PUMS_ANCESTRY_CODE_GUYANESE = 370

# PUMS columns we’ll keep in the filtered output
PUMS_KEEP_COLS = [
    "STATE",     # State code (string-like)
    "PUMA",      # Public Use Microdata Area
    "PWGTP",     # Person weight
    "AGEP",      # Age
    "SEX",       # Sex
    "RAC1P",     # Detailed race
    "HISP",      # Hispanic origin
    "NATIVITY",  # 1=native, 2=foreign-born
    "ANC1P",     # Ancestry (first)
    "ANC2P",     # Ancestry (second)
    "SCHL",      # Education
    "ESR",       # Employment status recode
    "PINCP",     # Total personal income
]

# Simple state FIPS to name mapping (covers 50 states + DC)
STATE_FIPS_TO_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York", "37": "North Carolina",
    "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
    "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota", "47": "Tennessee",
    "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
}

# Requirements file content (for convenience)
REQUIREMENTS = """
pandas>=2.0
requests>=2.31
""".strip()

# -----------------
# Helpers
# -----------------

def save_requirements_txt():
    req_path = Path("requirements.txt")
    if not req_path.exists():
        req_path.write_text(REQUIREMENTS, encoding="utf-8")
        print(f"[setup] Wrote {req_path}")


def census_get(url: str, params: Dict[str, str | int | None]) -> List[List[str]]:
    """Generic GET to Census API; returns 2D array (rows) from JSON.
    Adds key if present in environment.
    """
    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params = {**params, "key": api_key}
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def _to_df(rows: List[List[str]]) -> pd.DataFrame:
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


# -----------------
# ACS 5-year detailed tables (summary)
# -----------------

def fetch_acs5_state_ancestry(year: int = DEFAULT_YEAR_SUMMARY) -> pd.DataFrame:
    """Fetch state-level counts for Guyanese ancestry (single & multiple)."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_str = f"NAME,{VAR_B04004_GUYANESE_SINGLE},{VAR_B04005_GUYANESE_MULTI}"
    rows = census_get(base, {"get": vars_str, "for": "state:*"})
    df = _to_df(rows)
    for c in [VAR_B04004_GUYANESE_SINGLE, VAR_B04005_GUYANESE_MULTI]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["guyanese_total_ancestry"] = df[VAR_B04004_GUYANESE_SINGLE] + df[VAR_B04005_GUYANESE_MULTI]
    df.rename(columns={"state": "state_fips"}, inplace=True)
    df["state_name"] = df["state_fips"].map(lambda x: STATE_FIPS_TO_NAME.get(x.zfill(2), x))
    out = OUT_DIR / f"acs5_state_ancestry_{year}.csv"
    df.to_csv(out, index=False)
    print(f"[ok] Wrote {out}")
    return df


def fetch_acs5_county_ancestry_for_state(state_fips: str, year: int = DEFAULT_YEAR_SUMMARY) -> pd.DataFrame:
    """Fetch county-level counts for Guyanese ancestry for a single state (e.g., NY=36)."""
    state_fips = state_fips.zfill(2)
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_str = f"NAME,{VAR_B04004_GUYANESE_SINGLE},{VAR_B04005_GUYANESE_MULTI}"
    rows = census_get(base, {"get": vars_str, "for": "county:*", "in": f"state:{state_fips}"})
    df = _to_df(rows)
    for c in [VAR_B04004_GUYANESE_SINGLE, VAR_B04005_GUYANESE_MULTI]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["guyanese_total_ancestry"] = df[VAR_B04004_GUYANESE_SINGLE] + df[VAR_B04005_GUYANESE_MULTI]
    df.rename(columns={"state": "state_fips", "county": "county_fips"}, inplace=True)
    out = OUT_DIR / f"acs5_county_ancestry_{year}_{state_fips}.csv"
    df.to_csv(out, index=False)
    print(f"[ok] Wrote {out}")
    return df


def fetch_acs5_state_birthplace_guyana(year: int = DEFAULT_YEAR_SUMMARY) -> pd.DataFrame:
    """Fetch state-level counts of foreign-born whose place of birth is Guyana (B05006_171E)."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    vars_str = f"NAME,{VAR_B05006_GUYANA}"
    rows = census_get(base, {"get": vars_str, "for": "state:*"})
    df = _to_df(rows)
    df[VAR_B05006_GUYANA] = pd.to_numeric(df[VAR_B05006_GUYANA], errors="coerce")
    df.rename(columns={"state": "state_fips"}, inplace=True)
    df["state_name"] = df["state_fips"].map(lambda x: STATE_FIPS_TO_NAME.get(x.zfill(2), x))
    out = OUT_DIR / f"acs5_state_birthplace_guyana_{year}.csv"
    df.to_csv(out, index=False)
    print(f"[ok] Wrote {out}")
    return df


# -----------------
# PUMS (1-year) download & filter
# -----------------

PUMS_1Y_PERSONS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip"


def _iter_pums_csv_members(zip_path: Path) -> Iterable[io.BytesIO]:
    """Yield file-like handles for each person CSV inside the PUMS US zip.
    The ZIP typically contains two person files (e.g., psam_pusa.csv, psam_pusb.csv).
    """
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            name_lower = name.lower()
            if name_lower.endswith('.csv') and ('psam_pus' in name_lower or 'psam_pusa' in name_lower or 'psam_pusb' in name_lower):
                with zf.open(name) as f:
                    yield io.BytesIO(f.read())


def download_pums_1y_persons(dest_zip: Path) -> Path:
    if dest_zip.exists():
        print(f"[skip] Found {dest_zip}")
        return dest_zip
    print(f"[dl] {PUMS_1Y_PERSONS_URL}")
    with requests.get(PUMS_1Y_PERSONS_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest_zip, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    print(f"[ok] Wrote {dest_zip}")
    return dest_zip


def filter_pums_guyanese(zip_path: Path) -> pd.DataFrame:
    """Read PUMS person CSVs from the ZIP, filter for Guyanese ancestry (ANC1P==370 or ANC2P==370).
    Return a concatenated DataFrame of selected columns.
    """
    dfs: List[pd.DataFrame] = []
    for fh in _iter_pums_csv_members(zip_path):
        # Read in chunks to keep memory in check
        for chunk in pd.read_csv(fh, dtype=str, chunksize=250_000):
            # Ensure needed columns exist; some chunk readers may infer mixed types otherwise
            for col in ["ANC1P", "ANC2P", "PWGTP"]:
                if col not in chunk.columns:
                    raise RuntimeError(f"PUMS file missing expected column: {col}")
            # Filter
            sel = chunk[(chunk["ANC1P"].astype('Int64') == PUMS_ANCESTRY_CODE_GUYANESE) |
                        (chunk["ANC2P"].astype('Int64') == PUMS_ANCESTRY_CODE_GUYANESE)]
            if sel.empty:
                continue
            # Keep relevant cols, cast numeric where appropriate
            keep = [c for c in PUMS_KEEP_COLS if c in sel.columns]
            sel = sel[keep].copy()
            # Numeric casts
            for numc in ["PWGTP", "AGEP", "RAC1P", "HISP", "NATIVITY", "SCHL", "ESR", "PINCP"]:
                if numc in sel.columns:
                    sel[numc] = pd.to_numeric(sel[numc], errors="coerce")
            sel["STATE"] = sel["STATE"].astype(str).str.zfill(2)
            dfs.append(sel)
    if not dfs:
        raise RuntimeError("No Guyanese records found in PUMS — check inputs/year.")
    out_df = pd.concat(dfs, ignore_index=True)
    out = OUT_DIR / f"pums_{DEFAULT_YEAR_PUMS}_guyanese_persons.csv"
    out_df.to_csv(out, index=False)
    print(f"[ok] Wrote {out} (rows={len(out_df):,})")
    return out_df


def weighted_group_sum(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Weighted count (sum of PWGTP) grouped by `by` columns."""
    if "PWGTP" not in df.columns:
        raise ValueError("PWGTP weight is required for weighted totals")
    gp = df.groupby(by, dropna=False, as_index=False)["PWGTP"].sum()
    gp.rename(columns={"PWGTP": "weighted_count"}, inplace=True)
    return gp


def summarize_pums(df: pd.DataFrame) -> None:
    # By state
    st = weighted_group_sum(df, ["STATE"])
    st["state_name"] = st["STATE"].map(lambda x: STATE_FIPS_TO_NAME.get(str(x).zfill(2), str(x)))
    st_out = OUT_DIR / f"pums_{DEFAULT_YEAR_PUMS}_state_weighted_counts.csv"
    st.to_csv(st_out, index=False)
    print(f"[ok] Wrote {st_out}")

    # By detailed race among Guyanese
    if "RAC1P" in df.columns:
        race = weighted_group_sum(df, ["RAC1P"]).sort_values("weighted_count", ascending=False)
        race_out = OUT_DIR / f"pums_{DEFAULT_YEAR_PUMS}_race_weighted_counts.csv"
        race.to_csv(race_out, index=False)
        print(f"[ok] Wrote {race_out}")

    # By nativity (1=native-born, 2=foreign-born)
    if "NATIVITY" in df.columns:
        nat = weighted_group_sum(df, ["NATIVITY"]).sort_values("weighted_count", ascending=False)
        nat_out = OUT_DIR / f"pums_{DEFAULT_YEAR_PUMS}_nativity_weighted_counts.csv"
        nat.to_csv(nat_out, index=False)
        print(f"[ok] Wrote {nat_out}")


# -----------------
# CLI
# -----------------

def run_acs(args: argparse.Namespace) -> None:
    year = args.year or DEFAULT_YEAR_SUMMARY
    fetch_acs5_state_ancestry(year)
    if args.geo == "county":
        if not args.state_fips:
            print("--state-fips is required for county pulls (e.g., 36 for NY)")
            sys.exit(2)
        fetch_acs5_county_ancestry_for_state(args.state_fips, year)
    fetch_acs5_state_birthplace_guyana(year)


def run_pums(args: argparse.Namespace) -> None:
    year = args.year or DEFAULT_YEAR_PUMS
    if year != 2023:
        print("[warn] This script pins the PUMS download URL to 2023; update PUMS_1Y_PERSONS_URL for other years.")
    zip_path = RAW_DIR / f"pums_{year}_1y_persons.zip"
    download_pums_1y_persons(zip_path)
    df = filter_pums_guyanese(zip_path)
    summarize_pums(df)


def run_all(args: argparse.Namespace) -> None:
    run_acs(argparse.Namespace(year=DEFAULT_YEAR_SUMMARY, geo="state", state_fips=None))
    run_pums(argparse.Namespace(year=DEFAULT_YEAR_PUMS))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Guyanese in US data pipeline (ACS summary + PUMS)")
    p.add_argument("--all", dest="run_all_flag", action="store_true", help="Run everything end-to-end")
    sub = p.add_subparsers(dest="cmd", required=False)

    # acs
    p_acs = sub.add_parser("acs", help="Pull ACS 5-year summary tables for ancestry & birthplace")
    p_acs.add_argument("--year", type=int, default=DEFAULT_YEAR_SUMMARY, help="ACS 5-year year (default 2023)")
    p_acs.add_argument("--geo", choices=["state", "county"], default="state", help="Geography level to pull (state or county)")
    p_acs.add_argument("--state-fips", type=str, help="If --geo county, which state (e.g., 36 for NY)")
    p_acs.set_defaults(func=run_acs)

    # pums
    p_pums = sub.add_parser("pums", help="Download + filter ACS 1-year PUMS for Guyanese ancestry")
    p_pums.add_argument("--year", type=int, default=DEFAULT_YEAR_PUMS, help="PUMS year (default 2023)")
    p_pums.set_defaults(func=run_pums)

    # all
    p_all = sub.add_parser("all", help="Run everything end-to-end")
    p_all.set_defaults(func=run_all)

    return p


def main(argv: List[str] | None = None) -> None:
    save_requirements_txt()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "run_all_flag", False):
        run_all(args)
        return
    if not getattr(args, "cmd", None):
        # default to --all if no subcommand provided
        run_all(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
