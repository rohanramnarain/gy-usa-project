#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guyanese_acs_pipeline.py

One-stop script to fetch U.S. Census ACS data for Guyanese Americans and
generate simple charts: top occupations and health-insurance stats.

What it does
------------
1) Uses ACS Summary Table B04006 to get counts for "Guyanese ancestry".
2) Uses ACS PUMS API with `tabulate=weight (PWGTP)` to compute:
   - Top occupations (by SOC code, `SOCP`) for employed persons 16+.
   - Insurance coverage (HICOV) and the mix of coverage types (HINS1..HINS7,
     plus PRIVCOV/PUBCOV) for all ages (you can switch to 0–64 if desired).

Outputs
-------
- CSVs in ./outputs/
  * guyanese_top_occupations_<year>.csv
  * guyanese_insurance_mix_<year>.csv
  * guyanese_coverage_<year>.csv  (HICOV yes/no weighted counts & rates)
- PNG charts in ./outputs/
  * guyanese_top_occupations_<year>.png
  * guyanese_insurance_mix_<year>.png
  * guyanese_coverage_<year>.png

Usage
-----
$ pip install pandas requests matplotlib python-dotenv
$ export CENSUS_API_KEY=YOUR_KEY   # optional but recommended
$ python guyanese_acs_pipeline.py --year 2023 --survey acs1 --topn 15

Notes
-----
- We filter ancestry by ANC1P=370 (first ancestry entry) and ANC2P=370 (second).
  We tabulate each separately then sum them to capture anyone listing Guyanese
  in either ancestry slot.
- We use SOC (`SOCP`) for occupation categories to make labeling easier
  via the variables metadata endpoint.
- PUMS tabulations are weighted with PWGTP. (For variance/SEs, use replicate
  weights PWGTP1..PWGTP80; out of scope for this quick script.)

References
----------
- ACS B04006 "People Reporting Ancestry": Guyanese = B04006_045E
- ACS PUMS API docs & examples for `tabulate`: https://api.census.gov/data/<year>/acs/<survey>/pums/examples.html
- PUMS variables (OCCP/SOCP, HICOV, HINS*; ANC1P/ANC2P): variables endpoint
- ANC1P code list confirms 370 = "Guyanese".
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
import matplotlib.pyplot as plt

# Fallback SOC code -> title mapping (commonly appearing codes)
SOC_FALLBACK_TITLES = {
    "311121": "Home health aides",
    "311131": "Nursing assistants",
    "291141": "Registered nurses",
    "434051": "Customer service representatives",
    "1191XX": "Other managers (misc.)",
    "412031": "Retail salespersons",
    "339030": "Security guards & related",
    "439061": "Office clerks, general",
    "132011": "Accountants & auditors",
    "412010": "Cashiers",
    "252020": "Elementary & middle school teachers",
    "151252": "Software developers",
    "411011": "First-line supervisors of retail sales",
    "533030": "Driver/sales workers & truck drivers",
    "311122": "Personal care aides",
}


# ------------------------
# Config & helpers
# ------------------------

DEFAULT_YEAR = 2023          # last ACS 1-year with PUMS as of writing
DEFAULT_SURVEY = "acs1"      # "acs1" or "acs5" for PUMS; summary table uses matching base
TIMEOUT = 60

CENSUS_BASE = "https://api.census.gov/data"
SUMMARY_TABLE = "B04006_045E"  # Guyanese people reporting ancestry (estimate)

# Health insurance variables
HICOV_VAR = "HICOV"   # 1/2 yes/no
HINS_VARS = ["HINS1", "HINS2", "HINS3", "HINS4", "HINS5", "HINS6", "HINS7"]  # employer, direct, Medicare, Medicaid, TRICARE, VA, IHS
DERIVED_VARS = ["PRIVCOV", "PUBCOV"]  # derived indicators; not mutually exclusive with others

def get_api_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY")

def ensure_outputs_dir() -> pathlib.Path:
    outdir = pathlib.Path("outputs")
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir

def _api_get(url: str, params: Dict[str,str]) -> List[List[str]]:
    """GET helper with API key passthrough and basic error handling."""
    key = get_api_key()
    if key:
        params = {**params, "key": key}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def fetch_variable_labels(year: int, survey: str, var: str) -> Dict[str, str]:
    """
    Fetch value labels for a PUMS variable (e.g., SOCP, OCCP, HICOV, HINS1) from the
    variables metadata endpoint. Returns {code -> label}. If unavailable, returns {}.
    """
    # Example: https://api.census.gov/data/2023/acs/acs1/pums/variables/SOCP.json
    url = f"{CENSUS_BASE}/{year}/acs/{survey}/pums/variables/{var}.json"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        values = data.get("values") or {}
        # values is expected to be {code: {"label": "label text", ...}, ...} but
        # sometimes it's {code: "Label"}; handle both.
        out = {}
        for k, v in values.items():
            if isinstance(v, dict):
                out[k] = v.get("label", str(v))
            else:
                out[k] = str(v)
        return out
    except Exception:
        return {}

def fetch_ancestry_counts_by_state(year: int, base_survey: str) -> pd.DataFrame:
    """
    Pull B04006 Guyanese ancestry counts by state from ACS summary tables.
    base_survey: 'acs1' or 'acs5'
    """
    url = f"{CENSUS_BASE}/{year}/acs/{base_survey}"
    params = {"get": f"NAME,{SUMMARY_TABLE}", "for": "state:*"}
    rows = _api_get(url, params)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.rename(columns={SUMMARY_TABLE: "guyanese_ancestry"}, inplace=True)
    # Types
    df["guyanese_ancestry"] = pd.to_numeric(df["guyanese_ancestry"], errors="coerce").fillna(0).astype(int)
    return df[["NAME", "state", "guyanese_ancestry"]].sort_values("guyanese_ancestry", ascending=False)

def build_pums_tabulate_params(row_var: str, filters: Dict[str, str]) -> Dict[str, str]:
    """
    Build params for a PUMS tabulation query.
    - row_var: e.g., "SOCP" or "HICOV" or "HINS1"
    - filters example: {"AGEP": "16:99", "ESR": "1,2", "ANC1P": "370"}
    """
    # 'row+VAR' must be a key with empty value per API examples
    params = {"tabulate": "weight (PWGTP)", f"row+{row_var}": ""}
    params.update(filters)
    return params

def pums_tabulate(year: int, survey: str, row_var: str, filters: Dict[str, str]) -> pd.DataFrame:
    """
    Run a single PUMS tabulation, returning DataFrame with columns [row_var, 'weight (PWGTP)'].
    """
    url = f"{CENSUS_BASE}/{year}/acs/{survey}/pums"
    params = build_pums_tabulate_params(row_var, filters)
    rows = _api_get(url, params)
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    # Identify the weight column (exact header is 'weight (PWGTP)')
    # The PUMS tabulate API usually returns the weighted result in a column named 'tabulate'.
    # Fall back to any weight-like column if that ever changes.
    if "tabulate" in df.columns:
        wcol = "tabulate"
    else:
        weight_col = [c for c in df.columns if "weight" in c.lower() or "pwg" in c.lower()]
        if weight_col:
            wcol = weight_col[0]
        else:
            # As a last resort, pick the non-row_var numeric column
            candidates = [c for c in df.columns if c != row_var]
            if not candidates:
                raise RuntimeError(f"Could not find weight/estimate column in PUMS response for {row_var}. Columns: {df.columns.tolist()}")
            wcol = candidates[-1]
    df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
    # Normalize column names
    df.rename(columns={row_var: "code", wcol: "weighted_count"}, inplace=True)
    # Some APIs return extra cols; keep just what we need
    keep = ["code", "weighted_count"]
    df = df[keep].groupby("code", as_index=False)["weighted_count"].sum()
    return df

def combine_ancestry_slots(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Sum weighted counts by 'code' across ancestry slots (ANC1P and ANC2P)."""
    return (pd.concat([df1, df2], ignore_index=True)
              .groupby("code", as_index=False)["weighted_count"].sum()
              .sort_values("weighted_count", ascending=False))

def add_labels(df: pd.DataFrame, labels: Dict[str, str], code_col: str = "code", label_col: str = "label") -> pd.DataFrame:
    """Attach human-readable labels to code column via mapping dict."""
    df[label_col] = df[code_col].map(labels).fillna(df[code_col])
    return df

def top_occupations_guyanese(year: int, survey: str, topn: int = 15) -> pd.DataFrame:
    """
    Compute top occupations (by SOC code SOCP) for Guyanese ancestry, employed (ESR 1 or 2), ages 16+.
    Returns sorted DataFrame with columns [SOCP code, label, weighted_count, share].
    """
    base_filters = {"AGEP": "16:99", "ESR": "1,2"}  # civilian employed at work or with a job but not at work
    # Tabulate separately for ANC1P and ANC2P and sum
    df_anc1 = pums_tabulate(year, survey, row_var="SOCP", filters={**base_filters, "ANC1P": "370"})
    df_anc2 = pums_tabulate(year, survey, row_var="SOCP", filters={**base_filters, "ANC2P": "370"})
    combined = combine_ancestry_slots(df_anc1, df_anc2)
    # Labels for SOCP codes
    labels = fetch_variable_labels(year, survey, "SOCP")
    combined = add_labels(combined, labels)
    # Apply fallback titles when API labels are missing or equal to code
    combined['label'] = combined.apply(
        lambda r: (SOC_FALLBACK_TITLES.get(str(r['code']).strip(), r['label'])
                    if (pd.isna(r['label']) or str(r['label']).strip()==str(r['code']).strip()) else r['label']), axis=1)
    combined["share"] = combined["weighted_count"] / combined["weighted_count"].sum()
    return combined.sort_values("weighted_count", ascending=False).head(topn).reset_index(drop=True)


def insurance_coverage_guyanese(year: int, survey: str, ages: str = "0:99") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute health insurance coverage for Guyanese ancestry.
    - HICOV distribution (Yes/No)
    - Coverage mix for HINS1..HINS7 plus PRIVCOV/PUBCOV (note: categories overlap)
    Returns (hicov_df, mix_df)
    """
    base_filters = {"AGEP": ages}
    # HICOV Yes/No
    df_h1 = pums_tabulate(year, survey, row_var=HICOV_VAR, filters={**base_filters, "ANC1P": "370"})
    df_h2 = pums_tabulate(year, survey, row_var=HICOV_VAR, filters={**base_filters, "ANC2P": "370"})
    hicov = combine_ancestry_slots(df_h1, df_h2)

    # Try fetching labels from API, else use a sensible fallback
    hicov_labels = fetch_variable_labels(year, survey, HICOV_VAR)
    # Standardize code to str for mapping
    hicov["code"] = hicov["code"].astype(str).str.strip()
    hicov = add_labels(hicov, hicov_labels)
    # Fallback mapping if API didn't return nice labels (common: shows "1"/"2")
    fallback_map = {"1": "With coverage", "2": "No coverage", "1.0": "With coverage", "2.0": "No coverage"}
    hicov["label"] = hicov.apply(
        lambda r: fallback_map.get(r["code"], r["label"] if r["label"] != r["code"] else r["label"]),
        axis=1
    )
    total = hicov["weighted_count"].sum()
    hicov["rate"] = hicov["weighted_count"] / total

    # Coverage mix (each variable is 1/2; we want % with value==1)
    mix_records = []
    for var in HINS_VARS + DERIVED_VARS:
        d1 = pums_tabulate(year, survey, row_var=var, filters={**base_filters, "ANC1P": "370"})
        d2 = pums_tabulate(year, survey, row_var=var, filters={**base_filters, "ANC2P": "370"})
        d = combine_ancestry_slots(d1, d2)
        d["code"] = d["code"].astype(str).str.strip()
        yes_row = d[d["code"] == "1"]
        yes_count = float(yes_row["weighted_count"].sum()) if not yes_row.empty else 0.0
        mix_records.append({"variable": var, "weighted_count": yes_count})

    mix_df = pd.DataFrame(mix_records)
    mix_total = total  # denominator = total persons in hicov table with ages filter
    mix_df["percent"] = mix_df["weighted_count"] / mix_total
    # Attach friendly names for variables
    friendly = {
        "HINS1": "Employer/Union",
        "HINS2": "Direct-Purchase",
        "HINS3": "Medicare",
        "HINS4": "Medicaid",
        "HINS5": "TRICARE",
        "HINS6": "VA",
        "HINS7": "Indian Health Service",
        "PRIVCOV": "Any Private",
        "PUBCOV": "Any Public",
    }
    mix_df["label"] = mix_df["variable"].map(friendly).fillna(mix_df["variable"])
    mix_df = mix_df.sort_values("percent", ascending=False).reset_index(drop=True)
    return hicov, mix_df



def plot_top_occupations(df: pd.DataFrame, outpath: pathlib.Path, year: int, topn: int):
    # Larger figure to avoid title overflow and improve readability
    plt.figure(figsize=(14, 8))
    ylabels = df["label"].tolist()  # human-readable SOC titles
    values = df["weighted_count"].tolist()
    plt.barh(range(len(df)), values)  # default colors per instructions
    plt.yticks(range(len(df)), ylabels)
    plt.gca().invert_yaxis()
    plt.xlabel("Estimated number of employed persons (weighted)")
    plt.title(f"Top {topn} Occupations — Guyanese ancestry, ACS {year} PUMS", pad=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()



def plot_hicov(hicov: pd.DataFrame, outpath: pathlib.Path, year: int, ages: str):
    # Make labels human-friendly and figure larger to prevent title overflow
    labels = hicov["label"].tolist()
    # Fallback in case labels still look like codes
    labels = ["With coverage" if str(x).strip() in ("1","1.0") else ("No coverage" if str(x).strip() in ("2","2.0") else str(x)) for x in labels]

    values = (hicov["rate"] * 100).round(1).tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_ylabel("Percent of people (%)")
    ax.set_title(f"Health Insurance Coverage — Guyanese ancestry, ACS {year} (Ages {ages})", pad=14)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_mix(mix_df: pd.DataFrame, outpath: pathlib.Path, year: int, ages: str):
    plt.figure(figsize=(10, 6))
    labels = mix_df["label"].tolist()
    values = (mix_df["percent"] * 100).round(1).tolist()
    plt.barh(range(len(mix_df)), values)
    plt.yticks(range(len(mix_df)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Share of people with this type of coverage (%)\n(Note: categories overlap; totals may exceed 100%)")
    plt.title(f"Coverage Mix — Guyanese ancestry, ACS {year} (Ages {ages})")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ------------------------
# Main workflow
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="ACS data pipeline for Guyanese Americans: jobs & healthcare")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="ACS year (e.g., 2023)")
    parser.add_argument("--survey", type=str, default=DEFAULT_SURVEY, choices=["acs1", "acs5"], help="ACS survey")
    parser.add_argument("--topn", type=int, default=15, help="Top N occupations to plot")
    parser.add_argument("--ages", type=str, default="0:99", help="Age range for healthcare stats (e.g., '0:64' or '0:99')")
    args = parser.parse_args()

    outdir = ensure_outputs_dir()

    # 1) Summary table counts by state (optional context)
    try:
        states_df = fetch_ancestry_counts_by_state(args.year, args.survey)
        states_csv = outdir / f"guyanese_ancestry_by_state_{args.year}.csv"
        states_df.to_csv(states_csv, index=False)
        print(f"[Saved] {states_csv}")
    except Exception as e:
        print(f"[Warn] Failed to fetch ancestry by state: {e}")

    # 2) Top occupations (SOC)
    try:
        top_occ = top_occupations_guyanese(args.year, args.survey, topn=args.topn)
        occ_csv = outdir / f"guyanese_top_occupations_{args.year}.csv"
        top_occ.to_csv(occ_csv, index=False)
        print(f"[Saved] {occ_csv}")
        occ_png = outdir / f"guyanese_top_occupations_{args.year}.png"
        plot_top_occupations(top_occ, occ_png, args.year, args.topn)
        print(f"[Saved] {occ_png}")
    except Exception as e:
        print(f"[Error] Failed to compute top occupations: {e}")

    # 3) Healthcare coverage
    try:
        hicov_df, mix_df = insurance_coverage_guyanese(args.year, args.survey, ages=args.ages)
        cov_csv = outdir / f"guyanese_coverage_{args.year}.csv"
        hicov_df.to_csv(cov_csv, index=False)
        print(f"[Saved] {cov_csv}")
        mix_csv = outdir / f"guyanese_insurance_mix_{args.year}.csv"
        mix_df.to_csv(mix_csv, index=False)
        print(f"[Saved] {mix_csv}")

        cov_png = outdir / f"guyanese_coverage_{args.year}.png"
        plot_hicov(hicov_df, cov_png, args.year, args.ages)
        print(f"[Saved] {cov_png}")

        mix_png = outdir / f"guyanese_insurance_mix_{args.year}.png"
        plot_mix(mix_df, mix_png, args.year, args.ages)
        print(f"[Saved] {mix_png}")
    except Exception as e:
        print(f"[Error] Failed to compute healthcare stats: {e}")

if __name__ == "__main__":
    main()
