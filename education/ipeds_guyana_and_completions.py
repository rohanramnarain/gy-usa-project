
#!/usr/bin/env python3
"""
IPEDs + Open Doors helper

What this script does
---------------------
1) Guyana international students by institution & state (across years)
   - NOTE: IPEDS does **not** publish country-of-citizenship counts.
     Use IIE Open Doors "places of origin" data for country-level counts.
     Pass a CSV from Open Doors filtered to Guyana (or a full file the script can filter).

2) IPEDS Completions by field (CIP) where available
   - Reads IPEDS Completions (C) multiple-record files for one or more years.
   - Optionally joins with IPEDS Directory (HD) files to add institution name and state.

Outputs
-------
- outputs/guyana_by_institution.csv
- outputs/guyana_by_state.csv
- outputs/ipeds_completions_by_field.csv
- outputs/state_summary.csv (aggregated Guyana counts by state & year)

Usage
-----
# Guyana counts from Open Doors
python ipeds_guyana_and_completions.py \
  --open-doors-csv path/to/opendoors_places_of_origin.csv \
  --country "Guyana"

# IPEDS completions by field (CIP) + join HD
python ipeds_guyana_and_completions.py \
  --ipeds-completions-dir /path/to/ipeds/C/ \
  --ipeds-hd-dir /path/to/ipeds/HD/

# Do both in one go
python ipeds_guyana_and_completions.py \
  --open-doors-csv path/to/opendoors.csv --country Guyana \
  --ipeds-completions-dir /path/to/ipeds/C --ipeds-hd-dir /path/to/ipeds/HD

Assumptions & notes
-------------------
- Open Doors column names vary by file. Use --od-country-field, --od-year-field, etc. to map.
- IPEDS Completions (C) files typically include columns like:
  UNITID, CIPCODE, MAJORNUM, AWLEVEL, CTOTALT (and possibly CTOTALM, CTOTALW).
- IPEDS Directory (HD) files add INSTNM (name) and STABBR (state).
- The script tries to infer the year from the filename when needed.
- Outputs are *tidy* CSVs with explicit year, ids, and values.
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd


# ------------------------
# Helpers
# ------------------------

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_year_from_filename(fp: Path) -> Optional[int]:
    """
    Try to infer a 4-digit year from a filename like 'C2023_A.csv' or 'hd2022.csv'.
    """
    m = re.search(r'(19|20)\d{2}', fp.name)
    return int(m.group(0)) if m else None


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ------------------------
# Open Doors (Guyana) part
# ------------------------

def load_opendoors(
    csv_path: Path,
    country_value: str,
    country_field: str,
    institution_field: str,
    state_field: str,
    year_field: str,
    count_field: str
) -> pd.DataFrame:
    """
    Load an Open Doors CSV (international students by place of origin) and filter to a single country.
    Returns tidy rows: year, institution, state, count
    """
    df = pd.read_csv(csv_path, dtype=str)
    df = normalize_cols(df)

    # force numeric for counts when possible
    if count_field.lower() in df.columns:
        df[count_field.lower()] = pd.to_numeric(df[count_field.lower()], errors='coerce').fillna(0).astype(int)

    # filter
    mask = df[country_field.lower()].str.strip().str.casefold() == country_value.strip().casefold()
    df = df.loc[mask].copy()

    # Select & rename
    keep = {
        year_field.lower(): 'year',
        institution_field.lower(): 'instnm',
        state_field.lower(): 'stabbr',
        count_field.lower(): 'guyana_students'
    }
    # some files may not include state; allow missing
    for k in list(keep.keys()):
        if k not in df.columns:
            # create missing columns as NA
            df[k] = pd.NA

    tidy = df[list(keep.keys())].rename(columns=keep)
    # coerce year to int if possible
    with pd.option_context('mode.chained_assignment', None):
        tidy['year'] = pd.to_numeric(tidy['year'], errors='coerce').astype('Int64')

    # Drop rows missing institution or count
    tidy = tidy.dropna(subset=['instnm']).copy()
    return tidy


def summarize_opendoors_state(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Guyana counts by state & year.
    """
    out = (tidy_df
           .dropna(subset=['stabbr'])
           .groupby(['year', 'stabbr'], as_index=False)['guyana_students']
           .sum()
           .sort_values(['year', 'stabbr']))
    return out


# ------------------------
# IPEDS Completions part
# ------------------------

def read_ipeds_csv(fp: Path) -> pd.DataFrame:
    """
    Read an IPEDS CSV, handling encoding/low-memory issues gracefully.
    """
    try:
        return pd.read_csv(fp, dtype=str, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(fp, dtype=str, encoding='latin-1', low_memory=False)


def load_directory_lookup(hd_dir: Path) -> pd.DataFrame:
    """
    Load the most complete set of HD files from a directory and return a UNION with columns:
    year, unitid, instnm, stabbr
    """
    rows = []
    for fp in sorted(hd_dir.glob('*.csv')):
        df = read_ipeds_csv(fp)
        df = normalize_cols(df)
        year = infer_year_from_filename(fp)
        # required fields (some HD files use uppercase, we normalized)
        needed = ['unitid', 'instnm']
        for col in needed:
            if col not in df.columns:
                continue
        # state sometimes is 'stabbr' or 'stabbr' present; occasionally 'state_abbr' in derived files
        state_col = 'stabbr' if 'stabbr' in df.columns else ('state_abbr' if 'state_abbr' in df.columns else None)
        if state_col is None:
            df['stabbr'] = pd.NA
            state_col = 'stabbr'

        df_out = pd.DataFrame({
            'year': [year] * len(df),
            'unitid': df['unitid'],
            'instnm': df['instnm'],
            'stabbr': df[state_col]
        })
        rows.append(df_out)
    if not rows:
        return pd.DataFrame(columns=['year', 'unitid', 'instnm', 'stabbr'])
    out = pd.concat(rows, ignore_index=True)
    return out


def load_completions(completions_dir: Path) -> pd.DataFrame:
    """
    Read all Completions multiple-record CSVs in a directory and return tidy:
    year, unitid, cipcode, majornum, awlevel, ctotalt
    """
    frames = []
    for fp in sorted(completions_dir.glob('*.csv')):
        df = read_ipeds_csv(fp)
        df = normalize_cols(df)
        year = infer_year_from_filename(fp)
        # Flexible column names
        colmap = {}
        for c in ['unitid', 'cipcode', 'majornum', 'major_num', 'awlevel', 'ctotalt', 'ctotal_t', 'ciptitle']:
            if c in df.columns:
                colmap[c] = c
        # minimally require unitid, cipcode, awlevel and a total completions field
        total_col = 'ctotalt' if 'ctotalt' in df.columns else ('ctotal_t' if 'ctotal_t' in df.columns else None)
        if not all(k in df.columns for k in ['unitid', 'cipcode', 'awlevel']) or total_col is None:
            # skip files that don't match expected schema
            continue

        # build subframe
        subcols = ['unitid', 'cipcode', 'awlevel', total_col]
        if 'majornum' in df.columns:
            subcols.append('majornum')
        elif 'major_num' in df.columns:
            subcols.append('major_num')
        if 'ciptitle' in df.columns:
            subcols.append('ciptitle')

        sub = df[subcols].copy()
        sub.rename(columns={total_col: 'ctotal'}, inplace=True)
        if 'major_num' in sub.columns and 'majornum' not in sub.columns:
            sub.rename(columns={'major_num': 'majornum'}, inplace=True)

        # add year
        sub['year'] = year

        # clean types
        sub['ctotal'] = pd.to_numeric(sub['ctotal'], errors='coerce').fillna(0).astype(int)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=['year', 'unitid', 'cipcode', 'majornum', 'awlevel', 'ciptitle', 'ctotal'])

    out = pd.concat(frames, ignore_index=True)
    # normalize missing optional cols
    for c in ['majornum', 'ciptitle']:
        if c not in out.columns:
            out[c] = pd.NA
    # ensure column order
    out = out[['year', 'unitid', 'cipcode', 'majornum', 'awlevel', 'ciptitle', 'ctotal']]
    return out


def add_hd(out_df: pd.DataFrame, hd_union: pd.DataFrame) -> pd.DataFrame:
    """
    Join completions with HD on unitid and (if possible) same year; if missing, fallback to latest HD name/state.
    """
    if out_df.empty or hd_union.empty:
        return out_df

    # first try exact year match
    merged = out_df.merge(hd_union, on=['unitid', 'year'], how='left', suffixes=('', '_hd'))
    # for rows still missing instnm, use latest available HD
    if merged['instnm'].isna().any():
        latest_hd = (hd_union.sort_values(['unitid', 'year'])
                              .drop_duplicates('unitid', keep='last')[['unitid', 'instnm', 'stabbr']])
        merged = merged.drop(columns=['instnm', 'stabbr'])
        merged = merged.merge(latest_hd, on='unitid', how='left')
    return merged


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="IPEDS + Open Doors data helper")
    # Open Doors args
    ap.add_argument('--open-doors-csv', type=str, help='Path to Open Doors CSV (international students by place of origin).')
    ap.add_argument('--country', type=str, default='Guyana', help='Country/place of origin to filter (default: Guyana).')
    ap.add_argument('--od-country-field', type=str, default='place_of_origin', help='Column name for country/place in Open Doors CSV.')
    ap.add_argument('--od-inst-field', type=str, default='institution', help='Column name for institution name in Open Doors CSV.')
    ap.add_argument('--od-state-field', type=str, default='state', help='Column name for state in Open Doors CSV (if present).')
    ap.add_argument('--od-year-field', type=str, default='academic_year', help='Column name for year in Open Doors CSV.')
    ap.add_argument('--od-count-field', type=str, default='total', help='Column name for count in Open Doors CSV.')

    # IPEDS completions args
    ap.add_argument('--ipeds-completions-dir', type=str, help='Directory containing IPEDS Completions CSVs (e.g., C2021_A.csv).')
    ap.add_argument('--ipeds-hd-dir', type=str, help='Directory containing IPEDS Directory (HD) CSVs (e.g., hd2021.csv).')

    ap.add_argument('--outdir', type=str, default='outputs', help='Output directory (default: outputs)')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) Open Doors (Guyana) -> institution/state/year tidy + summary
    if args.open_doors_csv:
        od_df = load_opendoors(
            csv_path=Path(args.open_doors_csv),
            country_value=args.country,
            country_field=args.od_country_field,
            institution_field=args.od_inst_field,
            state_field=args.od_state_field,
            year_field=args.od_year_field,
            count_field=args.od_count_field
        )
        if od_df.empty:
            print("Open Doors: no rows matched the given country or columns. Check column mappings.")
        else:
            inst_out = od_df[['year', 'instnm', 'stabbr', 'guyana_students']].sort_values(['year', 'instnm'])
            inst_out.to_csv(outdir / 'guyana_by_institution.csv', index=False)

            state_out = summarize_opendoors_state(od_df)
            state_out.to_csv(outdir / 'guyana_by_state.csv', index=False)

            # also a convenient cross-year state summary (wide)
            state_pivot = state_out.pivot(index='stabbr', columns='year', values='guyana_students').fillna(0).astype(int)
            state_pivot.to_csv(outdir / 'state_summary.csv')

            print(f"Wrote {outdir/'guyana_by_institution.csv'}, {outdir/'guyana_by_state.csv'}, and {outdir/'state_summary.csv'}")

    # 2) IPEDS Completions by field (CIP)
    if args.ipeds_completions_dir:
        comp = load_completions(Path(args.ipeds_completions_dir))
        if comp.empty:
            print("Completions: no usable CSVs found in directory. Check files & schema.")
        else:
            if args.ipeds_hd_dir:
                hd_union = load_directory_lookup(Path(args.ipeds_hd_dir))
                comp = add_hd(comp, hd_union)

            # order columns
            order = ['year', 'unitid', 'instnm', 'stabbr', 'cipcode', 'ciptitle', 'awlevel', 'majornum', 'ctotal']
            for c in order:
                if c not in comp.columns:
                    comp[c] = pd.NA
            comp = comp[order].sort_values(['year', 'unitid', 'cipcode', 'awlevel', 'majornum'])

            comp.to_csv(outdir / 'ipeds_completions_by_field.csv', index=False)
            print(f"Wrote {outdir/'ipeds_completions_by_field.csv'}")

    if not args.open_doors_csv and not args.ipeds_completions_dir:
        print("Nothing to do. Provide --open-doors-csv and/or --ipeds-completions-dir.")
        return


if __name__ == "__main__":
    main()
