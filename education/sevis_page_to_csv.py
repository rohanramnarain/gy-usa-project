
#!/usr/bin/env python3
"""
sevis_page_to_csv.py

Download a SEVIS Data Mapping Tool "Data" page (e.g. STEM or non‑STEM monthly snapshot)
and turn the main HTML table into a CSV.

Usage:
  python sevis_page_to_csv.py "https://studyinthestates.dhs.gov/sevis-data-mapping-tool/october-2025-stem-sevis-data-mapping-tool-data" -o october_2025_stem.csv
"""
import argparse
import sys
import requests
import pandas as pd

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text

def pick_main_table(tables):
    if not tables:
        return None
    tables = [t for t in tables if not t.empty]
    if not tables:
        return None
    tables.sort(key=lambda df: df.shape[1], reverse=True)
    return tables[0]

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').copy()
    df.columns = [str(c).strip() for c in df.columns]
    # remove duplicate header rows (where row equals header names)
    header_set = set(df.columns)
    mask_dupe_header = df.apply(lambda row: set(str(v).strip() for v in row.values) == header_set, axis=1)
    if mask_dupe_header.any():
        df = df[~mask_dupe_header].copy()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="SEVIS Data Mapping Tool Data page URL")
    ap.add_argument("-o", "--out", default="sevis_data.csv", help="Output CSV path (default: sevis_data.csv)")
    args = ap.parse_args()

    html = fetch_html(args.url)
    tables = pd.read_html(html, flavor="lxml")
    df = pick_main_table(tables)
    if df is None or df.empty:
        print("No tables found on the page, or parsing failed.", file=sys.stderr)
        sys.exit(2)

    df = clean_frame(df)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({df.shape[0]} rows, {df.shape[1]} columns)")

if __name__ == "__main__":
    main()
