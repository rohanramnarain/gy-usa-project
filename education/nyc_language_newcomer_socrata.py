
#!/usr/bin/env python3
"""
NYC Open Data (Socrata) — Home language & newcomer indicators

What this script does
---------------------
- Queries one or two NYC Open Data (Socrata) datasets by ID.
- Produces borough-by-language distributions (tidy CSV) and an optional wide pivot.
- Optionally merges an immigrant/newcomer indicator dataset at district/borough level.
- Highlights languages relevant to Indo‑Caribbean communities (customizable).

Inputs
------
- --languages-dataset-id  (required)
- --borough-field, --language-field, and optionally --count-field (if dataset already aggregated)
- If the dataset is row-level (one student/record per row), the script will use COUNT(*) via SoQL.
- --newcomer-dataset-id (optional) with --newcomer-borough-field/--newcomer-district-field/--newcomer-flag-field
- --app-token (optional) Socrata app token for higher rate limits

Outputs
-------
- outputs/nyc_borough_by_language.csv  (tidy)
- outputs/nyc_borough_by_language_wide.csv (pivot table)
- outputs/nyc_notes.txt  (includes note: "language ≠ nationality")

Usage
-----
python nyc_language_newcomer_socrata.py \
  --languages-dataset-id <abcd-1234> \
  --borough-field borough \
  --language-field home_language

# For an already-aggregated dataset with a column 'count'
python nyc_language_newcomer_socrata.py \
  --languages-dataset-id <abcd-1234> \
  --borough-field borough \
  --language-field language \
  --count-field count

# With a newcomer/immigrant indicator dataset (example fields shown)
python nyc_language_newcomer_socrata.py \
  --languages-dataset-id <abcd-1234> \
  --borough-field borough \
  --language-field home_language \
  --newcomer-dataset-id <wxyz-7890> \
  --newcomer-borough-field borough \
  --newcomer-flag-field is_newcomer
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
import pandas as pd


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def socrata_get(domain: str, dataset_id: str, params: Dict[str, Any], app_token: Optional[str] = None) -> pd.DataFrame:
    base = f"https://{domain}/resource/{dataset_id}.json"
    headers = {}
    if app_token:
        headers['X-App-Token'] = app_token

    rows = []
    offset = 0
    limit = int(params.get('$limit', 50000))  # page size
    while True:
        page_params = dict(params)
        page_params['$offset'] = offset
        resp = requests.get(base, headers=headers, params=page_params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        if len(data) < limit:
            break
        offset += limit

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def languages_aggregate(domain: str,
                        dataset_id: str,
                        borough_field: str,
                        language_field: str,
                        count_field: Optional[str],
                        where: Optional[str],
                        app_token: Optional[str]) -> pd.DataFrame:
    """
    Return tidy borough-language counts. If count_field is None, uses COUNT(*) via SoQL.
    """
    if count_field:
        select = f"{borough_field} as borough, {language_field} as language, sum({count_field}) as n"
    else:
        select = f"{borough_field} as borough, {language_field} as language, count(1) as n"

    params = {
        '$select': select,
        '$where': where if where else None,
        '$group': 'borough, language',
        '$order': 'borough, language',
        '$limit': 50000
    }
    # Remove None
    params = {k: v for k, v in params.items() if v is not None}

    df = socrata_get("data.cityofnewyork.us", dataset_id, params, app_token)
    if df.empty:
        return df

    # normalize
    df.columns = [c.lower() for c in df.columns]
    # coerce counts
    if 'n' in df.columns:
        df['n'] = pd.to_numeric(df['n'], errors='coerce').fillna(0).astype(int)

    # trim strings
    for col in ['borough', 'language']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df[['borough', 'language', 'n']].sort_values(['borough', 'language'])


def newcomer_aggregate(domain: str,
                       dataset_id: str,
                       borough_field: Optional[str],
                       district_field: Optional[str],
                       flag_field: Optional[str],
                       where: Optional[str],
                       app_token: Optional[str]) -> pd.DataFrame:
    """
    Aggregate newcomer/immigrant indicator by borough or district.
    """
    geo_field = borough_field or district_field
    if not geo_field:
        return pd.DataFrame()

    if not flag_field:
        # just count rows
        select = f"{geo_field} as geo, count(1) as num_records"
        group = "geo"
    else:
        # count flagged
        select = f"""{geo_field} as geo,
                     sum(case({flag_field}='true',1,0)) as newcomers,
                     count(1) as total"""
        group = "geo"

    params = {
        '$select': select,
        '$where': where if where else None,
        '$group': group,
        '$order': 'geo',
        '$limit': 50000
    }
    params = {k: v for k, v in params.items() if v is not None}

    df = socrata_get("data.cityofnewyork.us", dataset_id, params, app_token)
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]
    for c in ['newcomers', 'total', 'num_records']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    return df


def standardize_language(name: str) -> str:
    """
    Normalize language labels (very light-touch to keep source integrity).
    """
    s = (name or "").strip()
    # Unify some common variants
    replacements = {
        "hindi/urdu": "hindi-urdu",
        "urdu/hindi": "hindi-urdu",
        "guyana creole english": "guyanese creole",
        "guyanese english": "guyanese creole",
        "trinidadian english creole": "trinidadian creole",
    }
    key = s.lower()
    return replacements.get(key, s)


def main():
    ap = argparse.ArgumentParser(description="NYC Open Data (Socrata) — borough x language & newcomer")
    ap.add_argument("--languages-dataset-id", type=str, required=True, help="Socrata dataset id for home language distribution")
    ap.add_argument("--borough-field", type=str, default="borough", help="Field containing borough")
    ap.add_argument("--language-field", type=str, default="home_language", help="Field containing language")
    ap.add_argument("--count-field", type=str, default=None, help="If dataset is aggregated, provide the count column name")
    ap.add_argument("--language-where", type=str, default=None, help="Optional SoQL $where filter for languages dataset")

    ap.add_argument("--newcomer-dataset-id", type=str, default=None, help="Optional dataset id for newcomer/immigrant indicators")
    ap.add_argument("--newcomer-borough-field", type=str, default=None, help="Borough field in newcomer dataset (or leave blank and use district)")
    ap.add_argument("--newcomer-district-field", type=str, default=None, help="District field in newcomer dataset")
    ap.add_argument("--newcomer-flag-field", type=str, default=None, help="Boolean-like field indicating newcomer status (e.g., 'true'/'false')")
    ap.add_argument("--newcomer-where", type=str, default=None, help="Optional SoQL $where for newcomer dataset")

    ap.add_argument("--app-token", type=str, default=None, help="Socrata app token for higher rate limits")

    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    ap.add_argument("--languages-of-interest", type=str, nargs="*", default=[
        "Hindi", "Urdu", "Bengali", "Punjabi", "Gujarati",
        "Tamil", "Marathi", "Guyanese Creole", "Trinidadian Creole",
        "Dutch", "Sranan Tongo"
    ], help="Languages to flag as relevant to Indo‑Caribbean communities (override as needed)")

    args = ap.parse_args()
    outdir = Path(args.outdir); ensure_outdir(outdir)

    # 1) Borough x language aggregation
    lang_df = languages_aggregate(
        domain="data.cityofnewyork.us",
        dataset_id=args.languages_dataset_id,
        borough_field=args.borough_field,
        language_field=args.language_field,
        count_field=args.count_field,
        where=args.language_where,
        app_token=args.app_token
    )

    if lang_df.empty:
        print("No rows returned from the languages dataset. Check dataset id/fields/filters.")
        return

    # Standardize language labels slightly
    lang_df['language_std'] = lang_df['language'].map(standardize_language)

    # Flag Indo‑Caribbean‑relevant languages (case-insensitive exact match after standardization)
    interest_set = {s.lower() for s in args.languages_of_interest}
    lang_df['is_indocaribbean_language'] = lang_df['language_std'].str.lower().isin(interest_set)

    # Save tidy
    tidy_path = outdir / "nyc_borough_by_language.csv"
    lang_df[['borough', 'language', 'n', 'language_std', 'is_indocaribbean_language']].to_csv(tidy_path, index=False)

    # Wide pivot
    wide = lang_df.pivot_table(index='borough', columns='language_std', values='n', aggfunc='sum', fill_value=0)
    wide_path = outdir / "nyc_borough_by_language_wide.csv"
    wide.to_csv(wide_path)

    # 2) Optional newcomer/immigrant indicator aggregation
    if args.newcomer_dataset_id:
        new_df = newcomer_aggregate(
            domain="data.cityofnewyork.us",
            dataset_id=args.newcomer_dataset_id,
            borough_field=args.newcomer_borough_field,
            district_field=args.newcomer_district_field,
            flag_field=args.newcomer_flag_field,
            where=args.newcomer_where,
            app_token=args.app_token
        )
        if not new_df.empty:
            # If this is borough-level, left-join to wide/tidy on borough for convenience
            if 'geo' in new_df.columns:
                if args.newcomer_borough_field:
                    # Save standalone newcomer summary
                    new_df.sort_values('geo').to_csv(outdir / "nyc_newcomer_summary.csv", index=False)
                else:
                    # district-level only: save as-is
                    new_df.sort_values('geo').to_csv(outdir / "nyc_newcomer_by_district.csv", index=False)

    # Notes file
    notes = (
        "Notes:\n"
        "- 'Home language' reflects self-reported language used at home and does **not** imply nationality or ethnicity.\n"
        "- Categories and labels can vary by dataset and year; always review the data dictionary.\n"
        "- The 'languages_of_interest' list is a convenience for Indo‑Caribbean analysis; please adjust to your needs.\n"
        "- If your dataset is already aggregated by borough & language, pass --count-field to sum that column.\n"
    )
    with open(outdir / "nyc_notes.txt", "w", encoding="utf-8") as f:
        f.write(notes)

    print(f"Wrote {tidy_path} and {wide_path}. See nyc_notes.txt for caveats.")


if __name__ == "__main__":
    main()
