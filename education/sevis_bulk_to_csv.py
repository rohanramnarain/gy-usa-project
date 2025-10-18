#!/usr/bin/env python3
"""
sevis_bulk_to_csv.py

Scrape monthly SEVIS Data Mapping Tool "Data" pages (STEM and/or non-STEM),
save each page as a CSV, and build Guyana summaries across months.

Defaults:
- Index page: https://studyinthestates.dhs.gov/sevis-by-the-numbers/sevis-by-the-numbers-data
- Includes BOTH STEM and non-STEM by default (you can limit with flags).
- Date range can be constrained with --start YYYY-MM --end YYYY-MM.

Outputs (by default under ./sevis_out/):
- raw/<YYYY-MM>[-stem].csv                        (raw table per month/page)
- guyana_state_by_month.csv                       (tidy: year, month, state, total, stem)
- guyana_school_by_month.csv (when available)     (tidy: year, month, institution, state?, total, stem)

Usage examples:
  # Everything
  python sevis_bulk_to_csv.py

  # STEM only, since 2024-01
  python sevis_bulk_to_csv.py --stem-only --start 2024-01

  # Non-STEM only, up to 2025-10
  python sevis_bulk_to_csv.py --non-stem-only --end 2025-10

Requirements (install once in your environment):
  python -m pip install lxml beautifulsoup4 html5lib
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import requests
import pandas as pd
from io import StringIO
from urllib.parse import urljoin
import html as ihtml  # for html.unescape


INDEX_URL = "https://studyinthestates.dhs.gov/sevis-by-the-numbers/sevis-by-the-numbers-data"
BASE = "https://studyinthestates.dhs.gov"


def month_to_num(mon: str) -> int:
    return datetime.strptime(mon[:3], "%b").month


def parse_link_text(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Given anchor text like 'September 2025 SEVIS Data Mapping Tool Data' or
    'September 2025 STEM SEVIS Data Mapping Tool Data' return (year, month, is_stem).
    """
    t = text.strip()
    m = re.search(
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        t, re.I
    )
    year = month = None
    is_stem = bool(re.search(r'\bSTEM\b', t, re.I))
    if m:
        month = month_to_num(m.group(1).title())
        year = int(m.group(2))
    return year, month, is_stem


def extract_data_links(index_html: str) -> List[Tuple[str, str, Optional[int], Optional[int], bool]]:
    """
    Return list of tuples: (href, anchor_text, year, month, is_stem)
    """
    links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', index_html, flags=re.I | re.S)
    out = []
    for href, text in links:
        if 'sevis-data-mapping-tool' in href and href.endswith('-sevis-data-mapping-tool-data'):
            year, month, is_stem = parse_link_text(re.sub(r'<.*?>', '', text))
            # some URLs may be relative; normalize
            if href.startswith('/'):
                url = BASE + href
            elif href.startswith('http'):
                url = href
            else:
                url = BASE + '/' + href.lstrip('/')
            out.append((url, re.sub(r'<.*?>', '', text).strip(), year, month, is_stem))
    # de-duplicate by URL (some pages might appear multiple times)
    seen = set()
    uniq = []
    for item in out:
        if item[0] not in seen:
            uniq.append(item)
            seen.add(item[0])
    # sort by year, month, is_stem
    uniq.sort(key=lambda x: (x[2] or 0, x[3] or 0, x[4], x[0]))
    return uniq


def fetch(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


# --------------------------
# Robust HTML/CSV extraction
# --------------------------

def _read_html_tables(html_text: str, header):
    """Return non-empty tables using lxml first, then html5lib; wrap string in StringIO to avoid deprecation."""
    sio = StringIO(html_text)
    for flavor in ("lxml", "html5lib"):
        try:
            tables = pd.read_html(sio, header=header, flavor=flavor)
            return [t for t in tables if not t.empty]
        except Exception:
            sio.seek(0)
            continue
    # last try: let pandas choose (if any parser available)
    sio.seek(0)
    try:
        return [t for t in pd.read_html(sio, header=header) if not t.empty]
    except Exception:
        return []


# match href='...csv' or href="...csv" or data-csv-url='...'
CSV_HREF_PAT = re.compile(
    r'href\s*=\s*["\']([^"\']*?\.csv[^"\']*)["\']|data-csv-url\s*=\s*["\']([^"\']+)["\']',
    re.I
)


def _find_csv_url(page_html: str, page_url: str) -> Optional[str]:
    for m in CSV_HREF_PAT.finditer(page_html):
        href = next((g for g in m.groups() if g), None)
        if href:
            return urljoin(page_url, ihtml.unescape(href))
    return None


def _detect_and_apply_header(df: pd.DataFrame) -> pd.DataFrame:
    """Try to find the header row heuristically and apply it."""
    tokens = re.compile(
        r'(StateAbbrv|CoC|Country|Student\s*Education\s*Level|Count\s+of\s+Active|Active\s+Students)',
        re.I
    )
    header_idx = None
    for i in range(min(len(df), 50)):
        row = df.iloc[i].astype(str)
        if row.str.contains(tokens, na=False).any():
            header_idx = i
            break
    if header_idx is None:
        return df
    header = df.iloc[header_idx].astype(str).str.strip().tolist()
    df = df.iloc[header_idx + 1:].copy()
    df.columns = header
    return df


def read_table_or_csv(page_html: str, page_url: str) -> pd.DataFrame:
    """
    Try to extract a table from HTML (multiple parsers), then AMP page,
    and finally fall back to a CSV link if present.
    """
    # 1) HTML tables with header
    tables = _read_html_tables(page_html, header=0)
    if tables:
        df = max(tables, key=lambda d: d.shape[1])
        if not all(isinstance(c, int) for c in df.columns):
            return df

    # 2) HTML tables without header + heuristic header detection
    tables = _read_html_tables(page_html, header=None)
    if tables:
        df = max(tables, key=lambda d: d.shape[1])
        return _detect_and_apply_header(df)

    # 3) AMP fallback (often static markup)
    amp_url = page_url.rstrip("/") + "/amp"
    try:
        amp_html = fetch(amp_url)
        tables = _read_html_tables(amp_html, header=0) or _read_html_tables(amp_html, header=None)
        if tables:
            df = max(tables, key=lambda d: d.shape[1])
            if all(isinstance(c, int) for c in df.columns):
                df = _detect_and_apply_header(df)
            return df
    except Exception:
        pass

    # 4) CSV fallback
    csv_url = _find_csv_url(page_html, page_url)
    if csv_url:
        r = requests.get(csv_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))

    raise ValueError("No table or CSV link found on page")


# --------------------------
# Downstream processing utils
# --------------------------

def safe_int(s) -> int:
    return int(pd.to_numeric(str(s).replace(',', ''), errors='coerce') or 0)


def detect_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    def pick(cols, *pats):
        for pat in pats:
            for c in cols:
                if re.search(pat, str(c), flags=re.I):
                    return c
        return None

    cols = list(df.columns)
    return {
        'country': pick(cols, r'\bCoC\b', r'Country of Citizenship', r'\bCountry\b'),
        'state': pick(cols, r'\bStateAbbrv\b', r'\bState\b', r'U\.?S\.?\s*State'),
        'school': pick(cols, r'\bU\.?S\.?\s*School\b', r'\bSchool Name\b', r'\bSchool\b'),
        'count': pick(cols, r'Count of Active STEM Students', r'Active Students', r'\bCount\b'),
        'edu': pick(cols, r'Student\s*Education\s*Level')
    }


def normalize_counts(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
    df[count_col] = (df[count_col].astype(str).str.replace(',', ''))
    df[count_col] = pd.to_numeric(df[count_col], errors='coerce').fillna(0).astype(int)
    return df


def within_range(year: int, month: int, start: Optional[str], end: Optional[str]) -> bool:
    if not year or not month:
        return True
    ym = year * 100 + month
    if start:
        y, m = map(int, start.split('-'))
        if ym < y * 100 + m:
            return False
    if end:
        y, m = map(int, end.split('-'))
        if ym > y * 100 + m:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index-url', default=INDEX_URL, help='Index page with monthly links')
    ap.add_argument('--outdir', default='sevis_out', help='Output directory')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--stem-only', action='store_true', help='Only STEM pages')
    g.add_argument('--non-stem-only', action='store_true', help='Only non-STEM pages')
    ap.add_argument('--start', type=str, default=None, help='Earliest YYYY-MM to include (inclusive)')
    ap.add_argument('--end', type=str, default=None, help='Latest YYYY-MM to include (inclusive)')
    ap.add_argument('--country', type=str, default='Guyana', help='Country filter for summaries')
    ap.add_argument('--sleep', type=float, default=0.8, help='Seconds between requests')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / 'raw').mkdir(parents=True, exist_ok=True)

    # 1) Fetch index and extract links
    index_html = fetch(args.index_url)
    links = extract_data_links(index_html)
    if not links:
        raise SystemExit("No monthly links found on index page.")

    # Filter by STEM/non-STEM
    if args.stem_only:
        links = [x for x in links if x[4] is True]
    elif args.non_stem_only:
        links = [x for x in links if x[4] is False]

    # Filter by date range
    links = [x for x in links if within_range(x[2], x[3], args.start, args.end)]
    if not links:
        raise SystemExit("No links after applying filters.")

    # 2) Iterate months, save raw CSV, and accumulate country summaries
    rows_state = []
    rows_school = []

    for url, text, year, month, is_stem in links:
        # slug for filename
        stem_tag = '-stem' if is_stem else ''
        if year and month:
            slug = f"{year:04d}-{month:02d}{stem_tag}"
        else:
            # fallback to last path segment
            slug = url.rstrip('/').split('/')[-1].replace('-sevis-data-mapping-tool-data', '')
        print(f"Fetching {text} -> {slug}")
        html = fetch(url)
        df = read_table_or_csv(html, url)

        # Save raw
        raw_path = outdir / 'raw' / f"{slug}.csv"
        df.to_csv(raw_path, index=False)

        # Detect columns and build country subset
        cols = detect_cols(df)
        country_col, state_col, school_col, count_col = (
            cols['country'], cols['state'], cols['school'], cols['count']
        )

        if not country_col or not count_col:
            time.sleep(args.sleep)
            continue

        df[count_col] = pd.to_numeric(df[count_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        country_mask = df[country_col].astype(str).str.casefold().eq(args.country.casefold())
        guy = df.loc[country_mask].copy()

        # State-level summary
        if state_col and state_col in guy.columns:
            state_sum = (guy.groupby(state_col, as_index=False)[count_col].sum())
            for _, r in state_sum.iterrows():
                rows_state.append({
                    'year': year, 'month': month,
                    'stem': is_stem,
                    'state': r[state_col], 'total': int(r[count_col])
                })
        else:
            # total only
            total = int(guy[count_col].sum())
            rows_state.append({
                'year': year, 'month': month, 'stem': is_stem,
                'state': None, 'total': total
            })

        # School-level summary (when available)
        if school_col and school_col in guy.columns:
            group_cols = [school_col] + ([state_col] if state_col and state_col in guy.columns else [])
            school_sum = (guy.groupby(group_cols, as_index=False)[count_col].sum())
            for _, r in school_sum.iterrows():
                rows_school.append({
                    'year': year, 'month': month, 'stem': is_stem,
                    'institution': r[school_col],
                    'state': r[state_col] if state_col and state_col in r else None,
                    'total': int(r[count_col])
                })

        time.sleep(args.sleep)

    # Write accumulated summaries
    if rows_state:
        df_state = pd.DataFrame(rows_state).sort_values(['year', 'month', 'state', 'stem'])
        df_state.to_csv(outdir / 'guyana_state_by_month.csv', index=False)
        print("Wrote", outdir / 'guyana_state_by_month.csv')

    if rows_school:
        df_school = pd.DataFrame(rows_school).sort_values(['year', 'month', 'institution', 'state', 'stem'])
        df_school.to_csv(outdir / 'guyana_school_by_month.csv', index=False)
        print("Wrote", outdir / 'guyana_school_by_month.csv')

    print("Done. Raw files in:", outdir / 'raw')


if __name__ == "__main__":
    main()
