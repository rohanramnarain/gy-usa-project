#!/usr/bin/env python3
# irs_soi_county_migration.py
# Usage:
#   python irs_soi_county_migration.py --year_label "2010-2011" --counties 36081 36047 12011 34013 36093 --out data/irs
# Notes:
#   - Matches year labels like "2010–2011" (en-dash) and "2010 to 2011".
#   - Accepts CSV/XLS/XLSX/ZIP; resolves relative URLs.
#   - Handles historical schemas including:
#       * y1_* / y2_* (origin/dest)
#       * statefips1/countyfips1 and statefips2/countyfips2
#       * State_Code_Origin/County_Code_Origin and State_Code_Dest/County_Code_Dest
#       * N1/Returns/NRET/Return_Num for return counts

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

DOWNLOADS_PAGE = "https://www.irs.gov/statistics/soi-tax-stats-migration-data-downloads"
HISTORICAL_PAGE = "https://www.irs.gov/statistics/soi-tax-stats-county-to-county-migration-data-files"
BASE = "https://www.irs.gov"
EXTS = (".csv", ".zip", ".xls", ".xlsx")

# ---------- HTTP helpers ----------

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; IRS-Scraper/1.0; +https://example.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s

def _get(url, session=None, **kwargs):
    session = session or _session()
    for i in range(3):
        try:
            r = session.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if i == 2:
                raise
            time.sleep(1.5 * (i + 1))

# ---------- Year label matching ----------

def _normalize_year_label(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+to\s+", "-", s)          # "2010 to 2011" -> "2010-2011"
    s = re.sub(r"[^\d\-]", "", s)
    return s

def _year_match(year_label: str, text: str) -> bool:
    yl = _normalize_year_label(year_label)
    years = re.findall(r"\d{4}", yl)
    if len(years) != 2:
        return False
    y1, y2 = years
    t = _normalize_year_label(text or "")
    return (
        (y1 in t and y2 in t) or
        (f"{y1}-{y2}" in t) or
        (f"{y1}{y2}" in t) or
        (f"{y1[-2:]}{y2[-2:]}" in t)
    )

def _href_has_downloadable_ext(href: str) -> bool:
    if not href:
        return False
    href_l = href.lower()
    if href_l.endswith(EXTS):
        return True
    q = parse_qs(urlparse(href).query)
    fname = (q.get("filename", [None])[0] or "").lower()
    return any(fname.endswith(ext) for ext in EXTS)

# ---------- Link discovery ----------

def find_county_links(year_label):
    """
    Find county-level inflow/outflow data links (csv/xls/xlsx/zip) for the given year label.
    Matches both anchor text and href, handles relative URLs, and de-dupes.
    """
    pages = [DOWNLOADS_PAGE, HISTORICAL_PAGE]
    sess = _session()
    links = []

    for page_url in pages:
        html = _get(page_url, session=sess).text
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True) or ""
            hay = f"{text} {href}".lower()

            if "county" not in hay:
                continue
            if not _year_match(year_label, hay):
                continue
            if not _href_has_downloadable_ext(href):
                continue

            links.append(urljoin(BASE, href))

    # De-dupe while preserving order
    seen = set()
    deduped = []
    for u in links:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

# ---------- File loading ----------

def load_csv_or_zip(url):
    """
    Load a CSV or Excel file from URL. If ZIP, prefer CSV inside, otherwise Excel.
    Returns a pandas.DataFrame with all columns as strings where possible.
    """
    sess = _session()
    r = _get(url, session=sess, stream=True)
    content_type = r.headers.get("Content-Type", "").lower()
    url_l = url.lower()

    def _read_excel(fobj):
        return pd.read_excel(fobj, dtype=str)

    def _read_csv(fobj, text_mode=False):
        if text_mode:
            return pd.read_csv(io.StringIO(fobj), dtype=str)
        return pd.read_csv(fobj, dtype=str)

    parsed = urlparse(url)
    fname = (parse_qs(parsed.query).get("filename", [parsed.path.split("/")[-1]])[0] or "").lower()
    ext = None
    for e in EXTS:
        if url_l.endswith(e) or fname.endswith(e):
            ext = e
            break

    content = r.content

    if ext == ".zip" or "zip" in content_type:
        z = zipfile.ZipFile(io.BytesIO(content))
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        xls_names = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))]
        if csv_names:
            with z.open(csv_names[0]) as f:
                return _read_csv(f)
        if xls_names:
            with z.open(xls_names[0]) as f:
                return _read_excel(f)
        raise ValueError("No CSV/XLS found inside ZIP.")

    if ext in (".xls", ".xlsx") or "excel" in content_type:
        return _read_excel(io.BytesIO(content))

    try:
        return _read_csv(io.BytesIO(content))
    except Exception:
        return _read_csv(r.text, text_mode=True)

# ---------- Schema normalization ----------

def _first_match(cols_lc_map, candidates):
    """Return the original column name for the first candidate key present (case-insensitive exact or startswith)."""
    # exact first
    for cand in candidates:
        if cand in cols_lc_map:
            return cols_lc_map[cand]
    # startswith fallback
    for cand in candidates:
        for lc, orig in cols_lc_map.items():
            if lc.startswith(cand):
                return orig
    return None

def _digits_zfill(x, width):
    if x is None:
        return None
    s = str(x)
    s = re.sub(r"\D", "", s)
    if s == "":
        s = "0"
    return s.zfill(width)

def normalize_cols(df):
    """
    Normalize various historical schemas to a consistent flow table:
      columns: orig_fips, dest_fips, returns (int)
    Handles:
      - y1_* / y2_* (origin/dest)
      - statefips1/countyfips1 and statefips2/countyfips2
      - State_Code_Origin/County_Code_Origin and State_Code_Dest/County_Code_Dest
      - N1/Returns/NRET/Return_Num for flows (returns)
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame")

    # Normalize column name lookup
    cols_lc_map = {c.strip().lower(): c for c in df.columns}

    # Candidate name sets seen across SOI files (lower-case)
    y1_state_candidates = [
        "y1_statefips", "statefips1", "state1fips", "y1_state", "statefips_from",
        "state_code_origin", "statecode_origin", "state_code_from", "state_from",
        "origin_state_code", "st_code_origin"
    ]
    y1_county_candidates = [
        "y1_countyfips", "countyfips1", "y1_county", "county1", "county_code1",
        "county_code_origin", "countycode_origin", "county_code_from", "county_from",
        "origin_county_code", "cty_code_origin"
    ]
    y2_state_candidates = [
        "y2_statefips", "statefips2", "state2fips", "y2_state", "statefips_to",
        "state_code_dest", "statecode_dest", "state_code_to", "state_to",
        "dest_state_code", "st_code_dest"
    ]
    y2_county_candidates = [
        "y2_countyfips", "countyfips2", "y2_county", "county2", "county_code2",
        "county_code_dest", "countycode_dest", "county_code_to", "county_to",
        "dest_county_code", "cty_code_dest"
    ]

    # Flow/return count
    returns_candidates_exact = ["n1", "returns", "nret", "return_num", "ret_num"]
    returns_candidates_fuzzy = ["n_1", "num_returns", "return", "ret"]

    c_y1_state = _first_match(cols_lc_map, y1_state_candidates)
    c_y1_county = _first_match(cols_lc_map, y1_county_candidates)
    c_y2_state = _first_match(cols_lc_map, y2_state_candidates)
    c_y2_county = _first_match(cols_lc_map, y2_county_candidates)

    c_returns = _first_match(cols_lc_map, returns_candidates_exact)
    if c_returns is None:
        c_returns = _first_match(cols_lc_map, returns_candidates_fuzzy)

    # If still missing, try generic token-based match (state vs county; origin vs dest)
    def _token_find(tokens_all, prefer_state=True):
        # tokens_all: list of token sets; we accept if all tokens appear in the normalized column name
        def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
        for lc, orig in cols_lc_map.items():
            n = norm(lc)
            for toks in tokens_all:
                if all(t in n for t in toks):
                    return orig
        return None

    if c_y1_state is None:
        c_y1_state = _token_find([["state", "origin"], ["state", "from"], ["statefips", "1"], ["state", "y1"]])
    if c_y1_county is None:
        c_y1_county = _token_find([["county", "origin"], ["county", "from"], ["countyfips", "1"], ["county", "y1"]])
    if c_y2_state is None:
        c_y2_state = _token_find([["state", "dest"], ["state", "to"], ["statefips", "2"], ["state", "y2"]])
    if c_y2_county is None:
        c_y2_county = _token_find([["county", "dest"], ["county", "to"], ["countyfips", "2"], ["county", "y2"]])

    # Minimal columns required
    needed = [c_y1_state, c_y1_county, c_y2_state, c_y2_county]
    if any(x is None for x in needed):
        raise ValueError(f"Could not infer FIPS columns from schema. Columns found: {list(df.columns)}")

    if c_returns is None:
        # Prefer 'exemptions' only as a last fallback if present (not ideal for "returns")
        # or pick a numeric-like column.
        exmpt = _first_match(cols_lc_map, ["exmpt_num", "exemptions", "n2"])
        if exmpt is not None:
            c_returns = exmpt
        else:
            numeric_like = []
            for c in df.columns:
                try:
                    ser = pd.to_numeric(df[c], errors="coerce")
                    if ser.notna().sum() >= max(5, int(0.5 * len(ser))):
                        numeric_like.append((c, ser))
                except Exception:
                    pass
            if numeric_like:
                c_returns = numeric_like[0][0]
            else:
                raise ValueError("Could not find returns column (e.g., N1, RETURNS, NRET, Return_Num).")

    # Build flow table
    flow = df[[c_y1_state, c_y1_county, c_y2_state, c_y2_county, c_returns]].copy()
    flow.columns = ["orig_statefips", "orig_countyfips", "dest_statefips", "dest_countyfips", "returns"]

    # Clean and pad FIPS
    flow["orig_statefips"] = flow["orig_statefips"].map(lambda x: _digits_zfill(x, 2))
    flow["orig_countyfips"] = flow["orig_countyfips"].map(lambda x: _digits_zfill(x, 3))
    flow["dest_statefips"] = flow["dest_statefips"].map(lambda x: _digits_zfill(x, 2))
    flow["dest_countyfips"] = flow["dest_countyfips"].map(lambda x: _digits_zfill(x, 3))

    flow["orig_fips"] = flow["orig_statefips"] + flow["orig_countyfips"]
    flow["dest_fips"] = flow["dest_statefips"] + flow["dest_countyfips"]

    flow["returns"] = pd.to_numeric(flow["returns"], errors="coerce").fillna(0).astype(int)

    return flow[["orig_fips", "dest_fips", "returns"]]

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Download & normalize IRS SOI county-to-county migration flows for selected counties.")
    ap.add_argument("--year_label", required=True, help='IRS year label, e.g. "2010-2011", "2021-2022" (en-dash and "to" are okay).')
    ap.add_argument("--counties", nargs="+", required=True, help="5-digit county FIPS (e.g., 36081 36047 12011 34013 36093)")
    ap.add_argument("--out", default="data/irs", help="Output directory (default: data/irs)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    links = find_county_links(args.year_label)
    if not links:
        raise SystemExit(f"No county inflow/outflow files found for {args.year_label} on IRS pages.")

    if not args.quiet:
        print(f"Found {len(links)} link(s) for {args.year_label}:")
        for u in links:
            print(" -", u)

    frames = []
    for url in links:
        try:
            df_raw = load_csv_or_zip(url)
            df_norm = normalize_cols(df_raw)
            frames.append(df_norm)
            if not args.quiet:
                print(f"Parsed: {url} -> {len(df_norm):,} rows")
        except Exception as e:
            print(f"Skip {url}: {e}", file=sys.stderr)

    if not frames:
        raise SystemExit("No parsable CSV/XLS found.")

    flows = pd.concat(frames, ignore_index=True)

    # Focus set
    focus = set([str(x).zfill(5) for x in args.counties])

    # Net migration matrix among focus counties
    mat = (
        flows[(flows["orig_fips"].isin(focus)) | (flows["dest_fips"].isin(focus))]
        .groupby(["orig_fips", "dest_fips"], as_index=False)["returns"].sum()
    )

    # Top inbound/outbound per focus county
    tops = []
    for f in focus:
        inbound = mat[mat["dest_fips"] == f].sort_values("returns", ascending=False).head(20).copy()
        outbound = mat[mat["orig_fips"] == f].sort_values("returns", ascending=False).head(20).copy()
        inbound["county"] = f; inbound["direction"] = "inbound"
        outbound["county"] = f; outbound["direction"] = "outbound"
        tops.append(pd.concat([inbound, outbound], ignore_index=True))
    tops = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame(columns=["orig_fips","dest_fips","returns","county","direction"])

    # Net for focus counties vs rest: inflow - outflow
    inflow = mat[mat["dest_fips"].isin(focus)].groupby("dest_fips")["returns"].sum().rename("inflow")
    outflow = mat[mat["orig_fips"].isin(focus)].groupby("orig_fips")["returns"].sum().rename("outflow")
    net = pd.concat([inflow, outflow], axis=1).fillna(0).astype(int)
    net["net_migration"] = net["inflow"] - net["outflow"]
    net = net.reset_index().rename(columns={"index": "county_fips", "dest_fips": "county_fips", "orig_fips": "county_fips"})

    # Outputs
    flows.to_csv(outdir / f"irs_flows_{args.year_label}.csv", index=False)
    mat.to_csv(outdir / f"irs_net_matrix_{args.year_label}.csv", index=False)
    tops.to_csv(outdir / f"irs_top_flows_{args.year_label}.csv", index=False)
    net.to_csv(outdir / f"irs_net_summary_{args.year_label}.csv", index=False)

    with open(outdir / "README_irs.txt", "w") as f:
        f.write(
            "IRS SOI county-to-county migration flows extracted for selected counties.\n"
            "Counts reflect returns; schemas vary by year, script normalizes common columns.\n"
            "Notes:\n"
            "- Link finder accepts CSV/XLS/XLSX/ZIP and resolves relative URLs.\n"
            "- Year label matches 'YYYY-YYYY', 'YYYY–YYYY' (en-dash), or 'YYYY to YYYY'.\n"
            "- 'returns' uses N1/Return_Num/Returns/NRET where available; otherwise falls back.\n"
            "- Origin/Destination columns recognized across multiple historical name variants.\n"
        )

    if not args.quiet:
        print(f"\nWrote:")
        print(f" - {outdir / f'irs_flows_{args.year_label}.csv'}")
        print(f" - {outdir / f'irs_net_matrix_{args.year_label}.csv'}")
        print(f" - {outdir / f'irs_top_flows_{args.year_label}.csv'}")
        print(f" - {outdir / f'irs_net_summary_{args.year_label}.csv'}")
        print(f" - {outdir / 'README_irs.txt'}")

if __name__ == "__main__":
    main()
