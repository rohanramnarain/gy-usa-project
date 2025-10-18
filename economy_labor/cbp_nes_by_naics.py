# cbp_nes_by_naics.py
# Usage:
#   python cbp_nes_by_naics.py --counties 36081 36047 12011 34013 36093 \
#     --cbp_year 2023 --nes_year 2022 --out data/cbp_nes [--api_key YOUR_KEY]

import argparse, re, requests
import pandas as pd
from pathlib import Path

CBP_BASE_TPL = "https://api.census.gov/data/{year}/cbp"
NES_BASE_TPL = "https://api.census.gov/data/{year}/nonemp"  # year-based (not /timeseries)

TARGET_SECTORS = {
    "Retail Trade": re.compile(r"^(44-45|446|447|448)"),
    "Food Services": re.compile(r"^(72|722)"),
    "Transportation/Warehousing": re.compile(r"^(48-49|485|484|492|493)"),
}

REQ_TIMEOUT = 60  # seconds


def get_vars_json(url, api_key=None):
    params = {"key": api_key} if api_key else None
    r = requests.get(url.rstrip("/") + "/variables.json", params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j["variables"]


def infer_cbp_count_var(vars_json):
    # CBP typically uses ESTAB (number of establishments)
    for k, meta in vars_json.items():
        if meta.get("label", "").strip().lower().startswith("number of establishments"):
            return k
    return "ESTAB"


def infer_nes_count_var(vars_json):
    # NES uses NESTAB (number of nonemployer establishments)
    for k, meta in vars_json.items():
        lab = meta.get("label", "").strip().lower()
        if "number of nonemployer establishments" in lab:
            return k
    return "NESTAB"


def infer_naics_var(vars_json):
    # Prefer explicit NAICSYYYY; choose the most recent year if multiple
    cands = [k for k in vars_json if re.fullmatch(r"NAICS\d{4}", k or "")]
    if cands:
        cands.sort(key=lambda k: int(re.search(r"\d{4}", k).group()), reverse=True)
        return cands[0]
    # Fallbacks sometimes seen
    for k in ("NAICS2022", "NAICS2017", "NAICS2012", "NAICS2007", "NAICS"):
        if k in vars_json:
            return k
    raise KeyError("No NAICS variable found in dataset metadata")


def split_fips(fips5):
    fips = str(fips5).zfill(5)
    return fips[:2], fips[2:]


def _request_json(url, params, api_key=None):
    p = dict(params or {})
    if api_key:
        p["key"] = api_key
    r = requests.get(url, params=p, timeout=REQ_TIMEOUT)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        detail = f"{url}?{r.request.body or r.request.url.split('?',1)[-1]}"
        raise SystemExit(f"HTTP error {r.status_code} for: {detail}") from e
    try:
        return r.json()
    except ValueError as e:
        raise SystemExit(f"Non-JSON response from {url}. First 200 chars:\n{r.text[:200]}") from e


def fetch_cbp_county(year, fips5, count_var, naics_var, api_key=None):
    base = CBP_BASE_TPL.format(year=year)
    s, c = split_fips(fips5)

    # Keep 'get' minimal; the API always appends the geo dims ('state','county') at the end
    get = ["NAME", naics_var, count_var]

    params = {
        "get": ",".join(get),
        "for": f"county:{c}",
        "in": f"state:{s}",
        # Needed filters to avoid 400s on CBP 2023
        "LFO": "001",       # all legal forms
        "EMPSZES": "001",   # all establishment sizes
        # Predicate for NAICS — "*" returns all codes
        naics_var: "*",
    }

    data = _request_json(base, params, api_key=api_key)
    if not data or len(data) < 2:
        return pd.DataFrame(columns=get + ["state", "county"])

    df = pd.DataFrame(data[1:], columns=data[0])
    # Ensure numeric
    df[count_var] = pd.to_numeric(df[count_var], errors="coerce")
    return df


def fetch_nes_county(year, fips5, count_var, naics_var, api_key=None):
    base = NES_BASE_TPL.format(year=year)
    s, c = split_fips(fips5)

    # Similar approach: keep 'get' minimal
    get = ["NAME", naics_var, count_var]
    params = {"get": ",".join(get), "for": f"county:{c}", "in": f"state:{s}"}
    # NOTE: Year-specific NES endpoints should NOT use a 'time' parameter

    data = _request_json(base, params, api_key=api_key)
    if not data or len(data) < 2:
        return pd.DataFrame(columns=get + ["state", "county"])

    df = pd.DataFrame(data[1:], columns=data[0])
    df[count_var] = pd.to_numeric(df[count_var], errors="coerce")
    return df


def flag_sector(naics_code):
    s = str(naics_code)
    for sect, pat in TARGET_SECTORS.items():
        if pat.search(s):
            return sect
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counties", nargs="+", required=True,
                    help="5-digit county FIPS (e.g., 36081 36047 12011 34013 36093)")
    ap.add_argument("--cbp_year", type=int, default=2023)
    ap.add_argument("--nes_year", type=int, default=2022)
    ap.add_argument("--out", default="data/cbp_nes")
    ap.add_argument("--api_key", default=None, help="Optional Census API key")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover variable names per dataset
    cbp_vars = get_vars_json(CBP_BASE_TPL.format(year=args.cbp_year), api_key=args.api_key)
    nes_vars = get_vars_json(NES_BASE_TPL.format(year=args.nes_year), api_key=args.api_key)

    cbp_count = infer_cbp_count_var(cbp_vars)   # typically ESTAB
    nes_count = infer_nes_count_var(nes_vars)   # typically NESTAB
    cbp_naics = infer_naics_var(cbp_vars)       # e.g., NAICS2017 for 2023 CBP
    nes_naics = infer_naics_var(nes_vars)       # e.g., NAICS2022 for 2022 NES

    cbp_rows, nes_rows = [], []
    for f in args.counties:
        cbp = fetch_cbp_county(args.cbp_year, f, cbp_count, cbp_naics, api_key=args.api_key)
        cbp["county_fips"] = str(f).zfill(5)
        cbp_rows.append(cbp)

        nes = fetch_nes_county(args.nes_year, f, nes_count, nes_naics, api_key=args.api_key)
        nes["county_fips"] = str(f).zfill(5)
        nes_rows.append(nes)

    cbp_all = pd.concat(cbp_rows, ignore_index=True) if cbp_rows else pd.DataFrame()
    nes_all = pd.concat(nes_rows, ignore_index=True) if nes_rows else pd.DataFrame()

    # Normalize NAICS column to a collision-proof name and remove dup columns if any
    if not cbp_all.empty:
        cbp_all = cbp_all.rename(columns={cbp_naics: "NAICS_CODE"})
        cbp_all = cbp_all.loc[:, ~cbp_all.columns.duplicated(keep="first")]
        # The API may return both 'state'/'county' (lowercase) appended; keep just these
        cbp_all.drop(columns=["STATE", "COUNTY"], errors="ignore", inplace=True)

    if not nes_all.empty:
        nes_all = nes_all.rename(columns={nes_naics: "NAICS_CODE"})
        nes_all = nes_all.loc[:, ~nes_all.columns.duplicated(keep="first")]
        nes_all.drop(columns=["STATE", "COUNTY"], errors="ignore", inplace=True)

    # Summaries
    if not cbp_all.empty:
        cbp_all["sector_flag"] = cbp_all["NAICS_CODE"].astype(str).map(flag_sector)
    if not nes_all.empty:
        nes_all["sector_flag"] = nes_all["NAICS_CODE"].astype(str).map(flag_sector)

    cbp_sum = (
        cbp_all.groupby(["county_fips", "NAICS_CODE", "sector_flag"], dropna=False)[cbp_count]
        .sum()
        .reset_index()
        .rename(columns={cbp_count: "establishments"})
    ) if not cbp_all.empty else pd.DataFrame(columns=["county_fips", "NAICS_CODE", "sector_flag", "establishments"])

    nes_sum = (
        nes_all.groupby(["county_fips", "NAICS_CODE", "sector_flag"], dropna=False)[nes_count]
        .sum()
        .reset_index()
        .rename(columns={nes_count: "nonemployers"})
    ) if not nes_all.empty else pd.DataFrame(columns=["county_fips", "NAICS_CODE", "sector_flag", "nonemployers"])

    # Tidy join + highlight target sectors
    if not cbp_sum.empty or not nes_sum.empty:
        tidy = cbp_sum.merge(nes_sum, on=["county_fips", "NAICS_CODE", "sector_flag"], how="outer")
        tidy["is_target_sector"] = tidy["sector_flag"].notna()
    else:
        tidy = pd.DataFrame(columns=["county_fips", "NAICS_CODE", "sector_flag", "establishments", "nonemployers", "is_target_sector"])

    # Outputs
    if not cbp_all.empty:
        cbp_all.to_csv(outdir / f"cbp_raw_{args.cbp_year}.csv", index=False)
    if not nes_all.empty:
        nes_all.to_csv(outdir / f"nes_raw_{args.nes_year}.csv", index=False)
    tidy.to_csv(outdir / f"cbp_nes_summary_{args.cbp_year}_{args.nes_year}.csv", index=False)

    with open(outdir / "README_cbp_nes.txt", "w") as f:
        f.write(
            "CBP (establishments) and NES (nonemployers) by NAICS, county-level.\n"
            "Sectors flagged: retail, food services, transportation/warehousing.\n"
            "Notes:\n"
            "- CBP uses LFO=001 and EMPSZES=001 with a wildcard NAICS predicate to avoid 400 errors.\n"
            "- NAICS columns normalized to NAICS_CODE.\n"
        )


if __name__ == "__main__":
    main()
