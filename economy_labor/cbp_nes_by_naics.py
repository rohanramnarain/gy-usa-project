# cbp_nes_by_naics.py
# Usage: python cbp_nes_by_naics.py --counties 36081 36047 12011 34013 36093 --cbp_year 2023 --nes_year 2022 --out data/cbp_nes

import argparse, json, re, requests
import pandas as pd
from pathlib import Path

CBP_BASE = "https://api.census.gov/data/{year}/cbp"
NES_BASE = "https://api.census.gov/data/timeseries/nonemp"

TARGET_SECTORS = {
    "Retail Trade": re.compile(r"^44-45|^446|^447|^448"),
    "Food Services": re.compile(r"^72|^722"),
    "Transportation/Warehousing": re.compile(r"^48-49|^485|^484|^492|^493"),
}

def get_vars_json(url):
    v = requests.get(url.rstrip("/") + "/variables.json").json()
    return {k: v["variables"][k] for k in v["variables"]}

def infer_cbp_count_var(vars_json):
    # CBP typically: ESTAB = number of establishments
    for k, meta in vars_json.items():
        if meta.get("label","").lower().startswith("establishments"):
            return k
    return "ESTAB"

def infer_nes_count_var(vars_json):
    # Nonemployer often has "N" (Number of firms) or 'RCPTOT' for receipts
    # Prefer count of firms
    candidates = []
    for k, meta in vars_json.items():
        lab = meta.get("label","").lower()
        if "number of firms" in lab or "number of establishments" in lab:
            candidates.append(k)
    if candidates:
        return candidates[0]
    # fallback common code
    return "N"

def split_fips(fips5):
    fips = str(fips5).zfill(5)
    return fips[:2], fips[2:]

def fetch_cbp_county(year, fips5, count_var):
    s, c = split_fips(fips5)
    get = ["NAME","NAICS2017",count_var,"state","county"]
    params = {"get": ",".join(get), "for": f"county:{c}", "in": f"state:{s}"}
    url = CBP_BASE.format(year=year)
    r = requests.get(url, params=params); r.raise_for_status()
    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    df[count_var] = pd.to_numeric(df[count_var], errors="coerce")
    return df

def fetch_nes_county(year, fips5, count_var):
    s, c = split_fips(fips5)
    get = ["NAME","NAICS2017",count_var,"time","state","county"]
    params = {"get": ",".join(get), "for": f"county:{c}", "in": f"state:{s}", "time": str(year)}
    r = requests.get(NES_BASE, params=params); r.raise_for_status()
    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    df[count_var] = pd.to_numeric(df[count_var], errors="coerce")
    return df

def flag_sector(naics):
    for sect, pat in TARGET_SECTORS.items():
        if pat.search(str(naics)):
            return sect
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counties", nargs="+", required=True, help="5-digit county FIPS (e.g., 36081 36047 12011 34013 36093)")
    ap.add_argument("--cbp_year", type=int, default=2023)
    ap.add_argument("--nes_year", type=int, default=2022)
    ap.add_argument("--out", default="data/cbp_nes")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Discover count variable names
    cbp_vars = get_vars_json(CBP_BASE.format(year=args.cbp_year))
    cbp_count = infer_cbp_count_var(cbp_vars)
    nes_vars = get_vars_json(NES_BASE)
    nes_count = infer_nes_count_var(nes_vars)

    cbp_all, nes_all = [], []
    for f in args.counties:
        cbp = fetch_cbp_county(args.cbp_year, f, cbp_count)
        cbp["county_fips"] = str(f).zfill(5)
        cbp_all.append(cbp)

        nes = fetch_nes_county(args.nes_year, f, nes_count)
        nes["county_fips"] = str(f).zfill(5)
        nes_all.append(nes)

    cbp_all = pd.concat(cbp_all, ignore_index=True)
    nes_all = pd.concat(nes_all, ignore_index=True)

    # Summaries
    cbp_all["sector_flag"] = cbp_all["NAICS2017"].map(flag_sector)
    nes_all["sector_flag"] = nes_all["NAICS2017"].map(flag_sector)

    cbp_sum = (cbp_all.groupby(["county_fips","NAICS2017","sector_flag"], dropna=False)[cbp_count]
               .sum().reset_index().rename(columns={cbp_count: "establishments"}))
    nes_sum = (nes_all.groupby(["county_fips","NAICS2017","sector_flag"], dropna=False)[nes_count]
               .sum().reset_index().rename(columns={nes_count: "nonemployers"}))

    # Tidy join + highlight target sectors
    tidy = cbp_sum.merge(nes_sum, on=["county_fips","NAICS2017","sector_flag"], how="outer")
    tidy["is_target_sector"] = tidy["sector_flag"].notna()

    # Outputs
    cbp_all.to_csv(outdir / f"cbp_raw_{args.cbp_year}.csv", index=False)
    nes_all.to_csv(outdir / f"nes_raw_{args.nes_year}.csv", index=False)
    tidy.to_csv(outdir / f"cbp_nes_summary_{args.cbp_year}_{args.nes_year}.csv", index=False)

    with open(outdir / "README_cbp_nes.txt","w") as f:
        f.write(
            "CBP (establishments) and NES (nonemployers) by NAICS, county-level.\n"
            "Sectors flagged: retail, food services, transportation/warehousing.\n"
            "These are CONTEXT indicators and NOT nationality-specific.\n"
        )

if __name__ == "__main__":
    main()
