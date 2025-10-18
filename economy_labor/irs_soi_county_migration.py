# irs_soi_county_migration.py
# Usage: python irs_soi_county_migration.py --year_label "2010-2011" --counties 36081 36047 12011 34013 36093 --out data/irs
# Tip: adjust --year_label to match the IRS page label (e.g., "2019-2020", "2021-2022") when available.

import argparse, re, io, requests, zipfile
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

DOWNLOADS_PAGE = "https://www.irs.gov/statistics/soi-tax-stats-migration-data-downloads"

def find_county_links(year_label):
    """Find county-level inflow/outflow CSV links for the given year label."""
    html = requests.get(DOWNLOADS_PAGE).text
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith((".csv",".zip")): 
            continue
        text = (a.get_text(strip=True) or "").lower()
        if "county" in text and year_label.lower() in text:
            links.append(href)
    # Also follow 'County-to-county migration data files' page for historical links
    hist = "https://www.irs.gov/statistics/soi-tax-stats-county-to-county-migration-data-files"
    html2 = requests.get(hist).text
    soup2 = BeautifulSoup(html2, "html.parser")
    for a in soup2.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text(strip=True) or "").lower()
        if "county" in text and year_label.lower() in text and href.lower().endswith(".csv"):
            links.append(href)
    return list(dict.fromkeys(links))  # dedupe

def load_csv_or_zip(url):
    r = requests.get(url)
    r.raise_for_status()
    if url.lower().endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # Pick first CSV inside
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                return pd.read_csv(z.open(name), dtype=str)
        raise ValueError("No CSV found inside ZIP.")
    else:
        return pd.read_csv(io.StringIO(r.text), dtype=str)

def normalize_cols(df):
    # Try to infer key columns across schema variants
    cols = {c.lower(): c for c in df.columns}
    # Typical columns (historical): y1_statefips, y1_countyfips, y2_statefips, y2_countyfips, n1 (returns), n2 (exempts), agi
    guess = {}
    for k in ["y1_statefips","y1_countyfips","y2_statefips","y2_countyfips","statefips1","countyfips1","statefips2","countyfips2"]:
        for c in cols:
            if c.startswith(k): guess[k] = cols[c]
    # Flow counts
    n_vars = [c for c in df.columns if re.fullmatch(r"(n1|n2|returns|exemptions|nret)", c, flags=re.I)]
    if not n_vars:
        # many files have 'n1' (returns) and 'n2' (exemptions)
        n_vars = [c for c in df.columns if c.lower() in ("n1","n2")]
    # Normalize: keep returns as flow_count
    if "n1" in [x.lower() for x in n_vars]:
        flow = df[[guess.get("y1_statefips") or guess.get("statefips1"),
                   guess.get("y1_countyfips") or guess.get("countyfips1"),
                   guess.get("y2_statefips") or guess.get("statefips2"),
                   guess.get("y2_countyfips") or guess.get("countyfips2"),
                   [c for c in df.columns if c.lower()=="n1"][0]]].copy()
        flow.columns = ["orig_statefips","orig_countyfips","dest_statefips","dest_countyfips","returns"]
    else:
        # fallback: first numeric column as count
        numcols = df.select_dtypes(include="object").columns
        flow = df.copy()
        flow["returns"] = pd.to_numeric(flow.get(n_vars[0], 0), errors="coerce")

    for c in ["orig_statefips","orig_countyfips","dest_statefips","dest_countyfips"]:
        flow[c] = flow[c].astype(str).str.zfill(2 if "state" in c else 3)

    flow["orig_fips"] = flow["orig_statefips"] + flow["orig_countyfips"]
    flow["dest_fips"] = flow["dest_statefips"] + flow["dest_countyfips"]
    flow["returns"] = pd.to_numeric(flow["returns"], errors="coerce").fillna(0).astype(int)
    return flow[["orig_fips","dest_fips","returns"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year_label", required=True, help='IRS year label exactly as on site, e.g. "2010-2011", "2021-2022"')
    ap.add_argument("--counties", nargs="+", required=True, help="5-digit county FIPS (e.g., 36081 36047 12011 34013 36093)")
    ap.add_argument("--out", default="data/irs")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    links = find_county_links(args.year_label)
    if not links:
        raise SystemExit(f"No county inflow/outflow CSVs found for {args.year_label} on IRS pages.")

    frames = []
    for url in links:
        try:
            df = load_csv_or_zip(url)
            frames.append(normalize_cols(df))
        except Exception as e:
            print(f"Skip {url}: {e}")

    if not frames:
        raise SystemExit("No parsable CSVs found.")
    flows = pd.concat(frames, ignore_index=True)

    focus = set([str(x).zfill(5) for x in args.counties])

    # Net migration matrix among focus counties
    mat = (flows[flows["orig_fips"].isin(focus) | flows["dest_fips"].isin(focus)]
           .groupby(["orig_fips","dest_fips"])["returns"].sum().reset_index())

    # For top origin/destination relative to each focus county
    tops = []
    for f in focus:
        inbound = mat[mat["dest_fips"]==f].sort_values("returns", ascending=False).head(20)
        outbound = mat[mat["orig_fips"]==f].sort_values("returns", ascending=False).head(20)
        inbound["county"] = f; inbound["direction"] = "inbound"
        outbound["county"] = f; outbound["direction"] = "outbound"
        tops.append(pd.concat([inbound, outbound], ignore_index=True))
    tops = pd.concat(tops, ignore_index=True)

    # Net for focus counties vs rest: inflow - outflow
    inflow = mat[mat["dest_fips"].isin(focus)].groupby("dest_fips")["returns"].sum().rename("inflow")
    outflow = mat[mat["orig_fips"].isin(focus)].groupby("orig_fips")["returns"].sum().rename("outflow")
    net = pd.concat([inflow, outflow], axis=1).fillna(0).astype(int)
    net["net_migration"] = net["inflow"] - net["outflow"]
    net = net.reset_index().rename(columns={"index":"county_fips","dest_fips":"county_fips","orig_fips":"county_fips"})

    # Outputs
    flows.to_csv(outdir / f"irs_flows_{args.year_label}.csv", index=False)
    mat.to_csv(outdir / f"irs_net_matrix_{args.year_label}.csv", index=False)
    tops.to_csv(outdir / f"irs_top_flows_{args.year_label}.csv", index=False)
    net.to_csv(outdir / f"irs_net_summary_{args.year_label}.csv", index=False)

    with open(outdir / "README_irs.txt","w") as f:
        f.write(
            "IRS SOI county-to-county migration flows extracted for selected counties.\n"
            "Counts reflect returns; schemas vary by year, script normalizes common columns.\n"
        )

if __name__ == "__main__":
    main()
