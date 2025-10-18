# cps_guyanese_indicators.py
# Usage:
#   python cps_guyanese_indicators.py --years 2019 2020 2021 2022 2023 --outdir data/cps --asec
# Env:
#   export IPUMS_API_KEY="YOUR_KEY"
#
# What this does
# - Submits an IPUMS CPS extract (ASEC by default), downloads CSV (.csv.gz) + DDI (.xml)
# - Filters to AGE >= 16 and birthplace label containing "guyan" (e.g., "Guyana", "British Guiana")
# - Computes weighted LFP, unemployment rate, avg usual hours for employed
# - Builds weighted OCC / IND distributions
# - Saves tidy CSVs + JSON README of the extract definition

import os
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from ipumspy import IpumsApiClient, MicrodataExtract, readers

# Optional: hard-code BPL numeric codes for Guyana if you prefer, otherwise we match by label.
BPL_GUYANA_CODES = set()  # e.g., {250}

def build_samples(years, asec=True):
    """CPS ASEC samples are cpsYYYY_03s (March supplement)."""
    if asec:
        return [f"cps{y}_03s" for y in years]
    # If you ever run non-ASEC, provide explicit monthly sample IDs instead.
    return [f"cps{y}_03" for y in years]

def parse_bpl_labels_from_ddi(ddi_path: Path):
    """
    Parse BPL code -> label from the DDI XML directly.
    This is robust even when helper methods don't expose value labels.
    """
    tree = ET.parse(ddi_path)
    root = tree.getroot()
    ns = {'ddi': root.tag.split('}')[0].strip('{')}
    var = root.find(".//ddi:var[@name='BPL']", ns)
    labels = {}
    if var is not None:
        for cat in var.findall(".//ddi:catgry", ns):
            v = cat.find("ddi:catValu", ns)
            l = cat.find("ddi:labl", ns)
            if v is None or l is None:
                continue
            key = v.text
            # Normalize code type
            try:
                key = int(key)
            except Exception:
                pass
            labels[key] = (l.text or "").strip()
    return labels

def make_guyana_mask(df: pd.DataFrame, bpl_labels: dict):
    """Return boolean mask where BPL label suggests Guyana (label contains 'guyan')."""
    if "BPL" not in df.columns:
        return pd.Series(False, index=df.index)
    def lab(v):
        # Handle int vs str codes gracefully
        try:
            return bpl_labels.get(int(v))
        except Exception:
            return bpl_labels.get(v)
    # If user supplied explicit numeric codes, use them first
    if BPL_GUYANA_CODES:
        base_mask = df["BPL"].isin(BPL_GUYANA_CODES)
    else:
        base_mask = df["BPL"].apply(lambda v: isinstance(lab(v), str) and ("guyan" in lab(v).lower()))
    return base_mask.fillna(False)

def tidy_share(sub: pd.DataFrame, var: str, w: str):
    """Weighted share table for a single variable code column."""
    if var not in sub.columns or sub.empty:
        return pd.DataFrame(columns=["var", "code", "weight", "share"])
    t = sub.groupby(var, dropna=False)[w].sum().reset_index(name="weight")
    total = t["weight"].sum()
    if total and total > 0:
        t["share"] = t["weight"] / total
    else:
        t["share"] = 0.0
    t = t.rename(columns={var: "code"})
    t["var"] = var
    return t[["var", "code", "weight", "share"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--outdir", default="data/cps")
    ap.add_argument("--asec", action="store_true", help="Use ASEC samples (cpsYYYY_03s). Recommended.")
    ap.add_argument("--occ_var", default="OCC2010", help="Harmonized CPS occupation variable (default OCC2010).")
    ap.add_argument("--ind_var", default="IND1990", help="Harmonized CPS industry variable (default IND1990).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("IPUMS_API_KEY")
    if not api_key:
        raise RuntimeError("Set IPUMS_API_KEY in your environment.")

    samples = build_samples(args.years, asec=args.asec)

    # Request only valid weight for the chosen sample set
    weight_var = "ASECWT" if args.asec else "WTFINL"

    variables = [
        "YEAR", "STATEFIP", "AGE", "SEX", "BPL",
        "LABFORCE", "EMPSTAT", "UHRSWORKT", args.occ_var, args.ind_var,
        weight_var,
    ]

    # Submit & download extract
    client = IpumsApiClient(api_key)
    extract = MicrodataExtract(
        collection="cps",
        samples=samples,
        variables=variables,
        description=f"CPS Guyanese indicators {args.years}",
        data_format="csv",  # CSV.GZ from API v2
    )
    client.submit_extract(extract)
    client.wait_for_extract(extract)
    client.download_extract(extract, download_dir=outdir)

    # Pair the newest XML with its CSV
    ddi_file = max(outdir.glob("*.xml"), key=lambda p: p.stat().st_mtime)
    ddi = readers.read_ipums_ddi(ddi_file)
    csv_name = getattr(ddi.file_description, "filename", None)
    data_path = outdir / csv_name if csv_name else None
    if not data_path or not data_path.exists():
        # Fallback to the most recent CSV/CSV.GZ
        candidates = list(outdir.glob("*.csv*")) or list(outdir.rglob("*.csv*"))
        if not candidates:
            raise FileNotFoundError("No CSV(.gz) found in the download dir.")
        data_path = max(candidates, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(data_path, compression="infer", low_memory=False)

    # --- Filter: AGE >= 16 & BPL label contains 'guyan'
    bpl_labels = parse_bpl_labels_from_ddi(ddi_file)
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

    guy_mask = make_guyana_mask(df, bpl_labels)
    age_mask = df["AGE"].ge(16)
    sub = df.loc[age_mask & guy_mask].copy()

    print("Guyanese-filtered unweighted N:", int(sub.shape[0]))
    if sub.empty:
        print("Warning: filter returned 0 rows. Check BPL labels in the DDI or set BPL_GUYANA_CODES.")

    # Ensure numerics
    for v in ["LABFORCE", "EMPSTAT", "UHRSWORKT", weight_var]:
        if v in sub.columns:
            sub[v] = pd.to_numeric(sub[v], errors="coerce")

    # Labor force flags
    if "LABFORCE" in sub.columns:
        sub["in_lf"] = sub["LABFORCE"].eq(2)  # 2 = in labor force (IPUMS CPS general code)
    else:
        sub["in_lf"] = sub["EMPSTAT"].isin([1, 2])  # fallback

    sub["employed"] = sub["EMPSTAT"].eq(1)
    sub["unemployed"] = sub["EMPSTAT"].eq(2)

    def wsum(s):
        return (s * sub[weight_var]).sum(skipna=True)

    total_pop = sub[weight_var].sum(skipna=True)
    lf_pop = wsum(sub["in_lf"])
    emp_pop = wsum(sub["employed"])
    unemp_pop = wsum(sub["unemployed"])

    lfp_rate = (lf_pop / total_pop) if pd.notna(total_pop) and total_pop > 0 else None
    unemp_rate = (unemp_pop / lf_pop) if pd.notna(lf_pop) and lf_pop > 0 else None

    # Hours worked among employed with positive UHRSWORKT
    hrs = sub.loc[sub["employed"] & sub["UHRSWORKT"].notna() & (sub["UHRSWORKT"] > 0), ["UHRSWORKT", weight_var]]
    hours_wavg = ((hrs["UHRSWORKT"] * hrs[weight_var]).sum() / hrs[weight_var].sum()) if len(hrs) else None

    # Defend against missing OCC/IND harmonized variables
    occ_var = args.occ_var if args.occ_var in sub.columns else ("OCC" if "OCC" in sub.columns else args.occ_var)
    ind_var = args.ind_var if args.ind_var in sub.columns else ("IND" if "IND" in sub.columns else args.ind_var)

    occ_tbl = tidy_share(sub, occ_var, weight_var)
    ind_tbl = tidy_share(sub, ind_var, weight_var)

    # Outputs
    out_micro = outdir / "cps_guyanese_micro_subset.csv"
    out_sum   = outdir / "cps_guyanese_summary.csv"
    out_occ   = outdir / "cps_guyanese_occ_distribution.csv"
    out_ind   = outdir / "cps_guyanese_ind_distribution.csv"

    keep_cols = ["YEAR", "STATEFIP", "AGE", "SEX", "BPL", "LABFORCE", "EMPSTAT", "UHRSWORKT", occ_var, ind_var, weight_var]
    keep_cols = [c for c in keep_cols if c in sub.columns]
    sub[keep_cols].to_csv(out_micro, index=False)

    summary = pd.DataFrame([{
        "years": "-".join(map(str, args.years)),
        "sample_type": "ASEC" if args.asec else "other",
        "weight": weight_var,
        "lfp_rate": lfp_rate,
        "unemp_rate": unemp_rate,
        "avg_hours_employed": hours_wavg,
        "n_unweighted": int(sub.shape[0]),
        "extract_csv": str(data_path.name),
        "ddi_xml": str(ddi_file.name),
    }])
    summary.to_csv(out_sum, index=False)

    occ_tbl.to_csv(out_occ, index=False)
    ind_tbl.to_csv(out_ind, index=False)

    # README metadata about the extract
    meta = {
        "collection": "cps",
        "samples": samples,
        "variables": variables,
        "filters": {
            "age_min": 16,
            "country_of_birth_filter": "BPL label contains 'guyan' (e.g., 'Guyana', 'British Guiana')",
        },
        "weights_used": weight_var,
        "data_files": {
            "csv": data_path.name,
            "ddi": ddi_file.name
        }
    }
    (outdir / "README_cps_extract.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
