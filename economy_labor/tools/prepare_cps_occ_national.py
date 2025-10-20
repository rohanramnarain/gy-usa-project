#!/usr/bin/env python3
"""
prepare_cps_occ_national.py
---------------------------
Turn a NATIONAL CPS occupation mix (OCC2010 codes, no hub column) into a
hub-level SOC mix that the econ_labor_pipeline can consume.

Inputs (defaults):
  • economy_labor/data/cps/cps_guyanese_occ_distribution.csv
      columns like: var, code, weight, share
      (var is ignored; "code" is OCC2010)
  • economy_labor/data/cps/occ2010_to_soc2018.csv
      columns: occ2010, soc_code (fill soc_code yourself; blanks will be dropped)

Output:
  • economy_labor/data/cps/cps_guyanese_occ_distribution_hub_soc.csv
      columns: hub, soc_code, share (replicated across hubs)

Run:
  python economy_labor/tools/prepare_cps_occ_national.py \
    --hubs NYC_MSA Miami_MSA NJ_State GA_State MD_State

Then run the pipeline using the prepared file:
  python economy_labor/econ_labor_pipeline.py --all \
    --cps-occ-file economy_labor/data/cps/cps_guyanese_occ_distribution_hub_soc.csv \
    --cps-hub-col hub --cps-soc-col soc_code --cps-share-col share
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="economy_labor/data/cps/cps_guyanese_occ_distribution.csv")
    ap.add_argument("--crosswalk", default="economy_labor/data/cps/occ2010_to_soc2018.csv")
    ap.add_argument("--outfile", default="economy_labor/data/cps/cps_guyanese_occ_distribution_hub_soc.csv")
    ap.add_argument("--occ-col", default="code", help="Column in infile that has OCC2010 code (e.g., 'code').")
    ap.add_argument("--share-col", default="share", help="Share column (0..1 or %).")
    ap.add_argument("--weight-col", default="weight", help="Optional weighted count column (ignored; share used).")
    ap.add_argument(
        "--hubs",
        nargs="*",
        default=["NYC_MSA", "Miami_MSA", "NJ_State", "GA_State", "MD_State"],
        help="Hub keys to replicate the national mix across (must match pipeline HUBS).",
    )
    args = ap.parse_args()

    src = Path(args.infile)
    cwpath = Path(args.crosswalk)
    outp = Path(args.outfile)

    if not src.exists():
        raise SystemExit(f"Input not found: {src}")
    if not cwpath.exists():
        raise SystemExit(
            f"Crosswalk not found: {cwpath}. Create it with columns occ2010,soc_code and fill soc_code values."
        )

    df = pd.read_csv(src)
    cw = pd.read_csv(cwpath)

    # Identify crosswalk columns
    occ_cw = "occ2010" if "occ2010" in cw.columns else next((c for c in cw.columns if c.lower().startswith("occ")), None)
    soc_cw = "soc_code" if "soc_code" in cw.columns else next((c for c in cw.columns if c.lower().startswith("soc")), None)
    if occ_cw is None or soc_cw is None:
        raise SystemExit("Crosswalk must include columns occ2010 and soc_code (or close variants).")

    # Normalize types
    df["occ2010"] = df[args.occ_col].astype(str).str.extract(r"(\d+)")[0]
    cw["occ2010"] = cw[occ_cw].astype(str).str.extract(r"(\d+)")[0]
    cw["soc_code"] = cw[soc_cw].astype(str)

    # Join to map OCC2010 -> SOC
    j = df.merge(cw[["occ2010", "soc_code"]], on="occ2010", how="left")

    # Clean shares
    j["share"] = pd.to_numeric(j[args.share_col], errors="coerce")
    if j["share"].median() and j["share"].median() > 1.5:
        j["share"] = j["share"] / 100.0

    before = len(j)
    j = j[j["soc_code"].notna() & j["share"].notna()].copy()
    mapped = len(j)

    if mapped == 0:
        raise SystemExit("No rows mapped. Fill some soc_code values in the crosswalk and rerun.")

    # Replicate across hubs
    out_frames = [j.assign(hub=h) for h in args.hubs]
    out = pd.concat(out_frames, ignore_index=True)
    out = out[["hub", "soc_code", "share"]]

    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"[ok] wrote {outp} → {len(out)} rows across hubs={args.hubs}\n"
          f"     mapped {mapped}/{before} OCC2010 rows via crosswalk (unmapped rows dropped)")


if __name__ == "__main__":
    main()
