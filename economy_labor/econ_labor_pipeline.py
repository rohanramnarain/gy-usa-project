#!/usr/bin/env python3
"""
econ_labor_pipeline.py
----------------------
Stitches together your economy/labor datasets to profile Guyanese in key hubs.

Inputs (expected under economy_labor/data/...):
  - cps/
      cps_guyanese_occ_distribution.csv        # Guyanese occupational mix by hub (preferred)
      cps_guyanese_ind_distribution.csv        # Guyanese industry mix by hub (optional)
      cps_guyanese_summary.csv                 # Denominators by hub (optional)
      cps_guyanese_micro_subset.csv            # Fallback micro subset if mix files missing (optional)
  - oews/
      oews_metro_median_wages_2024.csv         # MSA x SOC: median wages; if available: tot_emp
  - cbp_nes/
      cbp_raw_2023.csv                         # County x NAICS (establishments, emp, etc.)
      nes_raw_2022.csv                         # County x NAICS (nonemployers)
  - irs/
      irs_flows_2010-2011.csv                  # county-to-county flows (origin,dest,flows)

Outputs (under economy_labor/outputs/):
  - cps_occ_lq_by_hub.csv
  - cps_mix_expected_wage_vs_oews.csv
  - cbp_nes_intensity_by_hub.csv
  - irs_netflows_to_hub.csv

Figures (under economy_labor/figures/):
  - lq_top10_<hub>.png                 # if LQ computable
  - expected_wage_vs_overall_<hub>.png
  - cbp_nes_top_naics_<hub>.png
  - irs_top_origins_<hub>.png

Run:
  python economy_labor/econ_labor_pipeline.py --all
  # or specify hubs (CBSA and/or state):
  python economy_labor/econ_labor_pipeline.py --hubs NYC_MSA Miami_MSA NJ_State

Requirements:
  pandas, matplotlib (no geopandas needed here)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ----------------------------
# Paths
# ----------------------------
BASE = Path("economy_labor")
DATA = BASE / "data"
OUT = BASE / "outputs"
FIG = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Config: hubs (edit as you like)
# ----------------------------
HUBS = {
    # Metro hubs (CBSA/MSA)
    "NYC_MSA": {
        "cbsa": "35620",
        "name": "New York-Newark-Jersey City, NY-NJ-PA",
        "counties": ["36005","36047","36061","36081","36085","34003","34017","34031","34039","34041","34013","34019"]  # NYC+NJ fringe (editable)
    },
    "Miami_MSA": {
        "cbsa": "33100",
        "name": "Miami-Fort Lauderdale-West Palm Beach, FL",
        "counties": ["12011","12086","12099"]
    },
    # State hubs
    "NJ_State": {"state_fips": "34", "name": "New Jersey"},
    "GA_State": {"state_fips": "13", "name": "Georgia"},
    "MD_State": {"state_fips": "24", "name": "Maryland"},
}

# ----------------------------
# Helpers
# ----------------------------
def _fmt_thousands(ax, axis="y"):
    if axis == "x":
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

def _savefig(path: Path, tight=True):
    try:
        if tight:
            plt.gcf().tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {path}")

def _possible(cols, *cands):
    cand_lower = {c.lower() for c in cands}
    for c in cols:
        if c.lower() in cand_lower:
            return c
    return None

def _ensure_str_zfill(series, width):
    return series.astype(str).str.zfill(width)

def _to_2digit_naics(s):
    s = str(s)
    return s[:2] if s and s != "None" else None

# ----------------------------
# Loaders (defensive)
# ----------------------------
def load_cps_occ_mix() -> pd.DataFrame | None:
    path = DATA / "cps" / "cps_guyanese_occ_distribution.csv"
    if not path.exists():
        warnings.warn("Missing cps_guyanese_occ_distribution.csv; LQ/expected wage will be limited.")
        return None
    df = pd.read_csv(path)
    # Try to normalize column names
    hub = _possible(df.columns, "hub", "cbsa", "state_fips", "state", "msa", "region")
    soc = _possible(df.columns, "soc", "soc_code", "occupation_code")
    share = _possible(df.columns, "share", "guyanese_share", "occ_share")
    wt = _possible(df.columns, "weighted_count", "wt_count", "count_weighted")
    title = _possible(df.columns, "soc_title", "occupation", "occ_title")
    if not (hub and soc and share):
        raise ValueError("cps_guyanese_occ_distribution.csv must include hub, SOC code, and share columns.")
    keep = [c for c in [hub, soc, share, wt, title] if c]
    df = df[keep].copy()
    df.rename(columns={hub:"hub_id", soc:"soc_code", share:"guyanese_share"}, inplace=True)
    if wt: df.rename(columns={wt:"guyanese_weighted_count"}, inplace=True)
    if title: df.rename(columns={title:"soc_title"}, inplace=True)
    # Clean SOC format (e.g., 15-1256)
    df["soc_code"] = df["soc_code"].astype(str)
    return df

def load_cps_summary() -> pd.DataFrame | None:
    path = DATA / "cps" / "cps_guyanese_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hub = _possible(df.columns, "hub", "cbsa", "state_fips", "state", "msa", "region")
    total = _possible(df.columns, "weighted_total", "guyanese_total", "total_weighted", "n_weighted")
    if hub and total:
        return df[[hub, total]].rename(columns={hub:"hub_id", total:"guyanese_total"})
    return None

def load_oews() -> pd.DataFrame:
    path = DATA / "oews" / "oews_metro_median_wages_2024.csv"
    if not path.exists():
        raise FileNotFoundError("Missing oews_metro_median_wages_2024.csv in economy_labor/data/oews/")
    df = pd.read_csv(path)
    cbsa = _possible(df.columns, "cbsa", "msa", "area_code", "msa_cbsa")
    soc = _possible(df.columns, "soc_code", "soc", "occ_code")
    med = _possible(df.columns, "median_wage", "h_median", "a_median", "median_annual_wage")
    title = _possible(df.columns, "soc_title", "occupation_title", "occ_title")
    tot_emp = _possible(df.columns, "tot_emp", "employment", "emp")
    name = _possible(df.columns, "msa_name", "area_name", "cbsa_name")
    keep = [c for c in [cbsa, name, soc, title, med, tot_emp] if c]
    df = df[keep].copy()
    df.rename(columns={cbsa:"cbsa", soc:"soc_code", med:"median_wage"}, inplace=True)
    if title: df.rename(columns={title:"soc_title"}, inplace=True)
    if name:  df.rename(columns={name:"msa_name"}, inplace=True)
    if tot_emp: df.rename(columns={tot_emp:"tot_emp"}, inplace=True)
    df["cbsa"] = df["cbsa"].astype(str).str.extract(r"(\d+)")[0]
    df["soc_code"] = df["soc_code"].astype(str)
    return df

def load_cbp() -> pd.DataFrame | None:
    path = DATA / "cbp_nes" / "cbp_raw_2023.csv"
    if not path.exists():
        warnings.warn("Missing cbp_raw_2023.csv; CBP indicators will be skipped.")
        return None
    df = pd.read_csv(path)
    # Expect county FIPS + NAICS
    county = _possible(df.columns, "county_fips", "fips", "county", "geoid")
    naics = _possible(df.columns, "naics", "naics_code")
    est = _possible(df.columns, "establishments", "est", "estab")
    emp = _possible(df.columns, "employment", "emp")
    keep = [c for c in [county, naics, est, emp] if c]
    df = df[keep].copy()
    df.rename(columns={county:"county_fips", naics:"naics", est:"establishments"}, inplace=True)
    df["county_fips"] = _ensure_str_zfill(df["county_fips"], 5)
    df["naics2"] = df["naics"].map(_to_2digit_naics)
    return df

def load_nes() -> pd.DataFrame | None:
    path = DATA / "cbp_nes" / "nes_raw_2022.csv"
    if not path.exists():
        warnings.warn("Missing nes_raw_2022.csv; NES indicators will be skipped.")
        return None
    df = pd.read_csv(path)
    county = _possible(df.columns, "county_fips", "fips", "county", "geoid")
    naics = _possible(df.columns, "naics", "naics_code")
    firms = _possible(df.columns, "firms", "nonemployers", "num_firms")
    keep = [c for c in [county, naics, firms] if c]
    df = df[keep].copy()
    df.rename(columns={county:"county_fips", naics:"naics", firms:"nonemployers"}, inplace=True)
    df["county_fips"] = _ensure_str_zfill(df["county_fips"], 5)
    df["naics2"] = df["naics"].map(_to_2digit_naics)
    return df

def load_irs_flows() -> pd.DataFrame | None:
    path = DATA / "irs" / "irs_flows_2010-2011.csv"
    if not path.exists():
        warnings.warn("Missing irs_flows_2010-2011.csv; IRS flow charts will be skipped.")
        return None
    df = pd.read_csv(path)
    orig = _possible(df.columns, "origin_fips", "origin", "from_fips")
    dest = _possible(df.columns, "dest_fips", "destination", "to_fips")
    n = _possible(df.columns, "flows", "num_returns", "n_returns", "exemptions")
    keep = [c for c in [orig, dest, n] if c]
    df = df[keep].copy()
    df.rename(columns={orig:"origin_fips", dest:"dest_fips", n:"flows"}, inplace=True)
    df["origin_fips"] = _ensure_str_zfill(df["origin_fips"], 5)
    df["dest_fips"] = _ensure_str_zfill(df["dest_fips"], 5)
    return df

# ----------------------------
# Analyses
# ----------------------------
def run_occ_expected_wage_and_lq(hubs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cps_occ = load_cps_occ_mix()
    oews = load_oews()

    if cps_occ is None:
        raise SystemExit("Cannot compute occupation analyses without cps_guyanese_occ_distribution.csv")

    # Identify which rows correspond to CBSA vs State hubs
    out_mix = []
    for hub in hubs:
        conf = HUBS[hub]
        # Filter CPS by hub id (supports cbsa or state_fips or a direct 'hub' code)
        if "cbsa" in conf:
            mask = cps_occ["hub_id"].astype(str).str.contains(conf["cbsa"])
        elif "state_fips" in conf:
            mask = cps_occ["hub_id"].astype(str).str.zfill(2).str.contains(conf["state_fips"].zfill(2))
        else:
            mask = cps_occ["hub_id"].astype(str).str.contains(hub)
        sub = cps_occ[mask].copy()
        if sub.empty:
            warnings.warn(f"CPS mix has no rows for hub={hub}. Check hub_id values.")
            continue
        sub["hub"] = hub
        out_mix.append(sub)
    if not out_mix:
        raise SystemExit("No CPS rows matched any selected hub.")
    mix = pd.concat(out_mix, ignore_index=True)

    # Expected wage: join with OEWS median wages
    if "cbsa" in HUBS[hubs[0]]:
        # Metro expected wage only for CBSA hubs
        cbsa_hubs = [h for h in hubs if "cbsa" in HUBS[h]]
        frames = []
        for hub in cbsa_hubs:
            conf = HUBS[hub]
            m = mix[mix["hub"] == hub]
            oo = oews[oews["cbsa"] == conf["cbsa"]]
            j = m.merge(oo, how="inner", on=["soc_code"])
            if j.empty:
                warnings.warn(f"No SOC overlap for expected wage in hub={hub}.")
                continue
            j["mix_expected_wage"] = j["guyanese_share"] * j["median_wage"]
            # Overall MSA median (simple median across SOC medians)
            overall_msa_median = oo["median_wage"].median()
            frames.append(j.assign(overall_msa_median=overall_msa_median))
        expw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        expw = pd.DataFrame()

    # LQ: need an "all workers" baseline share; use OEWS tot_emp if present
    lq_frames = []
    if "tot_emp" in oews.columns:
        for hub in hubs:
            conf = HUBS[hub]
            if "cbsa" not in conf:
                continue   # LQ only for metro if we have OEWS by CBSA
            m = mix[mix["hub"] == hub]
            oo = oews[oews["cbsa"] == conf["cbsa"]].copy()
            if oo.empty:
                continue
            # compute all-worker share by SOC in that MSA
            if "tot_emp" not in oo.columns or oo["tot_emp"].isna().all():
                continue
            msa_total_emp = oo["tot_emp"].sum()
            oo["allworkers_share"] = oo["tot_emp"] / msa_total_emp
            j = m.merge(oo[["soc_code","soc_title","allworkers_share"]], on="soc_code", how="inner")
            if j.empty:
                continue
            j["lq"] = j["guyanese_share"] / j["allworkers_share"].replace(0, pd.NA)
            j["hub"] = hub
            lq_frames.append(j[["hub","soc_code","soc_title","guyanese_share","allworkers_share","lq"]].copy())
    cps_lq = pd.concat(lq_frames, ignore_index=True) if lq_frames else pd.DataFrame()

    # Export: expected wage summary per hub
    if not expw.empty:
        summ = (expw.groupby(["hub"], as_index=False)
                    .agg(mix_expected_wage=("mix_expected_wage","sum"),
                         overall_msa_median=("overall_msa_median","median")))
        expw_out = OUT / "cps_mix_expected_wage_vs_oews.csv"
        expw.to_csv(expw_out, index=False)
        print(f"[ok] wrote {expw_out}")
        # quick figures
        for _, row in summ.iterrows():
            hub = row["hub"]
            plt.figure(figsize=(6.5,4.2))
            plt.bar(["Mix-expected wage","MSA median"], [row["mix_expected_wage"], row["overall_msa_median"]])
            plt.title(f"Expected wage from Guyanese occupation mix — {hub}")
            plt.ylabel("Dollars")
            _fmt_thousands(plt.gca(), axis="y")
            _savefig(FIG / f"expected_wage_vs_overall_{hub}.png")
        summ_out = OUT / "cps_mix_expected_wage_vs_oews_summary.csv"
        summ.to_csv(summ_out, index=False)
        print(f"[ok] wrote {summ_out}")

    # Export: LQ table and figures
    if not cps_lq.empty:
        lq_out = OUT / "cps_occ_lq_by_hub.csv"
        cps_lq.to_csv(lq_out, index=False)
        print(f"[ok] wrote {lq_out}")
        for hub, sub in cps_lq.groupby("hub"):
            top = sub.sort_values("lq", ascending=False).head(10)
            plt.figure(figsize=(8,5))
            plt.bar(top["soc_title"].fillna(top["soc_code"]), top["lq"])
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top occupation LQs — {hub}")
            plt.ylabel("Location quotient (Guyanese vs all workers)")
            plt.grid(axis="y", alpha=0.25)
            _savefig(FIG / f"lq_top10_{hub}.png")

    return expw, cps_lq


def run_cbp_nes(hubs: list[str]) -> pd.DataFrame | None:
    cbp = load_cbp()
    nes = load_nes()
    if cbp is None and nes is None:
        return None

    frames = []
    for hub in hubs:
        conf = HUBS[hub]
        counties = conf.get("counties", [])
        if not counties:
            # If hub is a state hub, take all counties we see in CBP/NES with that state prefix
            sf = conf.get("state_fips")
            if sf:
                counties = sorted({c for c in (cbp or pd.DataFrame()).get("county_fips", pd.Series([],dtype=str)).tolist()
                                   if isinstance(c, str) and c.startswith(sf)})
        if cbp is not None:
            cbp_h = cbp[cbp["county_fips"].isin(counties)].copy()
            if not cbp_h.empty:
                # Within-county NAICS mix (shares) to avoid population denominators
                cbp_h = (cbp_h.groupby(["county_fips","naics2"], as_index=False)
                             .agg(establishments=("establishments","sum")))
                total_est = cbp_h.groupby("county_fips")["establishments"].transform("sum")
                cbp_h["est_share"] = cbp_h["establishments"] / total_est.replace(0, pd.NA)
                cbp_h["source"] = "CBP"
                cbp_h["hub"] = hub
                frames.append(cbp_h)
        if nes is not None:
            nes_h = nes[nes["county_fips"].isin(counties)].copy()
            if not nes_h.empty:
                nes_h = (nes_h.groupby(["county_fips","naics2"], as_index=False)
                             .agg(nonemployers=("nonemployers","sum")))
                total_ne = nes_h.groupby("county_fips")["nonemployers"].transform("sum")
                nes_h["ne_share"] = nes_h["nonemployers"] / total_ne.replace(0, pd.NA)
                nes_h["source"] = "NES"
                nes_h["hub"] = hub
                frames.append(nes_h)

    if not frames:
        warnings.warn("No CBP/NES rows matched hubs.")
        return None

    df = pd.concat(frames, ignore_index=True)
    out = OUT / "cbp_nes_intensity_by_hub.csv"
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}")

    # Simple figure: top NAICS2 by share across selected counties (average within hub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for hub, sub in df.groupby("hub"):
            # average shares at NAICS2 within hub
            if "est_share" in sub.columns:
                cbp_avg = (sub[sub["source"]=="CBP"]
                              .groupby("naics2", as_index=False)["est_share"].mean()
                              .nlargest(10, "est_share"))
                plt.figure(figsize=(8,5))
                plt.bar(cbp_avg["naics2"], cbp_avg["est_share"])
                plt.title(f"CBP: Top NAICS2 shares — {hub}")
                plt.ylabel("Share of establishments")
                plt.grid(axis="y", alpha=0.25)
                _savefig(FIG / f"cbp_top_naics_{hub}.png")
            if "ne_share" in sub.columns:
                nes_avg = (sub[sub["source"]=="NES"]
                              .groupby("naics2", as_index=False)["ne_share"].mean()
                              .nlargest(10, "ne_share"))
                plt.figure(figsize=(8,5))
                plt.bar(nes_avg["naics2"], nes_avg["ne_share"])
                plt.title(f"NES: Top NAICS2 shares — {hub}")
                plt.ylabel("Share of nonemployers")
                plt.grid(axis="y", alpha=0.25)
                _savefig(FIG / f"nes_top_naics_{hub}.png")

    return df


def run_irs_flows(hubs: list[str]) -> pd.DataFrame | None:
    flows = load_irs_flows()
    if flows is None:
        return None

    frames = []
    for hub in hubs:
        conf = HUBS[hub]
        counties = conf.get("counties", [])
        sf = conf.get("state_fips")
        if not counties and sf:
            counties = [c for c in flows["dest_fips"].unique().tolist() if isinstance(c, str) and c.startswith(sf)]
        if not counties:
            continue
        f_in = flows[flows["dest_fips"].isin(counties)].copy()
        f_out = flows[flows["origin_fips"].isin(counties)].copy()
        inbound = (f_in.groupby("origin_fips", as_index=False)["flows"].sum()
                      .rename(columns={"origin_fips":"other_fips","flows":"inflows"}))
        outbound = (f_out.groupby("dest_fips", as_index=False)["flows"].sum()
                        .rename(columns={"dest_fips":"other_fips","flows":"outflows"}))
        j = inbound.merge(outbound, on="other_fips", how="outer").fillna(0)
        j["net"] = j["inflows"] - j["outflows"]
        j["hub"] = hub
        frames.append(j.sort_values("net", ascending=False).head(15))

        # Figure: top origins by net inflow
        top = j.sort_values("net", ascending=False).head(10)
        plt.figure(figsize=(8,5))
        plt.barh(top["other_fips"].astype(str), top["net"])
        plt.gca().invert_yaxis()
        plt.title(f"IRS: Top net in-flow origins → {hub} (county FIPS)")
        plt.xlabel("Net in-flow (returns or exemptions)")
        _fmt_thousands(plt.gca(), axis="x")
        plt.grid(axis="x", alpha=0.25)
        _savefig(FIG / f"irs_top_origins_{hub}.png")

    if not frames:
        warnings.warn("No IRS flow rows matched hubs.")
        return None

    df = pd.concat(frames, ignore_index=True)
    out = OUT / "irs_netflows_to_hub.csv"
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}")
    return df

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Economy/Labor pipeline for Guyanese hubs")
    ap.add_argument("--hubs", nargs="*", default=list(HUBS.keys()), help="Subset of hubs to process")
    ap.add_argument("--all", action="store_true", help="Run all modules")
    args = ap.parse_args()
    hubs = args.hubs

    # Modules
    expw, lq = run_occ_expected_wage_and_lq(hubs)
    run_cbp_nes(hubs)
    run_irs_flows(hubs)

    print("\n[done] Outputs in economy_labor/outputs and figures in economy_labor/figures")

if __name__ == "__main__":
    main()
