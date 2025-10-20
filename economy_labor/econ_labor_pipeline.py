#!/usr/bin/env python3
"""
 econ_labor_pipeline.py
 ----------------------
 Profiles the labor & economic footprint around Guyanese hubs by combining:
   • CPS-based occupation mix (identity-specific)
   • OEWS metro wages (context)
   • (Optional) OEWS total employment for LQs
   • CBP (County Business Patterns) + NES (Nonemployer) (context)
   • IRS county-to-county flows (context)

 Inputs (expected layout under economy_labor/data/...):
   cps/
     cps_guyanese_occ_distribution.csv         # hub × SOC × share (or weighted counts)
     cps_guyanese_summary.csv                  # (optional) hub denominators
   oews/
     oews_metro_median_wages_2024.csv          # CBSA × SOC median wage, (+ tot_emp if available)
   cbp_nes/
     cbp_raw_2023.csv                          # county × NAICS establishments (CBP)
     nes_raw_2022.csv                          # county × NAICS nonemployers (NES)
   irs/
     irs_flows_2010-2011.csv                   # county-to-county flows

 Outputs (economy_labor/outputs):
   cps_mix_expected_wage_vs_oews.csv
   cps_mix_expected_wage_vs_oews_summary.csv
   cps_occ_lq_by_hub.csv
   cbp_nes_intensity_by_hub.csv
   irs_netflows_to_hub.csv

 Figures (economy_labor/figures):
   expected_wage_vs_overall_<hub>.png
   lq_top10_<hub>.png
   cbp_top_naics_<hub>.png
   nes_top_naics_<hub>.png
   irs_top_origins_<hub>.png

 Run:
   python economy_labor/econ_labor_pipeline.py --all
   python economy_labor/econ_labor_pipeline.py --hubs NYC_MSA Miami_MSA

 Tips:
   • If your CPS occ file uses different header names, pass overrides, e.g.:
       --cps-hub-col cbsa --cps-soc-col soc --cps-share-col share --cps-weight-col weighted_count --cps-title-col soc_title
   • If your CPS file has only weighted counts (no share), the script computes shares by hub.
   • If shares are in percent (e.g., 12.3), they will be normalized to proportions.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
# Matplotlib defaults
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ----------------------------
# Config: hubs (edit to taste)
# ----------------------------
HUBS = {
    # Metro hubs (CBSA/MSA based)
    "NYC_MSA": {
        "cbsa": "35620",
        "name": "New York-Newark-Jersey City, NY-NJ-PA",
        "counties": [
            "36005","36047","36061","36081","36085",  # NYC
            "34003","34017","34031","34039","34041","34013","34019"  # NJ ring (editable)
        ],
    },
    "Miami_MSA": {
        "cbsa": "33100",
        "name": "Miami–Fort Lauderdale–West Palm Beach, FL",
        "counties": ["12011","12086","12099"],
    },
    # State hubs (no CBSA)
    "NJ_State": {"state_fips": "34", "name": "New Jersey"},
    "GA_State": {"state_fips": "13", "name": "Georgia"},
    "MD_State": {"state_fips": "24", "name": "Maryland"},
}

# ----------------------------
# Utility helpers
# ----------------------------
def _fmt_thousands(ax, axis="y"):
    if axis == "x":
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def _savefig(path: Path, tight: bool = True):
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


def _ensure_str_zfill(series: pd.Series, width: int) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].fillna(series.astype(str)).str.zfill(width)


def _to_2digit_naics(s) -> str | None:
    s = str(s)
    return s[:2] if s and s.lower() != "none" else None


# ----------------------------
# Loaders (defensive & override-friendly)
# ----------------------------

def load_cps_occ_mix(
    cps_occ_path: Path | None = None,
    hub_col: str | None = None,
    soc_col: str | None = None,
    share_col: str | None = None,
    weight_col: str | None = None,
    title_col: str | None = None,
) -> pd.DataFrame | None:
    """
    Standardize the CPS Guyanese occupation-mix file into columns:
      hub_id, soc_code, guyanese_share, [guyanese_weighted_count], [soc_title]

    • Auto-detects likely columns, or take explicit overrides from CLI.
    • If 'guyanese_share' missing but a weighted count exists, compute shares by hub.
    • If shares look like percents (>1), divide by 100.
    """
    path = cps_occ_path or (DATA / "cps" / "cps_guyanese_occ_distribution.csv")
    if not path.exists():
        warnings.warn(f"Missing {path}; cannot compute occupation analyses.")
        return None

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(candidates: list[str]) -> str | None:
        # exact lowercase match first
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        # then contains match
        for c in df.columns:
            low = c.lower()
            if any(cand.lower() in low for cand in candidates):
                return c
        return None

    hub = hub_col or find_col(["hub", "cbsa", "msa", "msa_code", "state_fips", "state", "fips", "region"])  # noqa
    soc = soc_col or find_col(["soc", "soc_code", "occ", "occupation_code"])  # noqa
    title = title_col or find_col(["soc_title", "occupation", "occ_title"])  # noqa
    share = share_col or find_col(["share", "guyanese_share", "occ_share", "pct", "percent"])  # noqa
    wt = weight_col or find_col(["weighted_count", "wt_count", "count_weighted", "n_weighted", "weight"])  # noqa

    if not hub or not soc:
        raise ValueError(
            "Could not find required columns in CPS occ distribution.\n"
            f"Detected hub={hub}, soc={soc}, share={share}, weight={wt}, title={title}.\n"
            "Provide overrides via --cps-hub-col / --cps-soc-col (and optionally --cps-share-col / --cps-weight-col)."
        )

    keep = [hub, soc]
    if share:
        keep.append(share)
    if wt:
        keep.append(wt)
    if title:
        keep.append(title)

    df = df[keep].copy()
    df.rename(columns={hub: "hub_id", soc: "soc_code"}, inplace=True)
    if share:
        df.rename(columns={share: "guyanese_share"}, inplace=True)
    if wt:
        df.rename(columns={wt: "guyanese_weighted_count"}, inplace=True)
    if title:
        df.rename(columns={title: "soc_title"}, inplace=True)

    # Normalize
    df["hub_id"] = df["hub_id"].astype(str)
    df["soc_code"] = df["soc_code"].astype(str)

    # Compute share if needed
    if "guyanese_share" not in df.columns and "guyanese_weighted_count" in df.columns:
        totals = df.groupby("hub_id")['guyanese_weighted_count'].transform("sum")
        df["guyanese_share"] = df["guyanese_weighted_count"] / totals.replace(0, pd.NA)

    if "guyanese_share" not in df.columns:
        raise ValueError("Could not find or compute 'guyanese_share' in CPS occ distribution file.")

    # If shares look like percent values (>1), normalize to proportions
    med_share = pd.to_numeric(df["guyanese_share"], errors="coerce").median()
    if pd.notna(med_share) and med_share > 1.5:
        df["guyanese_share"] = pd.to_numeric(df["guyanese_share"], errors="coerce") / 100.0

    print("[cps] columns → hub_id, soc_code, guyanese_share"
          f"{' + guyanese_weighted_count' if 'guyanese_weighted_count' in df.columns else ''}"
          f"{' + soc_title' if 'soc_title' in df.columns else ''}")
    return df


def load_cps_summary() -> pd.DataFrame | None:
    path = DATA / "cps" / "cps_guyanese_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hub = _possible(df.columns, "hub", "cbsa", "state_fips", "state", "msa", "region")
    total = _possible(df.columns, "weighted_total", "guyanese_total", "total_weighted", "n_weighted")
    if hub and total:
        return df[[hub, total]].rename(columns={hub: "hub_id", total: "guyanese_total"})
    return None


def load_oews() -> pd.DataFrame:
    path = DATA / "oews" / "oews_metro_median_wages_2024.csv"
    if not path.exists():
        raise FileNotFoundError("Missing economy_labor/data/oews/oews_metro_median_wages_2024.csv")
    df = pd.read_csv(path)

    cbsa = _possible(df.columns, "cbsa", "msa", "area_code", "msa_cbsa", "area")
    soc = _possible(df.columns, "soc_code", "soc", "occ_code")
    med = _possible(df.columns, "median_wage", "h_median", "a_median", "median_annual_wage")
    title = _possible(df.columns, "soc_title", "occupation_title", "occ_title")
    tot_emp = _possible(df.columns, "tot_emp", "employment", "emp")
    name = _possible(df.columns, "msa_name", "area_name", "cbsa_name")

    keep = [c for c in [cbsa, name, soc, title, med, tot_emp] if c]
    df = df[keep].copy()

    df.rename(columns={cbsa: "cbsa", soc: "soc_code", med: "median_wage"}, inplace=True)
    if title:
        df.rename(columns={title: "soc_title"}, inplace=True)
    if name:
        df.rename(columns={name: "msa_name"}, inplace=True)
    if tot_emp:
        df.rename(columns={tot_emp: "tot_emp"}, inplace=True)

    # Clean codes
    df["cbsa"] = _ensure_str_zfill(df["cbsa"], 5)
    df["soc_code"] = df["soc_code"].astype(str)
    return df


def load_cbp() -> pd.DataFrame | None:
    path = DATA / "cbp_nes" / "cbp_raw_2023.csv"
    if not path.exists():
        warnings.warn("Missing cbp_raw_2023.csv; CBP indicators skipped.")
        return None
    df = pd.read_csv(path)
    county = _possible(df.columns, "county_fips", "fips", "county", "geoid")
    naics = _possible(df.columns, "naics", "naics_code")
    est = _possible(df.columns, "establishments", "est", "estab")
    emp = _possible(df.columns, "employment", "emp")  # not required
    keep = [c for c in [county, naics, est, emp] if c]
    df = df[keep].copy()
    df.rename(columns={county: "county_fips", naics: "naics", est: "establishments"}, inplace=True)
    df["county_fips"] = _ensure_str_zfill(df["county_fips"], 5)
    df["naics2"] = df["naics"].map(_to_2digit_naics)
    return df


def load_nes() -> pd.DataFrame | None:
    path = DATA / "cbp_nes" / "nes_raw_2022.csv"
    if not path.exists():
        warnings.warn("Missing nes_raw_2022.csv; NES indicators skipped.")
        return None
    df = pd.read_csv(path)
    county = _possible(df.columns, "county_fips", "fips", "county", "geoid")
    naics = _possible(df.columns, "naics", "naics_code")
    firms = _possible(df.columns, "firms", "nonemployers", "num_firms")
    keep = [c for c in [county, naics, firms] if c]
    df = df[keep].copy()
    df.rename(columns={county: "county_fips", naics: "naics", firms: "nonemployers"}, inplace=True)
    df["county_fips"] = _ensure_str_zfill(df["county_fips"], 5)
    df["naics2"] = df["naics"].map(_to_2digit_naics)
    return df


def load_irs_flows() -> pd.DataFrame | None:
    path = DATA / "irs" / "irs_flows_2010-2011.csv"
    if not path.exists():
        warnings.warn("Missing irs_flows_2010-2011.csv; IRS flows skipped.")
        return None
    df = pd.read_csv(path)
    orig = _possible(df.columns, "origin_fips", "origin", "from_fips")
    dest = _possible(df.columns, "dest_fips", "destination", "to_fips")
    n = _possible(df.columns, "flows", "num_returns", "n_returns", "exemptions")
    keep = [c for c in [orig, dest, n] if c]
    df = df[keep].copy()
    df.rename(columns={orig: "origin_fips", dest: "dest_fips", n: "flows"}, inplace=True)
    df["origin_fips"] = _ensure_str_zfill(df["origin_fips"], 5)
    df["dest_fips"] = _ensure_str_zfill(df["dest_fips"], 5)
    return df

def load_hub_alias_map() -> dict | None:
    """Optional: load alias→hub_key mapping so your CPS 'hub_id' values can map to
    configured hubs like NYC_MSA, Miami_MSA, etc.
    CSV path: economy_labor/data/cps/cps_hub_aliases.csv with columns:
      alias, hub_key
    Example rows:
      NYC, NYC_MSA
      35620, NYC_MSA
      New York, NYC_MSA
      Miami, Miami_MSA
      33100, Miami_MSA
      NJ, NJ_State
    Returns a dict with lowercased alias → hub_key.
    """
    path = DATA / "cps" / "cps_hub_aliases.csv"
    if not path.exists():
        return None
    m = {}
    try:
        df = pd.read_csv(path)
        a = _possible(df.columns, "alias") or "alias"
        h = _possible(df.columns, "hub_key") or "hub_key"
        for _, r in df[[a, h]].dropna().iterrows():
            m[str(r[a]).strip().lower()] = str(r[h]).strip()
    except Exception:
        return None
    return m if m else None


# ----------------------------
# Analyses

def load_hub_alias_map() -> dict | None:
    """Optional CSV: economy_labor/data/cps/cps_hub_aliases.csv with columns alias,hub_key
    Returns a dict of lowercased alias -> hub_key.
    """
    path = DATA / "cps" / "cps_hub_aliases.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        a = _possible(df.columns, "alias") or "alias"
        h = _possible(df.columns, "hub_key") or "hub_key"
        m = {str(r[a]).strip().lower(): str(r[h]).strip() for _, r in df[[a, h]].dropna().iterrows()}
        return m or None
    except Exception:
        return None


def _guess_hub_from_text(s: str) -> str | None:
    """Heuristic mapper for messy hub labels in CPS file.
    Maps common strings/CBSA codes to configured HUBS keys.
    Extend as needed.
    """
    if not isinstance(s, str):
        s = str(s)
    t = s.strip().lower()
    # direct CBSA code hits
    if any(code in t for code in [HUBS['NYC_MSA'].get('cbsa', ''), '35620']):
        return 'NYC_MSA'
    if any(code in t for code in [HUBS['Miami_MSA'].get('cbsa', ''), '33100']):
        return 'Miami_MSA'
    # name tokens
    if any(k in t for k in ['new york', 'nyc', 'brooklyn', 'queens', 'manhattan', 'bronx', 'staten island']):
        return 'NYC_MSA'
    if any(k in t for k in ['miami', 'broward', 'fort lauderdale', 'palm beach']):
        return 'Miami_MSA'
    if any(k in t for k in ['new jersey', 'nj']):
        return 'NJ_State'
    if any(k in t for k in ['georgia', 'ga', 'atlanta']):
        return 'GA_State'
    if any(k in t for k in ['maryland', 'md', 'baltimore']):
        return 'MD_State'
    # 2-digit state FIPS contained
    for key, conf in HUBS.items():
        if 'state_fips' in conf and conf['state_fips'] in t:
            return key
    return None


def apply_hub_mapping_or_guess(cps_occ: pd.DataFrame, write_template: bool = False) -> pd.DataFrame:
    """Create a 'hub' column by applying alias map first, then heuristics.
    Optionally writes a template alias CSV with guesses for manual edits.
    """
    df = cps_occ.copy()
    alias_map = load_hub_alias_map() or {}
    df['hub'] = None

    # 1) alias map first
    if alias_map:
        df['alias_norm'] = df['hub_id'].astype(str).str.strip().str.lower()
        df.loc[df['alias_norm'].isin(alias_map.keys()), 'hub'] = df['alias_norm'].map(alias_map)
        df.drop(columns=['alias_norm'], inplace=True)

    # 2) heuristic for anything still unmapped
    mask_un = df['hub'].isna()
    if mask_un.any():
        df.loc[mask_un, 'hub'] = df.loc[mask_un, 'hub_id'].apply(_guess_hub_from_text)

    # 3) optionally write a template to help manual mapping
    if write_template:
        tmpl = (df[['hub_id', 'hub']]
                .assign(hub_key_guess=lambda x: x['hub'])
                .rename(columns={'hub': 'current_mapped'})
                .drop_duplicates('hub_id'))
        outp = DATA / 'cps' / 'cps_hub_aliases_template.csv'
        outp.parent.mkdir(parents=True, exist_ok=True)
        tmpl.to_csv(outp, index=False)
        print(f"[ok] wrote alias template → {outp}. Fill a 'hub_key' column and save as cps_hub_aliases.csv")

    # warn if still unmapped values exist
    still_un = df['hub'].isna()
    if still_un.any():
        sample = df.loc[still_un, 'hub_id'].astype(str).dropna().unique().tolist()[:15]
        warnings.warn(f"Unmapped CPS hub labels (first 15): {sample}. Add them to cps_hub_aliases.csv.")

    return df

# ----------------------------

def _match_hub_rows(cps_occ: pd.DataFrame, hub_key: str) -> pd.DataFrame:
    """Return CPS rows for a configured hub, matching by CBSA code, state FIPS, or hub name.
    """
    conf = HUBS[hub_key]
    # try CBSA first
    if "cbsa" in conf:
        cbsa = conf["cbsa"]
        mask = cps_occ["hub_id"].astype(str).str.contains(cbsa)
        sub = cps_occ[mask]
        if not sub.empty:
            return sub.assign(hub=hub_key)
    # then state fips
    if "state_fips" in conf:
        sf = conf["state_fips"].zfill(2)
        mask = cps_occ["hub_id"].astype(str).str.zfill(2).str.contains(sf)
        sub = cps_occ[mask]
        if not sub.empty:
            return sub.assign(hub=hub_key)
    # fall back: name contains
    name = conf.get("name")
    if name:
        tokens = [t for t in ["new york", "miami", "new jersey", "georgia", "maryland"] if t in name.lower()]
        if tokens:
            mask = cps_occ["hub_id"].str.lower().str.contains(tokens[0])
            sub = cps_occ[mask]
            if not sub.empty:
                return sub.assign(hub=hub_key)
    # final fallback: any row whose hub_id equals the hub key
    sub = cps_occ[cps_occ["hub_id"].astype(str).str.lower() == hub_key.lower()]
    if not sub.empty:
        return sub.assign(hub=hub_key)
    return pd.DataFrame(columns=list(cps_occ.columns) + ["hub"])  # empty


def run_occ_expected_wage_and_lq(hubs: list[str], args=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load OEWS first (used later and for inferring codes if needed)
    oews = load_oews()
    cps_occ = load_cps_occ_mix(
        cps_occ_path=getattr(args, "cps_occ_file", None),
        hub_col=getattr(args, "cps_hub_col", None),
        soc_col=getattr(args, "cps_soc_col", None),
        share_col=getattr(args, "cps_share_col", None),
        weight_col=getattr(args, "cps_weight_col", None),
        title_col=getattr(args, "cps_title_col", None),
    )
    if cps_occ is None or cps_occ.empty:
        raise SystemExit("Cannot compute occupation analyses without a CPS occupation file.")

    # Optional alias mapping for CPS hub_id → configured hub keys
    alias_map = load_hub_alias_map()
    matched_frames = []
    matched_idx = set()
    if alias_map:
        tmp = cps_occ.copy()
        tmp["alias_norm"] = tmp["hub_id"].astype(str).str.strip().str.lower()
        alias_hits = tmp[tmp["alias_norm"].isin(alias_map.keys())].copy()
        if not alias_hits.empty:
            alias_hits["hub"] = alias_hits["alias_norm"].map(alias_map)
            matched_frames.append(alias_hits.drop(columns=["alias_norm"]))
            matched_idx = set(alias_hits.index.tolist())

    # Collect CPS rows for each hub via pattern matching (skip rows already matched by alias)
    for hub in hubs:
        sub_all = _match_hub_rows(cps_occ, hub)
        sub = sub_all[~sub_all.index.isin(matched_idx)] if not sub_all.empty else sub_all
        if sub.empty:
            warnings.warn(f"CPS occupation mix has no rows for hub={hub}. Check hub mapping and HUBS.")
            continue
        matched_frames.append(sub)

    if not matched_frames:
        raise SystemExit("No CPS rows matched any selected hub.")
    mix = pd.concat(matched_frames, ignore_index=True)

    # -------- Expected wage (CBSA hubs only) --------
    frames = []
    cbsa_hubs = [h for h in hubs if "cbsa" in HUBS[h]]
    for hub in cbsa_hubs:
        conf = HUBS[hub]
        m = mix[mix["hub"] == hub]
        oo = oews[oews["cbsa"] == conf["cbsa"]]
        if m.empty or oo.empty:
            warnings.warn(f"No overlap for expected wage in hub={hub}.")
            continue
        j = m.merge(oo, on="soc_code", how="inner")
        if j.empty:
            warnings.warn(f"SOC codes did not intersect for hub={hub}.")
            continue
        j["mix_expected_wage"] = pd.to_numeric(j["guyanese_share"], errors="coerce") * pd.to_numeric(j["median_wage"], errors="coerce")
        overall_msa_median = pd.to_numeric(oo["median_wage"], errors="coerce").median()
        j = j.assign(overall_msa_median=overall_msa_median)
        frames.append(j)

    expw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not expw.empty:
        expw_out = OUT / "cps_mix_expected_wage_vs_oews.csv"
        expw.to_csv(expw_out, index=False)
        print(f"[ok] wrote {expw_out}")
        # Summaries + figs
        summ = (expw.groupby("hub", as_index=False)
                     .agg(mix_expected_wage=("mix_expected_wage", "sum"),
                          overall_msa_median=("overall_msa_median", "median")))
        summ_out = OUT / "cps_mix_expected_wage_vs_oews_summary.csv"
        summ.to_csv(summ_out, index=False)
        print(f"[ok] wrote {summ_out}")
        for _, row in summ.iterrows():
            hub = row["hub"]
            plt.figure(figsize=(6.6,4.2))
            plt.bar(["Mix-expected wage", "MSA median"], [row["mix_expected_wage"], row["overall_msa_median"]])
            plt.ylabel("Dollars")
            plt.title(f"Expected wage from Guyanese occupation mix — {hub}")
            _fmt_thousands(plt.gca(), axis="y")
            _savefig(FIG / f"expected_wage_vs_overall_{hub}.png")

    # -------- Location quotients (needs OEWS tot_emp) --------
    cps_lq = pd.DataFrame()
    if "tot_emp" in oews.columns and not oews["tot_emp"].isna().all():
        lq_frames = []
        for hub in cbsa_hubs:
            conf = HUBS[hub]
            m = mix[mix["hub"] == hub]
            oo = oews[oews["cbsa"] == conf["cbsa"]].copy()
            if m.empty or oo.empty:
                continue
            # compute all-worker SOC share in that MSA
            msa_total_emp = pd.to_numeric(oo["tot_emp"], errors="coerce").sum()
            if not msa_total_emp or msa_total_emp == 0:
                continue
            oo["allworkers_share"] = pd.to_numeric(oo["tot_emp"], errors="coerce") / msa_total_emp
            j = m.merge(oo[["soc_code", "soc_title", "allworkers_share"]], on="soc_code", how="inner")
            if j.empty:
                continue
            j["lq"] = pd.to_numeric(j["guyanese_share"], errors="coerce") / j["allworkers_share"].replace(0, pd.NA)
            j["hub"] = hub
            lq_frames.append(j[["hub", "soc_code", "soc_title", "guyanese_share", "allworkers_share", "lq"]])
        if lq_frames:
            cps_lq = pd.concat(lq_frames, ignore_index=True)
            lq_out = OUT / "cps_occ_lq_by_hub.csv"
            cps_lq.to_csv(lq_out, index=False)
            print(f"[ok] wrote {lq_out}")
            for hub, sub in cps_lq.groupby("hub"):
                top = sub.sort_values("lq", ascending=False).head(10)
                plt.figure(figsize=(8.4,5))
                labels = top["soc_title"].fillna(top["soc_code"])
                plt.bar(labels, top["lq"])
                plt.xticks(rotation=45, ha="right")
                plt.title(f"Top occupation LQs — {hub}")
                plt.ylabel("Location quotient (Guyanese vs all workers)")
                plt.grid(axis="y", alpha=0.25)
                _savefig(FIG / f"lq_top10_{hub}.png")
    else:
        warnings.warn("OEWS file lacks total employment (`tot_emp`); LQs skipped. Expected wage still computed.")

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
        if not counties and conf.get("state_fips"):
            sf = conf["state_fips"]
            # derive counties for that state from whichever dataset is available
            pool = []
            if cbp is not None:
                pool += cbp["county_fips"].dropna().astype(str).tolist()
            if nes is not None:
                pool += nes["county_fips"].dropna().astype(str).tolist()
            counties = sorted({c for c in pool if isinstance(c, str) and c.startswith(sf)})
        if not counties:
            continue

        if cbp is not None:
            cbp_h = cbp[cbp["county_fips"].isin(counties)].copy()
            if not cbp_h.empty:
                cbp_h = (
                    cbp_h.groupby(["county_fips", "naics2"], as_index=False)
                         .agg(establishments=("establishments", "sum"))
                )
                total_est = cbp_h.groupby("county_fips")["establishments"].transform("sum")
                cbp_h["est_share"] = cbp_h["establishments"] / total_est.replace(0, pd.NA)
                cbp_h["source"] = "CBP"
                cbp_h["hub"] = hub
                frames.append(cbp_h)

        if nes is not None:
            nes_h = nes[nes["county_fips"].isin(counties)].copy()
            if not nes_h.empty:
                nes_h = (
                    nes_h.groupby(["county_fips", "naics2"], as_index=False)
                         .agg(nonemployers=("nonemployers", "sum"))
                )
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

    # Simple visuals: average shares per NAICS2 in the hub
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for hub, sub in df.groupby("hub"):
            if "est_share" in sub.columns:
                cbp_avg = (
                    sub[sub["source"] == "CBP"].groupby("naics2", as_index=False)["est_share"].mean()
                    .nlargest(10, "est_share")
                )
                plt.figure(figsize=(8, 5))
                plt.bar(cbp_avg["naics2"], cbp_avg["est_share"])
                plt.title(f"CBP: Top NAICS2 shares — {hub}")
                plt.ylabel("Share of establishments")
                plt.grid(axis="y", alpha=0.25)
                _savefig(FIG / f"cbp_top_naics_{hub}.png")
            if "ne_share" in sub.columns:
                nes_avg = (
                    sub[sub["source"] == "NES"].groupby("naics2", as_index=False)["ne_share"].mean()
                    .nlargest(10, "ne_share")
                )
                plt.figure(figsize=(8, 5))
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
        if not counties and conf.get("state_fips"):
            sf = conf["state_fips"]
            pool = set(flows["dest_fips"].tolist() + flows["origin_fips"].tolist())
            counties = sorted([c for c in pool if isinstance(c, str) and c.startswith(sf)])
        if not counties:
            continue

        f_in = flows[flows["dest_fips"].isin(counties)].copy()
        f_out = flows[flows["origin_fips"].isin(counties)].copy()
        inbound = (
            f_in.groupby("origin_fips", as_index=False)["flows"].sum().rename(columns={"origin_fips": "other_fips", "flows": "inflows"})
        )
        outbound = (
            f_out.groupby("dest_fips", as_index=False)["flows"].sum().rename(columns={"dest_fips": "other_fips", "flows": "outflows"})
        )
        j = inbound.merge(outbound, on="other_fips", how="outer").fillna(0)
        j["net"] = j["inflows"] - j["outflows"]
        j["hub"] = hub
        frames.append(j.sort_values("net", ascending=False).head(15))

        # Figure: top origins by net inflow
        top = j.sort_values("net", ascending=False).head(10)
        plt.figure(figsize=(8.2, 5))
        plt.barh(top["other_fips"].astype(str), top["net"])
        plt.gca().invert_yaxis()
        plt.title(f"IRS: Top net in-flow origins → {hub} (county FIPS)")
        plt.xlabel("Net in-flow (returns/exemptions)")
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
    ap.add_argument("--hubs", nargs="*", default=list(HUBS.keys()), help="Subset of hubs to process (keys in HUBS dict)")
    ap.add_argument("--all", action="store_true", help="Run all modules (default behavior)")

    # CPS overrides
    ap.add_argument("--cps-occ-file", type=Path, default=DATA / "cps" / "cps_guyanese_occ_distribution.csv")
    ap.add_argument("--cps-hub-col", type=str, help="Column in CPS occ file that identifies the hub (e.g., 'cbsa' or 'state_fips').")
    ap.add_argument("--cps-soc-col", type=str, help="Column with SOC code (e.g., 'soc' or 'occ_code').")
    ap.add_argument("--cps-share-col", type=str, help="Column with Guyanese share within SOC (0..1 or %).")
    ap.add_argument("--cps-weight-col", type=str, help="Column with Guyanese weighted counts by SOC.")
    ap.add_argument("--cps-title-col", type=str, help="Optional SOC title column.")

    args = ap.parse_args()

    hubs = args.hubs

    # Run modules
    expw, lq = run_occ_expected_wage_and_lq(hubs, args)
    run_cbp_nes(hubs)
    run_irs_flows(hubs)

    print("\n[done] Outputs in economy_labor/outputs and figures in economy_labor/figures")


if __name__ == "__main__":
    main()
