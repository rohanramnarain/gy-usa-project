#!/usr/bin/env python3
"""
edu_pipeline.py
----------------
Create education-focused outputs about Guyanese in the U.S. using **ACS PUMS**
microdata you already generated, plus optional IPEDS context if you have the
CSV files downloaded.

Inputs (defaults assume you ran the earlier ACS pipeline):
- data/outputs/pums_2023_guyanese_persons.csv              # required
- data/outputs/pums_2023_state_weighted_counts.csv         # optional (denominators)
- (optional IPEDS context)
    education/ipeds/C2023_A.csv    # Completions survey: awards by CIP/level
    education/ipeds/HD2023.csv     # Institution Header: state, CBSA, etc.

Outputs (under education/outputs/):
- pums_guyanese_educ_attainment_national.csv
- pums_guyanese_educ_attainment_by_state.csv
- pums_guyanese_educ_attainment_by_nativity.csv
- pums_guyanese_field_of_degree_top.csv  (top FOD1P among BA+ holders)
- (if IPEDS provided)
  - ipeds_completions_state_summary.csv
  - ipeds_completions_top_cip_by_state.csv

Figures (under education/figures/):
- edu_attainment_national.png
- edu_attainment_top_states.png
- edu_attainment_by_nativity.png
- top_fields_degree_national.png

Run examples:
  python education/edu_pipeline.py --year 2023 --run all
  python education/edu_pipeline.py --year 2023 --run pums
  python education/edu_pipeline.py --year 2023 --run ipeds \
      --ipeds-completions education/ipeds/C2023_A.csv \
      --ipeds-header education/ipeds/HD2023.csv

Requirements: pandas, matplotlib
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ---------- Paths ----------
ROOT = Path(".")
OUTDIR_DEFAULT = Path("education/outputs")
FIGDIR_DEFAULT = Path("education/figures")

# ---------- Matplotlib defaults ----------
plt.rcParams.update({
    "figure.dpi": 180,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------- Helpers ----------

def _ensure_dirs(outdir: Path, figdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

from typing import Optional

def _fmt(ax, axis='y'):
    if axis == 'x':
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

def _savefig(path: Path):
    try:
        plt.gcf().tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {path}")

# ---------- PUMS education logic ----------

# Minimal bins for ACS PUMS educational attainment
# Works with either EDUCD or SCHL (tries EDUCD first).
EDUCD_BINS = [
    ("< HS",        lambda v: v < 62),
    ("HS or GED",   lambda v: v in (62, 63)),
    ("Some college",lambda v: v in (64, 65)),
    ("Associate",   lambda v: 100 <= v < 112),
    ("Bachelor's",  lambda v: v == 113),
    ("Graduate",    lambda v: v >= 114),
]

SCHL_BINS = [
    ("< HS",        lambda v: v is not None and v < 15),   # < High school diploma
    ("HS or GED",   lambda v: v == 15),
    ("Some college",lambda v: v in (16, 17, 18)),
    ("Associate",   lambda v: v == 19),
    ("Bachelor's",  lambda v: v == 20),
    ("Graduate",    lambda v: v >= 21),
]

def bin_education(df: pd.DataFrame) -> pd.Series:
    if "EDUCD" in df.columns:
        v = pd.to_numeric(df["EDUCD"], errors="coerce")
        out = pd.Series(index=df.index, dtype="object")
        for label, pred in EDUCD_BINS:
            out = out.where(~v.apply(lambda x: pd.notna(x) and pred(int(x))), other=label)
        return out
    elif "SCHL" in df.columns:
        v = pd.to_numeric(df["SCHL"], errors="coerce")
        out = pd.Series(index=df.index, dtype="object")
        for label, pred in SCHL_BINS:
            out = out.where(~v.apply(lambda x: pd.notna(x) and pred(int(x))), other=label)
        return out
    else:
        raise ValueError("PUMS file missing EDUCD and SCHL; cannot compute attainment bins.")

FOD_VALID_CODES = set(range(1100, 6200))  # broad guard; ACS FOD codes live in ~1xxx-6xxx

# ---------- PUMS processors ----------

def pums_paths(year: int) -> dict:
    base = Path("data/outputs")
    return {
        "guy_persons": base / f"pums_{year}_guyanese_persons.csv",
        "state_weights": base / f"pums_{year}_state_weighted_counts.csv",
    }


def run_pums_education(year: int, outdir: Path, figdir: Path) -> dict:
    paths = pums_paths(year)
    gp = paths["guy_persons"]
    if not gp.exists():
        raise FileNotFoundError(f"Missing {gp}. Run your ACS pipeline first.")

    df = pd.read_csv(gp)
    # Weight column
    wcol = "PWGTP" if "PWGTP" in df.columns else None
    if not wcol:
        raise ValueError("PUMS file missing PWGTP weight column.")

    # Clean keys
    if "STATEFIP" in df.columns:
        df["STATEFIP"] = df["STATEFIP"].astype(str).str.zfill(2)

    # Attainment bins
    df["attain_bin"] = bin_education(df)

    # NATIONAL attainment (weighted)
    nat = (df.groupby("attain_bin", as_index=False)[wcol].sum()
             .rename(columns={wcol:"weighted_count"})
             .sort_values("weighted_count", ascending=False))
    nat_out = outdir / "pums_guyanese_educ_attainment_national.csv"
    nat.to_csv(nat_out, index=False)
    print(f"[ok] wrote {nat_out}")

    # Plot national
    plt.figure(figsize=(7.5,5))
    plt.bar(nat["attain_bin"], nat["weighted_count"]) 
    plt.title("Guyanese Educational Attainment (ACS PUMS, weighted)")
    plt.ylabel("People (weighted)")
    plt.xlabel("Highest attainment")
    plt.grid(axis='y', alpha=0.25)
    _fmt(plt.gca())
    _savefig(figdir / "edu_attainment_national.png")

    # BY STATE
    if "STATEFIP" in df.columns:
        by_state = (df.groupby(["STATEFIP","attain_bin"], as_index=False)[wcol].sum()
                      .rename(columns={wcol:"weighted_count"}))
        st_out = outdir / "pums_guyanese_educ_attainment_by_state.csv"
        by_state.to_csv(st_out, index=False)
        print(f"[ok] wrote {st_out}")
        # quick top states chart for BA+ (Bachelor's + Graduate)
        ba_states = by_state[by_state["attain_bin"].isin(["Bachelor's","Graduate"])].copy()
        top = (ba_states.groupby("STATEFIP", as_index=False)["weighted_count"].sum()
                        .sort_values("weighted_count", ascending=False).head(12))
        plt.figure(figsize=(8,5))
        plt.bar(top["STATEFIP"], top["weighted_count"])
        plt.title("Where BA+ Guyanese are concentrated (ACS PUMS)")
        plt.xlabel("State FIPS")
        plt.ylabel("People (weighted)")
        plt.grid(axis='y', alpha=0.25)
        _fmt(plt.gca())
        _savefig(figdir / "edu_attainment_top_states.png")

    # BY NATIVITY
    if "NATIVITY" in df.columns:
        nat_df = (df.groupby(["NATIVITY","attain_bin"], as_index=False)[wcol].sum()
                    .rename(columns={wcol:"weighted_count"}))
        natv_out = outdir / "pums_guyanese_educ_attainment_by_nativity.csv"
        nat_df.to_csv(natv_out, index=False)
        print(f"[ok] wrote {natv_out}")
        # chart: BA+ by nativity
        ba_nat = nat_df[nat_df["attain_bin"].isin(["Bachelor's","Graduate"])].copy()
        pivot = (ba_nat.pivot(index="NATIVITY", columns="attain_bin", values="weighted_count")
                        .fillna(0))
        pivot["BA+"] = pivot.sum(axis=1)
        plt.figure(figsize=(6.5,4.5))
        plt.bar(pivot.index.astype(str), pivot["BA+"])
        plt.title("BA+ among Guyanese by nativity (ACS PUMS)")
        plt.xlabel("Nativity (1=US native, 2=Foreign-born)")
        plt.ylabel("People (weighted)")
        plt.grid(axis='y', alpha=0.25)
        _fmt(plt.gca())
        _savefig(figdir / "edu_attainment_by_nativity.png")

    # FIELD OF DEGREE (FOD1P) among BA+
    fod_cols = [c for c in ["FOD1P","FOD2P"] if c in df.columns]
    if fod_cols:
        # Keep BA+ only
        ba_plus_mask = df["attain_bin"].isin(["Bachelor's","Graduate"]) 
        subset = df.loc[ba_plus_mask, [*fod_cols, wcol]].copy()
        long = subset.melt(value_vars=fod_cols, value_name="FODP", var_name="which", ignore_index=False)
        long["FODP"] = pd.to_numeric(long["FODP"], errors="coerce")
        long = long[long["FODP"].between(1100, 6199, inclusive="both")]
        fld = (long.groupby("FODP", as_index=False)[wcol].sum()
                    .rename(columns={wcol:"weighted_count"})
                    .sort_values("weighted_count", ascending=False).head(15))
        fd_out = outdir / "pums_guyanese_field_of_degree_top.csv"
        fld.to_csv(fd_out, index=False)
        print(f"[ok] wrote {fd_out}")
        plt.figure(figsize=(9,5))
        plt.bar(fld["FODP"].astype(int).astype(str), fld["weighted_count"]) 
        plt.title("Top fields of degree among BA+ Guyanese (ACS PUMS FOD1P/FOD2P)")
        plt.xlabel("Field-of-degree code (FODP)")
        plt.ylabel("People (weighted)")
        plt.grid(axis='y', alpha=0.25)
        _fmt(plt.gca())
        _savefig(figdir / "top_fields_degree_national.png")
    else:
        warnings.warn("PUMS file lacks FOD1P/FOD2P; skipping field-of-degree.")

    return {
        "attainment_national": nat,
        "attainment_by_state": by_state if "STATEFIP" in df.columns else None,
        "attainment_by_nativity": nat_df if "NATIVITY" in df.columns else None,
    }

# ---------- Optional: IPEDS completions context ----------

IPEDS_REQ_COLS_C = {
    # Try common names; we will auto-detect actual column labels below
    "unitid": ("UNITID", "unitid"),
    "state":  ("STABBR", "stabbr", "STATE"),
    # completions counts vary by columns; we will sum across award levels
}

IPEDS_REQ_COLS_HD = {
    "unitid": ("UNITID","unitid"),
    "state":  ("STABBR","stabbr","STATE"),
    "inst":   ("INSTNM","instnm","NAME"),
}

AWARD_COL_PREFIXES = (
    "CTOTAL", "CIP", "AWARD",  # handle common IPEDS patterns
)


def _find_any_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def run_ipeds_context(comp_path: Path, hd_path: Path, outdir: Path) -> dict:
    if not comp_path.exists() or not hd_path.exists():
        raise FileNotFoundError("IPEDS files not found. Provide --ipeds-completions and --ipeds-header paths.")

    C = pd.read_csv(comp_path, low_memory=False)
    HD = pd.read_csv(hd_path, low_memory=False)

    # Identify key columns
    unitid_c = _find_any_col(C.columns, IPEDS_REQ_COLS_C["unitid"]) or "UNITID"
    unitid_h = _find_any_col(HD.columns, IPEDS_REQ_COLS_HD["unitid"]) or "UNITID"
    state_h  = _find_any_col(HD.columns, IPEDS_REQ_COLS_HD["state"])  or "STABBR"
    inst_h   = _find_any_col(HD.columns, IPEDS_REQ_COLS_HD["inst"])   or "INSTNM"

    # Sum completions across all numeric columns that look like award counts
    comp_cols = [c for c in C.columns if any(c.upper().startswith(p) for p in AWARD_COL_PREFIXES)]
    comp = C[[unitid_c, *comp_cols]].copy()
    for c in comp_cols:
        comp[c] = pd.to_numeric(comp[c], errors="coerce")
    comp["total_completions"] = comp[comp_cols].sum(axis=1, skipna=True)

    # Join to header for state info
    slim_hd = HD[[unitid_h, state_h, inst_h]].rename(columns={unitid_h:"UNITID", state_h:"STATE", inst_h:"INSTNM"})
    comp = comp.rename(columns={unitid_c:"UNITID"}).merge(slim_hd, on="UNITID", how="left")

    # State summary
    state_summary = (comp.groupby("STATE", as_index=False)["total_completions"].sum()
                         .sort_values("total_completions", ascending=False))
    ss_out = outdir / "ipeds_completions_state_summary.csv"
    state_summary.to_csv(ss_out, index=False)
    print(f"[ok] wrote {ss_out}")

    # Top institutions per state (top 10)
    comp_sorted = (comp.sort_values(["STATE","total_completions"], ascending=[True, False]))
    top_inst = comp_sorted.groupby("STATE").head(10)[["STATE","INSTNM","total_completions"]]
    ti_out = outdir / "ipeds_completions_top_cip_by_state.csv"
    top_inst.to_csv(ti_out, index=False)
    print(f"[ok] wrote {ti_out}")

    return {"state_summary": state_summary, "top_institutions": top_inst}

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Education pipeline for Guyanese analysis (ACS PUMS + optional IPEDS context)")
    ap.add_argument("--year", type=int, default=2023, help="ACS PUMS year to read from data/outputs/")
    ap.add_argument("--run", choices=["all","pums","ipeds"], default="all", help="Which modules to run")
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    ap.add_argument("--figdir", type=Path, default=FIGDIR_DEFAULT)
    ap.add_argument("--ipeds-completions", type=Path, default=Path("education/ipeds/C2023_A.csv"))
    ap.add_argument("--ipeds-header", type=Path, default=Path("education/ipeds/HD2023.csv"))
    args = ap.parse_args()

    _ensure_dirs(args.outdir, args.figdir)

    if args.run in ("all","pums"):
        run_pums_education(args.year, args.outdir, args.figdir)

    if args.run in ("all","ipeds"):
        try:
            run_ipeds_context(args.__dict__["ipeds_completions"], args.__dict__["ipeds_header"], args.outdir)
        except FileNotFoundError as e:
            warnings.warn(str(e))

    print("\n[done] Education outputs in", args.outdir, "and figures in", args.figdir)

if __name__ == "__main__":
    main()
