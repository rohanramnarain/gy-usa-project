# oews_metro_median_wages.py
# Usage:
#   python oews_metro_median_wages.py --msas 35620 33100 16980 --year 2024 --out data/oews
#   # also accepts valid 7-digit OEWS area codes (e.g., 0035620)

import argparse
import io
import tempfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://download.bls.gov/pub/time.series/oe/"
UA   = {"User-Agent": "Mozilla/5.0 (compatible; OEWS-fetch/1.0)"}

# ---------- tiny I/O helpers ----------

def read_map(name: str, dtype=str) -> pd.DataFrame:
    """TAB-delimited mapping files per oe.txt; real UA avoids 403."""
    r = requests.get(f"{BASE}{name}", headers=UA, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), sep="\t", dtype=dtype)

def iter_oews_data(chunksize: int = 2_000_000):
    """
    Stream the 'current' OEWS text dump (SPACE-delimited).
    Columns: series_id, year, period, value, footnote_codes
    """
    fname = "oe.data.0.Current"
    with requests.get(f"{BASE}{fname}", headers=UA, stream=True, timeout=300) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for b in r.iter_content(1024 * 1024):
                tmp.write(b)
            path = tmp.name
    cols = ["series_id", "year", "period", "value", "footnote_codes"]
    for ch in pd.read_csv(path, sep=r"\s+", engine="python", dtype=str, usecols=cols, chunksize=chunksize):
        yield ch

# ---------- parsing ----------

def parse_series_bits(df: pd.DataFrame) -> pd.DataFrame:
    """
    series_id layout (per oe.txt):
      [4:11] area_code (7)
      [17:23] occupation_code (6)
      [23:25] datatype_code (2)
    """
    s = df["series_id"]
    df["area_code"]       = s.str[4:11]
    df["occupation_code"] = s.str[17:23]
    df["datatype_code"]   = s.str[23:25]
    return df

# ---------- area normalization ----------

def normalize_msas_to_area7(msas, area_df):
    """
    Accepts:
      - 5-digit CBSA (e.g., 35620)  -> OEWS area_code = '00' + CBSA  (e.g., 0035620)
      - 7-digit OEWS area_code      -> accepted if present in oe.area
      - 7-digit of form CBSA+'00'   -> auto-corrected to '00'+CBSA
    Returns {input_str: valid_area_code or None}
    """
    msas = [str(m) for m in msas]
    valid = set(area_df["area_code"].astype(str))
    out = {}

    for m in msas:
        if m.isdigit() and len(m) == 7 and m in valid:
            out[m] = m
            continue
        if m.isdigit() and len(m) == 5:
            cand = "00" + m
            out[m] = cand if cand in valid else None
            continue
        if m.isdigit() and len(m) == 7 and m.endswith("00"):
            # user likely gave CBSA+'00' -> convert to '00'+CBSA
            cbsa = m[:5]
            cand = "00" + cbsa
            out[m] = cand if cand in valid else None
            continue
        out[m] = None
    return out

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msas", nargs="+", required=True,
                    help="5-digit CBSA (e.g., 35620) or 7-digit OEWS area_code (e.g., 0035620). "
                         "Note: OEWS metro codes are '00' + CBSA.")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="data/oews")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Maps (TAB-delimited per oe.txt)
    dt   = read_map("oe.datatype")     # datatype_code, datatype_name, ...
    area = read_map("oe.area")         # state_code, area_code, areatype_code, area_name, ...
    occ  = read_map("oe.occupation")   # occupation_code, occupation_name, ...

    # Resolve requested areas → valid 7-digit OEWS codes
    resolved = normalize_msas_to_area7(args.msas, area)
    if args.verbose:
        print("Requested → OEWS area_code")
        for k, v in sorted(resolved.items(), key=lambda x: x[0]):
            print(f"  {k} → {v}")
    wanted_area7 = {v for v in resolved.values() if v}
    missing = [k for k, v in resolved.items() if not v]
    if missing:
        raise SystemExit("Could not resolve these to OEWS area_code: " + ", ".join(missing))

    # Median datatype codes (hard-coded: 08 hourly, 13 annual)
    med_codes = {"08", "13"}

    # Stream once → filter by year + areas + medians
    keep = []
    years_seen = set()

    for chunk in iter_oews_data():
        years_seen.update(chunk["year"].unique().tolist())
        # limit to the target year early
        chunk = chunk[chunk["year"] == str(args.year)]
        if chunk.empty:
            continue
        chunk = parse_series_bits(chunk)
        # areas + medians
        chunk = chunk[chunk["area_code"].isin(wanted_area7)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["datatype_code"].isin(med_codes)]
        if not chunk.empty:
            keep.append(chunk[["series_id","year","value","area_code","occupation_code","datatype_code"]])

    if not keep:
        msg = [
            "No rows matched your filters.",
            f"  → Years present in file: {', '.join(sorted(y for y in years_seen if y)) or '(none)'}",
            f"  → Resolved area_codes: {sorted(wanted_area7) or '(none)'}",
        ]
        raise SystemExit("\n".join(msg))

    dat = pd.concat(keep, ignore_index=True)
    dat["value"] = pd.to_numeric(dat["value"], errors="coerce")

    # Attach readable names (best effort if headers differ)
    if "datatype_name" in dt.columns:
        dat = dat.merge(dt[["datatype_code","datatype_name"]], on="datatype_code", how="left")
    occ_name_col = "occupation_name" if "occupation_name" in occ.columns else \
                   next((c for c in occ.columns if "occupation" in c and "name" in c), None)
    if occ_name_col:
        dat = dat.merge(occ[["occupation_code", occ_name_col]].rename(columns={occ_name_col: "occupation_name"}),
                        on="occupation_code", how="left")
    area_name_col = "area_name" if "area_name" in area.columns else \
                    next((c for c in area.columns if "area" in c and "name" in c), None)
    if area_name_col:
        dat = dat.merge(area[["area_code", area_name_col]].rename(columns={area_name_col: "area_name"}),
                        on="area_code", how="left")

    # Prefer ANNUAL median when both exist for (area, occupation)
    def pick_best(g: pd.DataFrame) -> pd.Series:
        annual = g[g["datatype_code"] == "13"]
        row = annual.iloc[0] if len(annual) else g.iloc[0]
        unit = "annual_dollars" if len(annual) else "hourly_dollars"
        return pd.Series({
            "soc_code": row["occupation_code"],
            "soc_title": row.get("occupation_name"),
            "area_code": row["area_code"],
            "area_title": row.get("area_name"),
            "year": int(row["year"]),
            "median_wage": row["value"],
            "wage_unit": unit
        })

    tidy = dat.groupby(["area_code","occupation_code"], as_index=False).apply(pick_best).reset_index(drop=True)

    outpath = outdir / f"oews_metro_median_wages_{args.year}.csv"
    tidy.to_csv(outpath, index=False)

    with open(outdir / "README_oews.txt", "w") as f:
        f.write(
            "Source: BLS OEWS time-series (oe.data.0.Current, oe.datatype, oe.area, oe.occupation).\n"
            "Parsing per oe.txt; medians only (08 hourly, 13 annual), prefer annual when both exist.\n"
            "Areas: OEWS metro codes are '00' + CBSA (e.g., 35620 → 0035620).\n"
        )

    print(f"Wrote {len(tidy):,} rows to {outpath}")

if __name__ == "__main__":
    main()
