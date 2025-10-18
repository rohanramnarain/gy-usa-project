#!/usr/bin/env python3
"""
Build ZIP-level Guyanese totals (CSV + optional GeoJSON) **without re-downloading**.

This consumes your existing tract-level CSV and local tract/ZCTA shapefiles.
By default it runs fully offline using an area-weighted tract→ZCTA overlay.
Optionally add --fetch-pop to do population-weighted apportionment via a tiny ACS call.

Inputs (expected on disk):
- outputs/acs_tracts_guyanese.csv  (geoid, estimate, moe [plus others])
- data/zcta/tl_2020_us_zcta510.shp
- data/tract_36/tl_2020_36_tract.shp (NY)
- data/tract_12/tl_2020_12_tract.shp (FL)
- (Optional) HUD ZIP↔Tract crosswalk CSV (if you have it): --hud-xwalk

Outputs:
- outputs/zip_guyanese.csv
- outputs/zip_guyanese.geojson  (if --make-geojson)
"""
from __future__ import annotations
import os
import argparse
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

NYC_COUNTIES = {"005", "047", "061", "081", "085"}  # Bronx, Kings, New York, Queens, Richmond
SOFLA_COUNTIES_FL = {"011", "086", "099"}  # Broward, Miami-Dade, Palm Beach

FIPS_STATES = {"NY": "36", "FL": "12"}
B01003_TOTAL_E = "B01003_001E"

# -----------------------
# IO helpers
# -----------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_tract_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in ("estimate", "moe"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "geoid" not in df.columns and {"state", "county", "tract"}.issubset(df.columns):
        df["geoid"] = df["state"] + df["county"] + df["tract"]
    if "geoid" not in df.columns:
        raise ValueError("acs_tracts_guyanese.csv must include 'geoid' or state/county/tract.")
    return df[["geoid", "estimate", "moe"]].copy()


def read_tract_shapes(ny_path: str, fl_path: str) -> gpd.GeoDataFrame:
    g_ny = gpd.read_file(ny_path)[["GEOID", "COUNTYFP", "geometry"]].copy()
    g_fl = gpd.read_file(fl_path)[["GEOID", "COUNTYFP", "geometry"]].copy()
    g_ny = g_ny[g_ny["COUNTYFP"].isin(NYC_COUNTIES)]
    g_fl = g_fl[g_fl["COUNTYFP"].isin(SOFLA_COUNTIES_FL)]
    g = pd.concat([g_ny, g_fl], ignore_index=True)
    g = gpd.GeoDataFrame(g.rename(columns={"GEOID": "geoid"}), geometry="geometry", crs=g_ny.crs)
    return g


def read_zcta_shapes(path: str) -> gpd.GeoDataFrame:
    z = gpd.read_file(path)
    for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "GEOID"):
        if c in z.columns:
            z["zip"] = z[c].astype(str).str.zfill(5)
            break
    if "zip" not in z.columns:
        raise ValueError("Could not find a ZIP code column in ZCTA shapefile")
    return z[["zip", "geometry"]]


# -----------------------
# Crosswalks
# -----------------------

def build_overlay_xwalk(tracts_gdf: gpd.GeoDataFrame, zcta_gdf: gpd.GeoDataFrame,
                        pop_by_tract: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create tract→ZIP weights by spatial overlay (equal-area CRS for area calcs).
    If pop_by_tract provided with columns [geoid, tract_pop], weight by population; else by area.
    Returns: columns [zip, tract, weight].
    """
    tr = tracts_gdf.to_crs(5070).copy()
    zc = zcta_gdf.to_crs(5070).copy()

    # Clip ZCTAs to bbox of tracts to speed up
    minx, miny, maxx, maxy = tr.total_bounds
    bbox = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=tr.crs)
    try:
        zc = gpd.overlay(zc, bbox, how="intersection")
    except Exception:
        pass

    tr["_area"] = tr.area

    inter = gpd.overlay(tr, zc, how="intersection", keep_geom_type=False)
    if inter.empty:
        raise RuntimeError("Overlay produced 0 rows — check your CRS and input geometries.")
    inter["_iarea"] = inter.geometry.area

    # Area fraction within tract; handle cases where `_area` already exists as _area/_area_x/_area_y
    if ("_area" not in inter.columns) and ("_area_x" not in inter.columns) and ("_area_y" not in inter.columns):
        inter = inter.merge(tr[["geoid", "_area"]], on="geoid", how="left")
    denom_col = "_area" if "_area" in inter.columns else ("_area_x" if "_area_x" in inter.columns else ("_area_y" if "_area_y" in inter.columns else None))
    if denom_col is None:
        raise RuntimeError("Could not find tract area column after overlay/merge.")
    inter["area_frac"] = inter["_iarea"] / inter[denom_col].replace(0, np.nan)

    if pop_by_tract is not None:
        pop = pop_by_tract.copy()
        pop["tract_pop"] = pd.to_numeric(pop["tract_pop"], errors="coerce").fillna(0.0)
        inter = inter.merge(pop[["geoid", "tract_pop"]], on="geoid", how="left")
        inter["pop_part"] = inter["area_frac"].fillna(0) * inter["tract_pop"].fillna(0)
        denom = inter.groupby("geoid")["pop_part"].transform("sum")
        inter["weight"] = np.where(denom > 0, inter["pop_part"] / denom, inter["area_frac"].fillna(0))
    else:
        inter["weight"] = inter["area_frac"].fillna(0)

    xw = inter[["zip", "geoid", "weight"]].rename(columns={"geoid": "tract"})
    xw = xw[xw["weight"] > 1e-10]
    xw["zip"] = xw["zip"].astype(str).str.zfill(5)
    return xw[["zip", "tract", "weight"]]


def load_hud_xwalk(path: str) -> pd.DataFrame:
    xw = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in xw.columns}
    def pick(*opts):
        for o in opts:
            if o in cols:
                return cols[o]
        return None
    zip_col = pick("zip", "zipcode", "zip_code", "zcta5")
    tract_col = pick("tract", "geoid", "geotract", "tract_geoid")
    w_col = pick("res_ratio", "residential_ratio", "ratio_res", "ratio")
    if not (zip_col and tract_col and w_col):
        raise ValueError("HUD crosswalk must have ZIP, TRACT, and RES_RATIO (or equivalent)")
    out = xw[[zip_col, tract_col, w_col]].copy()
    out.columns = ["zip", "tract", "weight"]
    out["zip"] = out["zip"].astype(str).str.zfill(5)
    out["tract"] = out["tract"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


# -----------------------
# Aggregation
# -----------------------

def aggregate_zip(tract_stats: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:
    t = tract_stats.copy()
    x = xwalk.copy()
    t["tract6"] = t["geoid"].str[-6:]
    if x["tract"].str.len().median() >= 10:
        x["tract11"] = x["tract"].str[-11:]
        m = t.merge(x[["zip", "tract11", "weight"]], left_on="geoid", right_on="tract11", how="inner")
    else:
        m = t.merge(x[["zip", "tract", "weight"]], left_on="tract6", right_on="tract", how="inner")

    m["w_est"] = m["estimate"] * m["weight"]
    m["w_moe"] = m["moe"] * m["weight"]

    def rss(arr: pd.Series) -> float:
        return float(np.sqrt(np.sum(np.square(arr.values))))

    out = m.groupby("zip", as_index=False).agg(
        estimate=("w_est", "sum"),
        moe=("w_moe", rss)
    )
    out["estimate"] = out["estimate"].round(0).astype(int)
    out["moe"] = out["moe"].round(0).astype(int)
    return out


# -----------------------
# Optional population fetch
# -----------------------

def fetch_tract_pop(year: int, state_fips: str, county_filter: set[str]) -> pd.DataFrame:
    import requests
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": B01003_TOTAL_E, "for": "tract:*", "in": f"state:{state_fips}+county:*"}
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    df = df[df["county"].isin(county_filter)].copy()
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df["tract_pop"] = pd.to_numeric(df[B01003_TOTAL_E], errors="coerce").fillna(0.0)
    return df[["geoid", "tract_pop"]]


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate tract CSV to ZIPs using local shapes (no re-downloads)")
    ap.add_argument("--tracts-csv", required=True, help="Path to outputs/acs_tracts_guyanese.csv")
    ap.add_argument("--zcta-shp", required=True, help="Path to tl_2020_us_zcta510.shp")
    ap.add_argument("--ny-tract-shp", required=True, help="Path to tl_2020_36_tract.shp")
    ap.add_argument("--fl-tract-shp", required=True, help="Path to tl_2020_12_tract.shp")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--hud-xwalk", default=None, help="Optional HUD ZIP↔tract CSV to use instead of overlay")
    ap.add_argument("--make-geojson", action="store_true", help="Also export zip_guyanese.geojson by joining ZCTA geometry")
    ap.add_argument("--fetch-pop", action="store_true", help="Use population-weighted overlay (fetch ACS tract totals)")
    ap.add_argument("--year", type=int, default=2023, help="ACS 5y year for optional pop fetch")
    ap.add_argument("--tract-pop-csv", default=None, help="Optional local CSV with columns geoid,tract_pop")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print("[1/5] Reading tract stats...")
    tstats = read_tract_stats(args.tracts_csv)

    if args.hud_xwalk:
        print("[2/5] Loading HUD crosswalk (skip overlay)...")
        xwalk = load_hud_xwalk(args.hud_xwalk)
    else:
        print("[2/5] Reading local tract & ZCTA shapes...")
        tr_gdf = read_tract_shapes(args.ny_tract_shp, args.fl_tract_shp)
        zcta_gdf = read_zcta_shapes(args.zcta_shp)

        # Optional population attachment
        pop_df = None
        if args.tract_pop_csv and os.path.exists(args.tract_pop_csv):
            print("[3/5] Using local tract population CSV...")
            pop_df = pd.read_csv(args.tract_pop_csv, dtype={"geoid": str})[["geoid", "tract_pop"]]
        elif args.fetch_pop:
            print("[3/5] Fetching tract populations from ACS (small request)...")
            ny_pop = fetch_tract_pop(args.year, FIPS_STATES["NY"], NYC_COUNTIES)
            fl_pop = fetch_tract_pop(args.year, FIPS_STATES["FL"], SOFLA_COUNTIES_FL)
            pop_df = pd.concat([ny_pop, fl_pop], ignore_index=True)
        else:
            print("[3/5] Skipping population fetch (area-weighted overlay)...")

        print("[4/5] Building tract→ZCTA overlay crosswalk...")
        xwalk = build_overlay_xwalk(tr_gdf, zcta_gdf, pop_by_tract=pop_df)

    print("[5/5] Aggregating to ZIPs...")
    zip_df = aggregate_zip(tstats, xwalk)
    out_csv = os.path.join(args.outdir, "zip_guyanese.csv")
    zip_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(zip_df):,} ZIPs)")

    if args.make_geojson:
        print("[+] Creating GeoJSON by joining ZCTA shapes...")
        zcta_gdf = read_zcta_shapes(args.zcta_shp).to_crs(4326)
        zip_gdf = zcta_gdf.merge(zip_df, on="zip", how="inner")
        out_geo = os.path.join(args.outdir, "zip_guyanese.geojson")
        zip_gdf.to_file(out_geo, driver="GeoJSON")
        print(f"Saved {out_geo}")


if __name__ == "__main__":
    main()
