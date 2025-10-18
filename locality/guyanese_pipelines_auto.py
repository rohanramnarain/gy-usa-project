#!/usr/bin/env python3
"""
Guyanese Ancestry Pipelines — with Auto Downloads
=================================================

This single script ships two CLI workflows:

1) acs-tracts: Pull ACS 5-year tract-level "Guyanese" ancestry (B04006_045), then
   aggregate to ZIPs (via HUD ZIP↔tract crosswalk **or** auto-built tract↔ZCTA overlay)
   and to neighborhoods (via NYC tract↔NTA), and export ready-to-map CSVs and GeoJSONs
   for NYC & South Florida.

2) pums-puma: Aggregate ACS PUMS microdata for "Guyanese" ancestry to PUMAs for the top states
   by total Guyanese population (from ACS), returning a compact wide CSV with population,
   nativity split, education distribution, income (mean/median), and employment rate – suitable
   for dashboards.

Auto-downloads (new)
--------------------
If you pass `--auto`, the script will fetch any missing inputs into `./data`:
- ZCTA shapes (2020): TIGER/Line tl_2020_us_zcta510.zip
- NY & FL tract shapes (2020): tl_2020_36_tract.zip, tl_2020_12_tract.zip
- NYC NTA shapes (2020): ArcGIS FeatureServer → GeoJSON
- NYC tract→NTA crosswalk (2020) via NYC Open Data (Socrata) CSV
- ZIP↔tract crosswalk: If HUD file not supplied, builds a tract→ZCTA crosswalk by
  spatial overlay, weighting by tract total population (ACS B01003). This is a standard,
  transparent fallback when HUD RES_RATIO is unavailable.

Dependencies
------------
- Python 3.9+
- pandas, numpy, geopandas, requests, shapely, pyproj, tqdm

Install: `pip install pandas numpy geopandas requests shapely pyproj tqdm`
(Optional speedups: `pip install rtree`)

"""
from __future__ import annotations
import os
import io
import sys
import json
import math
import time
import zipfile
import tempfile
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Geo imports are optional at runtime depending on flags
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None  # allow non-geo runs

# --------------------------
# Helpers
# --------------------------

NYC_COUNTIES = {"005", "047", "061", "081", "085"}  # Bronx, Kings, New York, Queens, Richmond
SOFLA_COUNTIES_FL = {"011", "086", "099"}  # Broward, Miami-Dade, Palm Beach

FIPS_STATES = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72"
}

B04006_GUYANESE_E = "B04006_045E"
B04006_GUYANESE_M = "B04006_045M"
B01003_TOTAL_E = "B01003_001E"

DATA_DIR = "data"

# Official download URLs
URL_ZCTA_ZIP = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA5/tl_2020_us_zcta510.zip"
URL_TRACT_NY = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_36_tract.zip"
URL_TRACT_FL = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_12_tract.zip"
URL_NYC_NTA_GEOJSON = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0/"
    "query?where=1%3D1&outFields=*&f=geojson"
)
URL_NYC_TRACT_TO_NTA_CSV = "https://data.cityofnewyork.us/resource/hm78-6dwm.csv?$limit=50000"


def env_api_key() -> Optional[str]:
    return os.environ.get("CENSUS_API_KEY") or os.environ.get("CENSUS_KEY")


def census_get(url: str, params: Dict[str, str]) -> List[List[str]]:
    """GET wrapper that returns parsed JSON rows. Raises on HTTP errors."""
    key = env_api_key()
    if key:
        params = {**params, "key": key}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# --------------------------
# Part 1: ACS Summary Tables → tract → ZIP / Neighborhood
# --------------------------

def fetch_acs5_guyanese_tracts(year: int, state_fips: str,
                                county_filter: Optional[set[str]] = None) -> pd.DataFrame:
    """Fetch ACS 5-year tract-level Guyanese (estimate+MOE) for a state; optionally filter counties.

    Returns columns: state, county, tract, geoid, estimate, moe
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"{B04006_GUYANESE_E},{B04006_GUYANESE_M}",
        "for": "tract:*",
        "in": f"state:{state_fips}+county:*"
    }
    rows = census_get(base, params)
    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    # Cast & filter
    df[B04006_GUYANESE_E] = pd.to_numeric(df[B04006_GUYANESE_E], errors="coerce").fillna(0).astype(float)
    df[B04006_GUYANESE_M] = pd.to_numeric(df[B04006_GUYANESE_M], errors="coerce").fillna(0).astype(float)
    if county_filter:
        df = df[df["county"].isin(county_filter)].copy()
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df = df.rename(columns={B04006_GUYANESE_E: "estimate", B04006_GUYANESE_M: "moe"})
    return df[["state", "county", "tract", "geoid", "estimate", "moe"]]


def fetch_acs5_total_pop_tracts(year: int, state_fips: str,
                                 county_filter: Optional[set[str]] = None) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"{B01003_TOTAL_E}",
        "for": "tract:*",
        "in": f"state:{state_fips}+county:*"
    }
    rows = census_get(base, params)
    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    df[B01003_TOTAL_E] = pd.to_numeric(df[B01003_TOTAL_E], errors="coerce").fillna(0).astype(float)
    if county_filter:
        df = df[df["county"].isin(county_filter)].copy()
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df = df.rename(columns={B01003_TOTAL_E: "tract_pop"})
    return df[["geoid", "tract_pop"]]


def load_hud_zip_tract_xwalk(path: str) -> pd.DataFrame:
    """Load a HUD USPS ZIP↔Tract crosswalk and normalize columns.

    Expected columns (case-insensitive): 'ZIP', 'TRACT', and a weight column like 'RES_RATIO'.
    If multiple ratio columns exist, use RES_RATIO by default.
    """
    xw = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in xw.columns}
    # Normalize
    def pick(*options):
        for opt in options:
            if opt in cols:
                return cols[opt]
        return None

    zip_col = pick("zip", "zip_code", "zipcode", "zcta5")
    tract_col = pick("tract", "census_tract", "geoid", "tract_geoid")
    res_ratio_col = pick("res_ratio", "residential_ratio", "ratio_res", "ratio")
    if not (zip_col and tract_col and res_ratio_col):
        raise ValueError("Crosswalk must include ZIP, TRACT, and RES_RATIO (or equivalent) columns.")
    # Clean types
    xw = xw[[zip_col, tract_col, res_ratio_col]].copy()
    xw.columns = ["zip", "tract", "weight"]
    xw["zip"] = xw["zip"].str.zfill(5)
    xw["tract"] = xw["tract"].str.extract(r"(\d+)")
    xw["weight"] = pd.to_numeric(xw["weight"], errors="coerce").fillna(0.0)
    return xw


def load_tract_to_nta_xwalk_nyc(path: str) -> pd.DataFrame:
    """Load a NYC 2020 Census tract → NTA crosswalk."""
    xw = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in xw.columns}
    tract_col = None
    for opt in ("tract", "geoid", "tract_geoid", "bctcb2020", "tractid", "geoid20"):
        if opt in cols:
            tract_col = cols[opt]
            break
    nta_col = None
    for opt in ("nta2020", "ntacode", "nta_code", "nta"):
        if opt in cols:
            nta_col = cols[opt]
            break
    ntaname_col = None
    for opt in ("ntaname", "nta_name", "name", "ntaname2020"):
        if opt in cols:
            ntaname_col = cols[opt]
            break
    if not (tract_col and nta_col):
        raise ValueError("NTA crosswalk must include tract and NTA code columns.")
    xw = xw[[tract_col, nta_col] + ([ntaname_col] if ntaname_col else [])].copy()
    xw.columns = ["tract", "nta"] + (["ntaname"] if ntaname_col else [])
    xw["tract"] = xw["tract"].str.extract(r"(\d+)")
    return xw


def aggregate_to_zip(tract_df: pd.DataFrame, hud_xwalk: pd.DataFrame) -> pd.DataFrame:
    """Apportion tract estimates/MOEs into ZIPs using HUD residential weights.
    MOE_total = sqrt(sum((w*MOE_i)^2)).
    """
    t = tract_df.copy()
    t["tract6"] = t["geoid"].str[-6:]
    m = hud_xwalk.copy()
    # If crosswalk tract looks like 11-digit, merge on geoid; else on tract6
    if m["tract"].str.len().median() >= 10:
        m["tract11"] = m["tract"].str[-11:]
        merged = t.merge(m[["zip", "tract11", "weight"]], left_on="geoid", right_on="tract11", how="inner")
    else:
        merged = t.merge(m[["zip", "tract", "weight"]], left_on="tract6", right_on="tract", how="inner")
    merged["w_est"] = merged["estimate"] * merged["weight"]
    merged["w_moe"] = merged["moe"] * merged["weight"]
    agg = merged.groupby("zip", as_index=False).agg(
        estimate=("w_est", "sum"),
        moe=("w_moe", lambda s: float(np.sqrt(np.sum(np.square(s)))))
    )
    agg["estimate"] = agg["estimate"].round(0).astype(int)
    agg["moe"] = agg["moe"].round(0).astype(int)
    return agg


def aggregate_to_nta_nyc(tract_df: pd.DataFrame, nta_xwalk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tract estimates/MOEs to NYC NTA (tracts nest in NTAs).
    MOE_total = sqrt(sum(MOE_i^2)).
    """
    t = tract_df.copy()
    t["tract6"] = t["geoid"].str[-6:]
    x = nta_xwalk.copy()
    if x["tract"].str.len().median() >= 10:
        x["tract11"] = x["tract"].str[-11:]
        merged = t.merge(x[["tract11", "nta"] + (["ntaname"] if "ntaname" in x.columns else [])],
                         left_on="geoid", right_on="tract11", how="inner")
    else:
        merged = t.merge(x[["tract", "nta"] + (["ntaname"] if "ntaname" in x.columns else [])],
                         left_on="tract6", right_on="tract", how="inner")

    def rss(s: pd.Series) -> float:
        return float(np.sqrt(np.sum(np.square(s))))

    group_cols = ["nta"] + (["ntaname"] if "ntaname" in merged.columns else [])
    agg = merged.groupby(group_cols, as_index=False).agg(
        estimate=("estimate", "sum"),
        moe=("moe", rss)
    )
    agg["estimate"] = agg["estimate"].round(0).astype(int)
    agg["moe"] = agg["moe"].round(0).astype(int)
    return agg


# ---------- Auto helpers ----------

def download_and_unzip(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    local_zip = os.path.join(out_dir, os.path.basename(url))
    if not os.path.exists(local_zip):
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
    with zipfile.ZipFile(local_zip, 'r') as zf:
        zf.extractall(out_dir)
    return out_dir


def auto_get_zcta_shapes() -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    zdir = os.path.join(DATA_DIR, "zcta")
    shp = os.path.join(zdir, "tl_2020_us_zcta510.shp")
    if not os.path.exists(shp):
        download_and_unzip(URL_ZCTA_ZIP, zdir)
    gdf = gpd.read_file(shp)
    for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "GEOID"):
        if c in gdf.columns:
            gdf["zip"] = gdf[c].astype(str).str.zfill(5)
            break
    gdf = gdf.to_crs(4326)
    return gdf[["zip", "geometry"]]


def auto_get_tract_shapes(states: List[str]) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    out = []
    for st in states:
        if st == "36":
            url = URL_TRACT_NY
        elif st == "12":
            url = URL_TRACT_FL
        else:
            raise ValueError(f"Auto only supports NY (36) and FL (12) right now; got {st}")
        sdir = os.path.join(DATA_DIR, f"tract_{st}")
        shp = os.path.join(sdir, f"tl_2020_{st}_tract.shp")
        if not os.path.exists(shp):
            download_and_unzip(url, sdir)
        g = gpd.read_file(shp)
        # Normalize GEOID
        g["geoid"] = g.get("GEOID", g.get("GEOID20", g.get("GEOID10"))).astype(str)
        out.append(g[["geoid", "STATEFP", "COUNTYFP", "TRACTCE", "geometry"]])
    gdf = pd.concat(out)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=g.crs).to_crs(4326)
    return gdf


def auto_get_nyc_nta_shapes() -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    out_path = os.path.join(DATA_DIR, "nyc_ntas_2020.geojson")
    if not os.path.exists(out_path):
        r = requests.get(URL_NYC_NTA_GEOJSON, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    gdf = gpd.read_file(out_path)
    # Standardize
    if "NTA2020" in gdf.columns:
        gdf["nta"] = gdf["NTA2020"].astype(str)
    elif "NTACode" in gdf.columns:
        gdf["nta"] = gdf["NTACode"].astype(str)
    if "NTAName" in gdf.columns and "ntaname" not in gdf.columns:
        gdf = gdf.rename(columns={"NTAName": "ntaname"})
    keep = [c for c in ("nta", "ntaname", "geometry") if c in gdf.columns]
    return gdf[keep].to_crs(4326)


def auto_get_nyc_tract_to_nta() -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, "nyc_2020_tract_to_nta.csv")
    if not os.path.exists(out_path):
        r = requests.get(URL_NYC_TRACT_TO_NTA_CSV, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    return load_tract_to_nta_xwalk_nyc(out_path)


def build_tract_to_zcta_xwalk_overlay(tracts_gdf: "gpd.GeoDataFrame",
                                       zcta_gdf: "gpd.GeoDataFrame",
                                       pop_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create a tract→ZCTA crosswalk with weights based on the share of each tract's population
    that falls inside each ZCTA, approximated by area intersection × tract population.
    If pop_df is None, fall back to pure area fraction.
    Returns columns: zip, tract, weight
    """
    # Reproject to an equal-area CRS for robust area computations
    tr = tracts_gdf[["geoid", "STATEFP", "COUNTYFP", "geometry"]].to_crs(5070)
    z = zcta_gdf[["zip", "geometry"]].to_crs(5070)

    # Compute tract areas
    tr["tract_area"] = tr.geometry.area

    # Intersections
    print("Building tract→ZCTA overlay; this can take a few minutes on first run...")
    inter = gpd.overlay(tr, z, how="intersection", keep_geom_type=False)
    inter["inter_area"] = inter.geometry.area

    # Area fraction within tract
    inter = inter.merge(tr[["geoid", "tract_area"]], on="geoid", how="left")
    inter["area_frac"] = inter["inter_area"] / inter["tract_area"]

    # Attach population by tract if provided
    if pop_df is not None:
        pop_df = pop_df.copy()
        inter = inter.merge(pop_df, left_on="geoid", right_on="geoid", how="left")
        inter["pop_part"] = inter["area_frac"].fillna(0) * inter["tract_pop"].fillna(0)
        # Normalize to fractions per tract
        sums = inter.groupby("geoid")["pop_part"].transform("sum")
        inter["weight"] = np.where(sums > 0, inter["pop_part"] / sums, inter["area_frac"].fillna(0))
    else:
        inter["weight"] = inter["area_frac"].fillna(0)

    xwalk = inter[["zip", "geoid", "weight"]].copy()
    xwalk = xwalk.rename(columns={"geoid": "tract"})
    # Clean
    xwalk["zip"] = xwalk["zip"].astype(str).str.zfill(5)
    xwalk["tract"] = xwalk["tract"].astype(str)
    xwalk = xwalk[xwalk["weight"] > 1e-9]
    return xwalk[["zip", "tract", "weight"]]


# --------------------------
# Part 2: PUMS microdata → PUMA dashboard table
# --------------------------

PUMS_ANCESTRY_CODE_GUYANESE = "370"


def fetch_top_states_by_guyanese(n: int = 5, year: int = 2023) -> List[str]:
    base = f"https://api.census.gov/data/{year}/acs/acs1"
    params = {
        "get": f"{B04006_GUYANESE_E},state",
        "for": "state:*"
    }
    rows = census_get(base, params)
    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    df[B04006_GUYANESE_E] = pd.to_numeric(df[B04006_GUYANESE_E], errors="coerce").fillna(0)
    df = df.sort_values(B04006_GUYANESE_E, ascending=False).head(n)
    return df["state"].tolist()


def pums_get_people(year: int, state_fips: str, extra_predicates: Optional[Dict[str, str]] = None,
                    page_size: int = 50000) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs1/pums"
    variables = [
        "PUMA", "PWGTP", "ANC1P", "ANC2P", "SCHL", "ESR", "PINCP", "NATIVITY", "AGEP"
    ]
    params = {
        "get": ",".join(variables),
        "for": "public use microdata area:*",
        "in": f"state:{state_fips}",
    }
    if extra_predicates:
        params.update(extra_predicates)

    dfs = []
    page = 1
    while True:
        p = dict(params)
        p["page"] = str(page)
        p["per_page"] = str(page_size)
        rows = census_get(base, p)
        if not rows or len(rows) <= 1:
            break
        header, *data = rows
        df = pd.DataFrame(data, columns=header)
        dfs.append(df)
        if len(data) < page_size:
            break
        page += 1
        if page > 50:
            break
    if not dfs:
        return pd.DataFrame(columns=variables + ["state", "public use microdata area"])
    out = pd.concat(dfs, ignore_index=True)
    for col in ("PWGTP", "SCHL", "ESR", "PINCP", "NATIVITY", "AGEP"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def weighted_sum(s: pd.Series, w: pd.Series) -> float:
    return float(np.nansum(s.values * w.values))


def weighted_mean(s: pd.Series, w: pd.Series) -> float:
    wsum = float(np.nansum(w.values))
    if wsum == 0:
        return float("nan")
    return float(np.nansum(s.values * w.values) / wsum)


def weighted_median(x: pd.Series, w: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "w": w}).dropna().sort_values("x")
    if df.empty:
        return float("nan")
    cumsum = df["w"].cumsum()
    cutoff = df["w"].sum() / 2.0
    return float(df.loc[cumsum >= cutoff, "x"].iloc[0])


def pums_classify_education(schl: pd.Series) -> pd.Categorical:
    schl = pd.to_numeric(schl, errors="coerce").fillna(-1).astype(int)
    cats = pd.Series(index=schl.index, dtype=object)
    cats[(schl >= 1) & (schl <= 15)] = "<HS"
    cats[schl == 16] = "HS"
    cats[(schl >= 17) & (schl <= 21)] = "Some/AA"
    cats[schl == 22] = "BA"
    cats[(schl >= 23) & (schl <= 25)] = "Grad+"
    cats = cats.astype("category")
    return cats


def pums_workflow(args: argparse.Namespace) -> None:
    year = args.year
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if args.states and args.states.lower() != "auto":
        states = [FIPS_STATES[s.strip().upper()] for s in args.states.split(',') if s.strip()]
    else:
        print("Selecting top states by Guyanese population from ACS 1-year...")
        states = fetch_top_states_by_guyanese(n=args.topk, year=year)
    print(f"States FIPS included: {states}")

    frames = []
    for st in states:
        print(f"Downloading PUMS (year {year}) for state {st}...")
        df = pums_get_people(year, st)
        if df.empty:
            continue
        df["state"] = st
        frames.append(df)
    if not frames:
        print("No PUMS data retrieved.")
        return
    p = pd.concat(frames, ignore_index=True)

    ances = (p["ANC1P"].astype(str) == PUMS_ANCESTRY_CODE_GUYANESE) | (p["ANC2P"].astype(str) == PUMS_ANCESTRY_CODE_GUYANESE)
    p = p[ances].copy()
    p["is16+"] = p["AGEP"].fillna(0) >= 16
    p["edu_cat"] = pums_classify_education(p["SCHL"])
    p["native"] = (p["NATIVITY"].astype(float) == 1)
    p["foreign"] = (p["NATIVITY"].astype(float) == 2)

    esr = pd.to_numeric(p["ESR"], errors="coerce")
    p["emp"] = esr.isin([1, 2])
    p["unemp"] = esr.isin([3])
    p["in_labor"] = esr.isin([1, 2, 3])

    geo_cols = []
    for candidate in ("public use microdata area", "PUMA"):
        if candidate in p.columns:
            geo_cols.append(candidate)
            break
    if "state" not in p.columns:
        geo_cols.append("state")
    else:
        geo_cols = ["state"] + geo_cols

    def agg_func(group: pd.DataFrame) -> pd.Series:
        w = group["PWGTP"].fillna(0)
        pop = weighted_sum(pd.Series(np.ones(len(group))), w)
        native = weighted_sum(group["native"].astype(int), w)
        foreign = weighted_sum(group["foreign"].astype(int), w)
        shares = {}
        for cat in ["<HS", "HS", "Some/AA", "BA", "Grad+"]:
            shares[f"edu_{cat}"] = weighted_sum((group["edu_cat"] == cat).astype(int), w)
        inc_mean = weighted_mean(group["PINCP"].fillna(0), w)
        inc_median = weighted_median(group["PINCP"].fillna(0), w)
        g16 = group[group["is16+"]]
        w16 = g16["PWGTP"].fillna(0)
        emp = weighted_sum(g16["emp"].astype(int), w16)
        lf = weighted_sum(g16["in_labor"].astype(int), w16)
        emp_rate = float(emp / lf) if lf > 0 else float("nan")
        out = {
            "pop": pop,
            "native": native,
            "foreign": foreign,
            **shares,
            "income_mean": inc_mean,
            "income_median": inc_median,
            "employment_rate": emp_rate,
        }
        return pd.Series(out)

    wide = p.groupby(geo_cols).apply(agg_func).reset_index()

    count_cols = ["pop", "native", "foreign", "edu_<HS", "edu_HS", "edu_Some/AA", "edu_BA", "edu_Grad+"]
    for c in count_cols:
        if c in wide.columns:
            wide[c] = wide[c].round(0).astype(int)
    if "pop" in wide.columns and wide["pop"].gt(0).any():
        wide["pct_native"] = wide["native"] / wide["pop"]
        wide["pct_foreign"] = wide["foreign"] / wide["pop"]
        for cat in ["<HS", "HS", "Some/AA", "BA", "Grad+"]:
            col = f"edu_{cat}"
            wide[f"pct_{col}"] = wide[col] / wide["pop"]

    if args.puma_names_shp and gpd is not None:
        try:
            puma_gdf = gpd.read_file(args.puma_names_shp)
            puma_code_col = None
            for c in ("PUMACE20", "PUMACE10", "PUMACE", "PUMA", "GEOID10", "GEOID"):
                if c in puma_gdf.columns:
                    puma_code_col = c
                    break
            if puma_code_col is not None:
                puma_gdf["PUMA_CODE"] = puma_gdf[puma_code_col].astype(str).str.zfill(5)
                name_col = None
                for c in ("NAMELSAD10", "NAMELSAD20", "NAME", "NAMELSAD"):
                    if c in puma_gdf.columns:
                        name_col = c
                        break
                puma_names = puma_gdf[["PUMA_CODE"] + ([name_col] if name_col else [])].copy()
                puma_names = puma_names.rename(columns={name_col: "puma_name"} if name_col else {})
                if "public use microdata area" in wide.columns:
                    wide["PUMA_CODE"] = wide["public use microdata area"].astype(str).str.zfill(5)
                elif "PUMA" in wide.columns:
                    wide["PUMA_CODE"] = wide["PUMA"].astype(str).str.zfill(5)
                wide = wide.merge(puma_names, on="PUMA_CODE", how="left")
        except Exception as e:
            print(f"[Warning] Failed to attach PUMA names: {e}")

    out_csv = os.path.join(outdir, "pums_puma_guyanese_wide.csv")
    wide.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(wide):,} rows)")


# --------------------------
# ACS workflow (with auto)
# --------------------------

def acs_tracts_workflow(args: argparse.Namespace) -> None:
    year = args.year
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Fetch tracts for NY (NYC counties) and FL (South Florida counties)
    print("Downloading ACS 5-year tract-level Guyanese for NY (NYC) and FL (South Florida)...")
    ny_df = fetch_acs5_guyanese_tracts(year, FIPS_STATES["NY"], county_filter=NYC_COUNTIES)
    fl_df = fetch_acs5_guyanese_tracts(year, FIPS_STATES["FL"], county_filter=SOFLA_COUNTIES_FL)
    tracts_df = pd.concat([ny_df.assign(state_abbr="NY"), fl_df.assign(state_abbr="FL")], ignore_index=True)
    tracts_csv = os.path.join(outdir, "acs_tracts_guyanese.csv")
    tracts_df.to_csv(tracts_csv, index=False)
    print(f"Saved {tracts_csv} ({len(tracts_df):,} tracts)")

    # Decide crosswalk path
    hud_xwalk_df = None
    if args.hud_xwalk:
        try:
            hud_xwalk_df = load_hud_zip_tract_xwalk(args.hud_xwalk)
            print("Loaded HUD ZIP↔tract crosswalk from provided path.")
        except Exception as e:
            print(f"[Warning] Failed to read HUD crosswalk: {e}")

    # Auto mode: fetch/generate missing inputs
    if args.auto:
        os.makedirs(DATA_DIR, exist_ok=True)
        # NYC NTA crosswalk if missing
        if not args.nta_xwalk_nyc:
            try:
                print("Auto-fetching NYC tract→NTA crosswalk (2020) from NYC Open Data...")
                _ = auto_get_nyc_tract_to_nta()
                args.nta_xwalk_nyc = os.path.join(DATA_DIR, "nyc_2020_tract_to_nta.csv")
            except Exception as e:
                print(f"[Warning] Could not auto-fetch NYC tract→NTA crosswalk: {e}")
        # NTA shapes if missing
        if (not args.nta_shp_nyc) and (gpd is not None):
            try:
                print("Auto-fetching NYC NTA shapes (2020) GeoJSON...")
                _ = auto_get_nyc_nta_shapes()
                args.nta_shp_nyc = os.path.join(DATA_DIR, "nyc_ntas_2020.geojson")
            except Exception as e:
                print(f"[Warning] Could not auto-fetch NYC NTA shapes: {e}")
        # ZCTA shapes if missing and geopandas available
        if (not args.zcta_shp) and (gpd is not None):
            try:
                print("Auto-fetching ZCTA shapes (2020) from TIGER...")
                _ = auto_get_zcta_shapes()
                args.zcta_shp = os.path.join(DATA_DIR, "zcta", "tl_2020_us_zcta510.shp")
            except Exception as e:
                print(f"[Warning] Could not auto-fetch ZCTA shapes: {e}")
        # If no HUD crosswalk, try to build one via overlay using tracts + ZCTAs
        if (hud_xwalk_df is None) and (gpd is not None) and args.zcta_shp:
            try:
                print("Auto-building tract→ZCTA crosswalk via spatial overlay (weighted by tract population)...")
                # Load shapes
                tr_gdf = auto_get_tract_shapes([FIPS_STATES["NY"], FIPS_STATES["FL"]])
                # Filter to NYC + South Florida counties
                tr_gdf = tr_gdf[ tr_gdf["COUNTYFP"].isin(NYC_COUNTIES | SOFLA_COUNTIES_FL) ].copy()
                zcta_gdf = gpd.read_file(args.zcta_shp)
                # Get tract pop for weights
                ny_pop = fetch_acs5_total_pop_tracts(year, FIPS_STATES["NY"], county_filter=NYC_COUNTIES)
                fl_pop = fetch_acs5_total_pop_tracts(year, FIPS_STATES["FL"], county_filter=SOFLA_COUNTIES_FL)
                pop_df = pd.concat([ny_pop, fl_pop], ignore_index=True)
                hud_xwalk_df = build_tract_to_zcta_xwalk_overlay(tr_gdf, zcta_gdf, pop_df)
                # Save for reuse
                auto_xwalk = os.path.join(DATA_DIR, "auto_zip_tract_xwalk.csv")
                hud_xwalk_df.to_csv(auto_xwalk, index=False)
                print(f"Saved auto-built crosswalk → {auto_xwalk}")
            except Exception as e:
                print(f"[Warning] Could not auto-build tract→ZCTA crosswalk: {e}")

    # Crosswalk to ZIPs
    if hud_xwalk_df is not None:
        print("Aggregating to ZIPs using crosswalk...")
        zip_df = aggregate_to_zip(tracts_df, hud_xwalk_df)
        zip_csv = os.path.join(outdir, "zip_guyanese.csv")
        zip_df.to_csv(zip_csv, index=False)
        print(f"Saved {zip_csv} ({len(zip_df):,} ZIPs)")
    else:
        zip_df = None
        print("[Skip] ZIP aggregation (no crosswalk available; provide HUD CSV or run with --auto and geopandas installed)")

    # Crosswalk to NYC NTAs
    if args.nta_xwalk_nyc:
        try:
            print("Aggregating to NYC NTAs via tract→NTA crosswalk...")
            nta_xw = load_tract_to_nta_xwalk_nyc(args.nta_xwalk_nyc)
            nta_df = aggregate_to_nta_nyc(tracts_df[tracts_df["state_abbr"] == "NY"], nta_xw)
            nta_csv = os.path.join(outdir, "nta_guyanese_nyc.csv")
            nta_df.to_csv(nta_csv, index=False)
            print(f"Saved {nta_csv} ({len(nta_df):,} NTAs)")
        except Exception as e:
            print(f"[Warning] NTA aggregation failed: {e}")
            nta_df = None
    else:
        nta_df = None
        print("[Skip] NYC NTA aggregation (no NTA crosswalk available; pass --auto to fetch)")

    # GeoJSONs
    if args.zcta_shp and gpd is not None and zip_df is not None:
        try:
            print("Merging ZIP results with ZCTA shapes and exporting GeoJSON...")
            zcta_gdf = gpd.read_file(args.zcta_shp)
            # Standardize zip column
            for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "GEOID"):
                if c in zcta_gdf.columns:
                    zcta_gdf["zip"] = zcta_gdf[c].astype(str).str.zfill(5)
                    break
            zcta_gdf = zcta_gdf.to_crs(4326)
            zip_gdf = zcta_gdf.merge(zip_df, on="zip", how="inner")
            zip_geojson = os.path.join(outdir, "zip_guyanese.geojson")
            zip_gdf.to_file(zip_geojson, driver="GeoJSON")
            print(f"Saved {zip_geojson}")
        except Exception as e:
            print(f"[Warning] ZIP GeoJSON export failed: {e}")
    elif zip_df is not None:
        print("[Skip] ZIP GeoJSON (no ZCTA shapes or geopandas not installed)")

    if args.nta_shp_nyc and gpd is not None and ("nta_df" in locals()) and (nta_df is not None):
        try:
            print("Merging NTA results with NYC NTA shapes and exporting GeoJSON...")
            nta_shapes = gpd.read_file(args.nta_shp_nyc).to_crs(4326)
            # Standardize NTA code/name
            if "NTA2020" in nta_shapes.columns and "nta" not in nta_shapes.columns:
                nta_shapes["nta"] = nta_shapes["NTA2020"].astype(str)
            if "NTAName" in nta_shapes.columns and "ntaname" not in nta_shapes.columns:
                nta_shapes = nta_shapes.rename(columns={"NTAName": "ntaname"})
            nta_gdf = nta_shapes.merge(nta_df, on=[c for c in ("nta", "ntaname") if c in nta_df.columns], how="inner")
            nta_geojson = os.path.join(outdir, "nta_guyanese_nyc.geojson")
            nta_gdf.to_file(nta_geojson, driver="GeoJSON")
            print(f"Saved {nta_geojson}")
        except Exception as e:
            print(f"[Warning] NTA GeoJSON export failed: {e}")
    elif ("nta_df" in locals()) and (nta_df is not None):
        print("[Skip] NTA GeoJSON (no NTA shapes or geopandas not installed)")


# --------------------------
# CLI
# --------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Guyanese ancestry pipelines (ACS tracts + PUMS PUMAs)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Subcommand: acs-tracts
    p1 = sub.add_parser("acs-tracts", help="ACS 5-year tracts → ZIP & NYC NTA + GeoJSONs")
    p1.add_argument("--year", type=int, default=2023, help="ACS 5-year end year (e.g., 2023 = 2019–2023)")
    p1.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p1.add_argument("--hud-xwalk", type=str, default=None, help="Path to HUD ZIP↔tract crosswalk CSV (with RES_RATIO)")
    p1.add_argument("--nta-xwalk-nyc", type=str, default=None, help="Path to NYC 2020 tract→NTA crosswalk CSV")
    p1.add_argument("--zcta-shp", type=str, default=None, help="Path to ZCTA shapefile (.shp) or its directory")
    p1.add_argument("--nta-shp-nyc", type=str, default=None, help="Path or URL to NYC NTA shapes (GeoPackage/GeoJSON/Shapefile)")
    p1.add_argument("--auto", action="store_true", help="Auto-download/build required inputs into ./data if missing")
    p1.set_defaults(func=acs_tracts_workflow)

    # Subcommand: pums-puma
    p2 = sub.add_parser("pums-puma", help="PUMS person microdata → PUMA wide CSV for top states")
    p2.add_argument("--year", type=int, default=2023, help="ACS 1-year PUMS year")
    p2.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p2.add_argument("--topk", type=int, default=5, help="Top N states by Guyanese pop (from ACS 1-year)")
    p2.add_argument("--states", type=str, default="auto", help="Comma list of state abbreviations, or 'auto'")
    p2.add_argument("--puma-names-shp", type=str, default=None, help="Optional PUMA shapefile with names to join")
    p2.set_defaults(func=pums_workflow)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
