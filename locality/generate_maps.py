#!/usr/bin/env python3
"""
Generate maps (static PNG + interactive HTML) for ZIPs and NYC NTAs
from the ACS/PUMS Guyanese ancestry outputs you already created.

Inputs (any that exist will be used):
- outputs/zip_guyanese.csv               (zip, estimate, moe)
- outputs/zip_guyanese.geojson           (optional; if present we can use directly)
- data/zcta/tl_2020_us_zcta510.shp       (fallback geometry for ZIPs/ZCTAs)
- outputs/nta_guyanese_nyc.csv           (nta[, ntaname], estimate, moe)
- data/nyc_ntas_2020.geojson             (geometry for NYC NTAs)

Outputs (default folder: maps/):
- maps/zip_choropleth.png
- maps/zip_choropleth.html
- maps/nta_choropleth.png
- maps/nta_choropleth.html

Usage
-----
# Install deps (if needed):
#   pip install geopandas matplotlib mapclassify folium pandas numpy shapely pyproj

# Example (South Florida focus with a custom title)
python generate_maps.py \
  --zip-csv outputs/zip_guyanese.csv \
  --zip-geo outputs/zip_guyanese_sofla.geojson \
  --zcta-shp data/zcta/tl_2020_us_zcta510.shp \
  --outdir maps \
  --scheme quantiles \
  --k 6 \
  --column estimate \
  --title-zip "South Florida — Guyanese ancestry by ZIP"

# NYC NTA
python generate_maps.py \
  --nta-csv outputs/nta_guyanese_nyc.csv \
  --nta-geo data/nyc_ntas_2020.geojson \
  --outdir maps \
  --scheme quantiles \
  --k 6 \
  --column estimate \
  --title-nta "NYC — Guyanese ancestry by NTA"

You can switch --column to 'moe' or 'cv' (coefficient of variation) to map reliability.

New options:
- --title-zip / --title-nta : set custom titles.
- --bbox-zip / --bbox-nta   : clip/zoom static + HTML maps to a lon/lat bbox (MINX MINY MAXX MAXY).
"""
from __future__ import annotations
import os
import argparse
import warnings
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify as mc
import folium
from shapely.geometry import box

# ----------------------
# Helpers
# ----------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def calc_cv(df: pd.DataFrame, est_col: str = "estimate", moe_col: str = "moe") -> pd.Series:
    """Compute coefficient of variation using ACS MOE at 90% CI: SE = MOE / 1.645; CV = SE / estimate."""
    se = pd.to_numeric(df.get(moe_col, np.nan), errors="coerce") / 1.645
    est = pd.to_numeric(df.get(est_col, np.nan), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = se / est
    return cv.replace([np.inf, -np.inf], np.nan)


def _classify(series: pd.Series, scheme: str = "quantiles", k: int = 6):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    s = s.replace([np.inf, -np.inf], 0)
    uniq = np.unique(s)
    if len(uniq) <= 1:
        return None, None
    scheme = scheme.lower()
    try:
        if scheme in {"quantiles", "quantile", "q"}:
            b = mc.Quantiles(s, k=k)
        elif scheme in {"naturalbreaks", "jenks", "fisherjenks", "nb"}:
            b = mc.NaturalBreaks(s, k=min(k, len(uniq)-1))
        elif scheme in {"equal", "equalinterval"}:
            b = mc.EqualInterval(s, k=k)
        else:
            b = mc.Quantiles(s, k=k)
        return b, b.yb
    except Exception:
        return None, None


def _center_from_bounds(bounds: Iterable[float]) -> Tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    return [(miny + maxy) / 2.0, (minx + maxx) / 2.0]


def _clip_bbox(gdf: gpd.GeoDataFrame, bbox: Optional[Iterable[float]]) -> gpd.GeoDataFrame:
    if not bbox:
        return gdf
    minx, miny, maxx, maxy = map(float, bbox)
    return gpd.clip(gdf, box(minx, miny, maxx, maxy))


# ----------------------
# ZIP maps
# ----------------------

def load_zip_geo(zip_csv: str, zip_geo: Optional[str], zcta_shp: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not os.path.exists(zip_csv):
        print(f"[ZIP] Missing CSV: {zip_csv}")
        return None
    z = pd.read_csv(zip_csv, dtype={"zip": str})
    z["zip"] = z["zip"].str.zfill(5)
    if "cv" not in z.columns and {"estimate", "moe"}.issubset(z.columns):
        z["cv"] = calc_cv(z)

    gdf = None
    if zip_geo and os.path.exists(zip_geo):
        try:
            gdf = gpd.read_file(zip_geo)
            if "zip" not in gdf.columns:
                for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "GEOID"):
                    if c in gdf.columns:
                        gdf["zip"] = gdf[c].astype(str).str.zfill(5)
                        break
        except Exception as e:
            print(f"[ZIP] Failed reading {zip_geo}: {e}")
    if gdf is None and zcta_shp and os.path.exists(zcta_shp):
        try:
            gdf = gpd.read_file(zcta_shp)
            for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "GEOID"):
                if c in gdf.columns:
                    gdf["zip"] = gdf[c].astype(str).str.zfill(5)
                    break
        except Exception as e:
            print(f"[ZIP] Failed reading {zcta_shp}: {e}")
    if gdf is None:
        print("[ZIP] No geometry found. Provide --zip-geo or --zcta-shp.")
        return None

    gdf = gdf.to_crs(4326)
    m = gdf.merge(z, on="zip", how="inner")

    # Coalesce duplicate columns created by merge (prefer CSV side: *_y)
    for col in ("estimate", "moe"):
        cx, cy = f"{col}_x", f"{col}_y"
        if cx in m.columns and cy in m.columns:
            m[col] = m[cy].where(~m[cy].isna(), m[cx])
            m = m.drop(columns=[cx, cy])
    if "cv" not in m.columns and {"estimate", "moe"}.issubset(m.columns):
        m["cv"] = calc_cv(m)

    if m.empty:
        print("[ZIP] Join produced 0 rows — check that ZIPs in CSV match your geometry.")
    return m


def make_zip_maps(zip_gdf: gpd.GeoDataFrame, outdir: str, column: str = "estimate",
                  scheme: str = "quantiles", k: int = 6, title: Optional[str] = None,
                  bbox: Optional[Iterable[float]] = None):
    ensure_dir(outdir)
    title = title or f"Guyanese ancestry by ZIP ({column})"

    # Optional clip/zoom
    g = _clip_bbox(zip_gdf, bbox)

    # Static PNG (plot in projected CRS so it fills the canvas nicely)
    fig, ax = plt.subplots(figsize=(8, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brks, _ = _classify(g[column], scheme=scheme, k=k)
        legend_kwds = {"loc": "center left", "bbox_to_anchor": (1.02, 0.5), "title": column}
        if brks is not None:
            g.to_crs(3857).plot(
                column=column,
                scheme=brks.__class__.__name__, k=k,
                ax=ax, legend=True, legend_kwds=legend_kwds,
                linewidth=0.2, edgecolor="black"
            )
        else:
            g.to_crs(3857).plot(ax=ax, linewidth=0.2, edgecolor="black")
    ax.set_axis_off()
    ax.set_title(title)
    plt.subplots_adjust(right=0.84)  # make space for legend
    png_path = os.path.join(outdir, "zip_choropleth.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[ZIP] Wrote {png_path}")

    # Interactive HTML
    try:
        minx, miny, maxx, maxy = g.total_bounds
        mcenter = _center_from_bounds([minx, miny, maxx, maxy])
        fmap = folium.Map(location=mcenter, zoom_start=8, tiles="cartodbpositron")

        fields  = [c for c in ["zip", "estimate", "moe", "cv"] if c in g.columns]
        aliases = ["ZIP", "Estimate", "MOE", "CV (SE/Est)"][:len(fields)]
        gj = folium.GeoJson(g.to_json(),
                            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases))
        gj.add_to(fmap)
        fmap.fit_bounds([[miny, minx], [maxy, maxx]])

        # Title badge
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 9999; background: rgba(255,255,255,0.9); padding: 6px 10px;
                    border-radius: 6px; font-weight: 600; font-size: 16px;">
            {title}
        </div>'''
        fmap.get_root().html.add_child(folium.Element(title_html))

        html_path = os.path.join(outdir, "zip_choropleth.html")
        fmap.save(html_path)
        print(f"[ZIP] Wrote {html_path}")
    except Exception as e:
        print(f"[ZIP] HTML map failed: {e}")


# ----------------------
# NYC NTA maps
# ----------------------

def load_nta_geo(nta_csv: str, nta_geo: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not os.path.exists(nta_csv):
        print(f"[NTA] Missing CSV: {nta_csv}")
        return None
    n = pd.read_csv(nta_csv, dtype=str)
    for c in ("estimate", "moe"):
        if c in n.columns:
            n[c] = pd.to_numeric(n[c], errors="coerce")
    if "cv" not in n.columns and {"estimate", "moe"}.issubset(n.columns):
        n["cv"] = calc_cv(n)

    if "nta" not in n.columns:
        for c in n.columns:
            if c.lower() == "nta2020" or c.lower() == "ntacode":
                n = n.rename(columns={c: "nta"})
                break
    if "nta" not in n.columns:
        raise ValueError("[NTA] CSV must contain an 'nta' column.")

    if nta_geo and os.path.exists(nta_geo):
        g = gpd.read_file(nta_geo)
        if "nta" not in g.columns:
            for c in ("NTA2020", "NTACode", "NTACODE"):
                if c in g.columns:
                    g["nta"] = g[c].astype(str)
                    break
        if "ntaname" not in g.columns and "NTAName" in g.columns:
            g = g.rename(columns={"NTAName": "ntaname"})
        g = g.to_crs(4326)
        m = g.merge(n, on="nta", how="inner")
        if m.empty:
            print("[NTA] Join produced 0 rows — check that NTA codes match your geometry.")
        return m
    print("[NTA] No geometry found. Provide --nta-geo.")
    return None


def make_nta_maps(nta_gdf: gpd.GeoDataFrame, outdir: str, column: str = "estimate",
                  scheme: str = "quantiles", k: int = 6, title: Optional[str] = None,
                  bbox: Optional[Iterable[float]] = None):
    ensure_dir(outdir)
    title = title or f"Guyanese ancestry by NYC NTA ({column})"

    g = _clip_bbox(nta_gdf, bbox)

    # Static PNG
    fig, ax = plt.subplots(figsize=(8, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brks, _ = _classify(g[column], scheme=scheme, k=k)
        legend_kwds = {"loc": "center left", "bbox_to_anchor": (1.02, 0.5), "title": column}
        if brks is not None:
            g.to_crs(3857).plot(
                column=column,
                scheme=brks.__class__.__name__, k=k,
                ax=ax, legend=True, legend_kwds=legend_kwds,
                linewidth=0.2, edgecolor="black"
            )
        else:
            g.to_crs(3857).plot(ax=ax, linewidth=0.2, edgecolor="black")
    ax.set_axis_off()
    ax.set_title(title)
    plt.subplots_adjust(right=0.84)
    png_path = os.path.join(outdir, "nta_choropleth.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"[NTA] Wrote {png_path}")

    # Interactive HTML
    try:
        minx, miny, maxx, maxy = g.total_bounds
        mcenter = _center_from_bounds([minx, miny, maxx, maxy])
        fmap = folium.Map(location=mcenter, zoom_start=10, tiles="cartodbpositron")
        fields = [c for c in ["nta", "ntaname", "estimate", "moe", "cv"] if c in g.columns]
        aliases = ["NTA", "Name", "Estimate", "MOE", "CV (SE/Est)"][:len(fields)]
        gj = folium.GeoJson(g.to_json(),
                            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases))
        gj.add_to(fmap)
        fmap.fit_bounds([[miny, minx], [maxy, maxx]])

        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 9999; background: rgba(255,255,255,0.9); padding: 6px 10px;
                    border-radius: 6px; font-weight: 600; font-size: 16px;">
            {title}
        </div>'''
        fmap.get_root().html.add_child(folium.Element(title_html))

        html_path = os.path.join(outdir, "nta_choropleth.html")
        fmap.save(html_path)
        print(f"[NTA] Wrote {html_path}")
    except Exception as e:
        print(f"[NTA] HTML map failed: {e}")


# ----------------------
# CLI
# ----------------------

def main():
    ap = argparse.ArgumentParser(description="Make ZIP and NYC NTA maps from ACS Guyanese outputs")
    ap.add_argument("--zip-csv", default="outputs/zip_guyanese.csv", help="Path to ZIP-level CSV")
    ap.add_argument("--zip-geo", default="outputs/zip_guyanese.geojson", help="Optional ZIP geojson; else use ZCTA shapefile")
    ap.add_argument("--zcta-shp", default="data/zcta/tl_2020_us_zcta510.shp", help="ZCTA shapefile path (fallback)")
    ap.add_argument("--nta-csv", default="outputs/nta_guyanese_nyc.csv", help="Path to NYC NTA CSV")
    ap.add_argument("--nta-geo", default="data/nyc_ntas_2020.geojson", help="NYC NTA geometry (GeoJSON/Shapefile)")
    ap.add_argument("--outdir", default="maps", help="Output folder for maps")
    ap.add_argument("--scheme", default="quantiles", help="Classification scheme: quantiles|jenks|equal")
    ap.add_argument("--k", type=int, default=6, help="Number of classes for choropleth")
    ap.add_argument("--column", default="estimate", help="Which column to map: estimate|moe|cv")

    # New niceties
    ap.add_argument("--title-zip", default=None, help="Custom title for ZIP map")
    ap.add_argument("--title-nta", default=None, help="Custom title for NTA map")
    ap.add_argument("--bbox-zip", nargs=4, type=float, metavar=("MINX","MINY","MAXX","MAXY"),
                    help="Clip/zoom ZIP map to this lon/lat bbox")
    ap.add_argument("--bbox-nta", nargs=4, type=float, metavar=("MINX","MINY","MAXX","MAXY"),
                    help="Clip/zoom NTA map to this lon/lat bbox")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # ZIP maps
    zip_gdf = load_zip_geo(args.zip_csv, args.zip_geo, args.zcta_shp)
    if zip_gdf is not None and args.column in zip_gdf.columns:
        make_zip_maps(zip_gdf, args.outdir, column=args.column, scheme=args.scheme, k=args.k,
                      title=args.title_zip, bbox=args.bbox_zip)
    elif zip_gdf is not None:
        print(f"[ZIP] Column '{args.column}' not found; available: {list(zip_gdf.columns)}")

    # NTA maps
    nta_gdf = load_nta_geo(args.nta_csv, args.nta_geo)
    if nta_gdf is not None and args.column in nta_gdf.columns:
        make_nta_maps(nta_gdf, args.outdir, column=args.column, scheme=args.scheme, k=args.k,
                      title=args.title_nta, bbox=args.bbox_nta)
    elif nta_gdf is not None:
        print(f"[NTA] Column '{args.column}' not found; available: {list(nta_gdf.columns)}")


if __name__ == "__main__":
    main()
