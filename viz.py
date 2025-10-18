# viz.py
# Publication-ready figures from the pipeline outputs (+ NY counties choropleth)

from pathlib import Path
import math
import zipfile
import warnings
import requests

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Try geopandas for the choropleth; fall back gracefully if missing
try:
    import geopandas as gpd
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# ---------- Paths ----------
OUT = Path("data/outputs")
RAW = Path("data/raw")
FIG = Path("figures")
RAW.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------- Matplotlib defaults ----------
plt.rcParams.update({
    "figure.dpi": 180,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------- Helpers ----------
def save_fig(name: str):
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=220, bbox_inches="tight")
    print(f"[ok] wrote {p}")
    plt.close()

def add_caption(caption: str):
    # small footnote/caption for a humanities audience
    plt.figtext(0.01, -0.06, caption, ha="left", va="top", fontsize=9)

def format_thousands(ax, axis='y'):
    # Safe thousands formatter for either axis (no ScalarFormatter required)
    if axis == 'x':
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

def annotate_bars(ax, horizontal=False, fmt="{:,.0f}", pad=3):
    if horizontal:
        for rect in ax.patches:
            x = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            ax.annotate(fmt.format(x), (x, y),
                        xytext=(pad, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9)
    else:
        for rect in ax.patches:
            y = rect.get_height()
            x = rect.get_x() + rect.get_width()/2
            ax.annotate(fmt.format(y), (x, y),
                        xytext=(0, 3), textcoords="offset points",
                        va="bottom", ha="center", fontsize=9)

# Optional: map state FIPS -> USPS postal
FIPS_TO_POSTAL = {
  "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL",
  "13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME",
  "24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
  "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
  "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"
}

# Safer minimal labels for RAC1P (only where unambiguous)
RAC1P_MAP = {
    1: "White alone",
    2: "Black alone",
    6: "Asian alone",
    8: "Some other race",
    9: "Two or more races",
    # Others left as code to avoid mislabeling
}

# ---------- Load data ----------
st = pd.read_csv(OUT / "pums_2023_state_weighted_counts.csv")           # STATE, weighted_count
nat = pd.read_csv(OUT / "pums_2023_nativity_weighted_counts.csv")       # NATIVITY, weighted_count
race = pd.read_csv(OUT / "pums_2023_race_weighted_counts.csv")          # RAC1P, weighted_count
anc = pd.read_csv(OUT / "acs5_state_ancestry_2023.csv").rename(columns={"guyanese_total_ancestry":"ancestry_tbl"})
bp  = pd.read_csv(OUT / "acs5_state_birthplace_guyana_2023.csv").rename(columns={"B05006_171E":"born_in_guyana"})

# Normalize dtypes/keys
st["STATE"]       = st["STATE"].astype(str).str.zfill(2)
anc["state_fips"] = anc["state_fips"].astype(str).str.zfill(2)
bp["state_fips"]  = bp["state_fips"].astype(str).str.zfill(2)
for df, cols in [
    (st, ["weighted_count"]),
    (nat, ["NATIVITY", "weighted_count"]),
    (race, ["RAC1P", "weighted_count"]),
    (anc, ["ancestry_tbl"]),
    (bp,  ["born_in_guyana"]),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Figure 1: Top states (clean bar + labels) ----------
st["state"] = st["STATE"].map(FIPS_TO_POSTAL).fillna(st["STATE"])
topN = 12
top_states = st.sort_values("weighted_count", ascending=False).head(topN)

fig, ax = plt.subplots(figsize=(9,5.5))
ax.bar(top_states["state"], top_states["weighted_count"]) 
ax.set_title("Where Are the Guyanese? (ACS PUMS 2023, weighted ancestry)")
ax.set_xlabel("State (postal)")
ax.set_ylabel("People (weighted)")
ax.grid(axis="y", alpha=0.25)
format_thousands(ax)
annotate_bars(ax)
add_caption("Source: ACS 1-year PUMS 2023 (ancestry code = Guyanese). Weights (PWGTP) applied.")
save_fig("01_top_states_bar.png")

# ---------- Figure 2: Nativity mix (clean pie) ----------
nat_map = {1: "U.S. native", 2: "Foreign-born"}
nat["NATIVITY"] = pd.to_numeric(nat["NATIVITY"], errors="coerce")
nat["label"] = nat["NATIVITY"].map(nat_map).fillna(nat["NATIVITY"].astype(str))

fig, ax = plt.subplots(figsize=(6.5,6.5))
ax.pie(nat["weighted_count"], labels=nat["label"], autopct="%1.0f%%", startangle=90)
ax.set_title("Nativity Among Guyanese-Ancestry Respondents (ACS PUMS 2023)")
add_caption("Note: 'U.S. native' includes U.S. born and certain U.S. territories per ACS definitions.")
save_fig("02_nativity_pie.png")

# ---------- Figure 3: Visibility index scatter (enhanced variants) ----------
combo = anc.merge(bp[["state_fips","born_in_guyana"]], on="state_fips", how="left") \
           .merge(st[["STATE","weighted_count"]], left_on="state_fips", right_on="STATE", how="left")
combo["state"] = combo["state_fips"].astype(str).str.zfill(2).map(FIPS_TO_POSTAL).fillna(combo["state_fips"])

x = pd.to_numeric(combo["born_in_guyana"], errors="coerce")
y = pd.to_numeric(combo["ancestry_tbl"], errors="coerce")
sizes = 20 + 80 * (combo["weighted_count"].fillna(0) / combo["weighted_count"].max()).pow(0.5)

# 3A) Symlog scatter to decongest low values while keeping zeros
fig, ax = plt.subplots(figsize=(8.6,6.8))
ax.scatter(x, y, s=sizes, alpha=0.8)
ax.set_xscale('symlog', linthresh=500)
ax.set_yscale('symlog', linthresh=500)
maxv = max(x.max(skipna=True), y.max(skipna=True))
ax.plot([1, maxv], [1, maxv])  # parity line (start at 1 to show on symlog)
for _, r in combo.sort_values("ancestry_tbl", ascending=False).head(10).iterrows():
    if pd.notna(r["born_in_guyana"]) and pd.notna(r["ancestry_tbl"]):
        ax.annotate(r["state"], (r["born_in_guyana"], r["ancestry_tbl"]),
                    xytext=(5,5), textcoords="offset points", fontsize=9)
ax.set_title("Beyond Immigration: Ancestry vs. Born-in-Guyana (symlog scale)")
ax.set_xlabel("Foreign-born from Guyana (B05006)")
ax.set_ylabel("Guyanese ancestry (B04004+B04005)")
ax.grid(True, alpha=0.3)
add_caption("Symlog scale spreads out small values without dropping zeros. Point size ∝ ancestry total.")
save_fig("03a_visibility_scatter_symlog.png")

# 3B) Linear scatter with zoomed inset for the congested cluster
fig, ax = plt.subplots(figsize=(8.6,6.8))
ax.scatter(x, y, s=sizes, alpha=0.8)
maxv = math.ceil(max(x.max(skipna=True), y.max(skipna=True)) * 1.05)
ax.plot([0, maxv], [0, maxv])
ax.set_title("Beyond Immigration: Ancestry vs. Born-in-Guyana (with inset)")
ax.set_xlabel("Foreign-born from Guyana (B05006)")
ax.set_ylabel("Guyanese ancestry (B04004+B04005)")
ax.grid(True, alpha=0.3)
format_thousands(ax)
format_thousands(ax, axis='x')
# annotate a few largest
for _, r in combo.sort_values("ancestry_tbl", ascending=False).head(6).iterrows():
    if pd.notna(r["born_in_guyana"]) and pd.notna(r["ancestry_tbl"]):
        ax.annotate(r["state"], (r["born_in_guyana"], r["ancestry_tbl"]),
                    xytext=(5,5), textcoords="offset points", fontsize=9)
# inset zoom
axins = inset_axes(ax, width="45%", height="40%", loc="lower right")
zoom_mask = (x <= 50000) & (y <= 50000)
axins.scatter(x[zoom_mask], y[zoom_mask], s=sizes[zoom_mask], alpha=0.9)
axins.set_xlim(-1000, 52000)
axins.set_ylim(-1000, 52000)
axins.grid(True, alpha=0.3)
axins.set_xticks([0,10000,20000,30000,40000,50000])
axins.set_yticks([0,10000,20000,30000,40000,50000])
axins.set_title("Zoom: ≤50k", fontsize=10)
add_caption("Inset shows the dense cluster (≤50k on both axes). Parity line in main plot.")
save_fig("03b_visibility_scatter_inset.png")

# 3C) Ratio plot: Visibility index vs. foreign-born size (clearer separation)
ratio = (y / (x.replace(0, pd.NA)))
fig, ax = plt.subplots(figsize=(8.6,6.2))
ax.scatter(x, ratio, s=sizes, alpha=0.85)
ax.axhline(1.0, linestyle='--')
ax.set_xscale('symlog', linthresh=500)
ax.set_title("Visibility Index (Ancestry / Born-in-Guyana) vs. Foreign-born size")
ax.set_xlabel("Foreign-born from Guyana (B05006)")
ax.set_ylabel("Visibility index (ancestry ÷ born-in-Guyana)")
ax.grid(True, alpha=0.3)
for _, r in combo.sort_values("ancestry_tbl", ascending=False).head(8).iterrows():
    if pd.notna(r["born_in_guyana"]) and pd.notna(r["ancestry_tbl"]):
        ax.annotate(r["state"], (r["born_in_guyana"], r["ancestry_tbl"] / max(r["born_in_guyana"], 1)),
                    xytext=(5,5), textcoords="offset points", fontsize=9)
add_caption("Above 1 ⇒ multi‑generation presence beyond immigrant counts. Point size ∝ ancestry total.")
save_fig("03c_visibility_ratio.png")

# ---------- Figure 4: RAC1P distribution (bar with human labels where safe) ----------
race_sorted = race.sort_values("weighted_count", ascending=False).head(8).copy()
race_sorted["label"] = race_sorted["RAC1P"].map(lambda v: RAC1P_MAP.get(int(v), f"RAC1P {int(v)}"))

fig, ax = plt.subplots(figsize=(9,5.5))
ax.bar(race_sorted["label"], race_sorted["weighted_count"])
ax.set_title("How Datasets Classify Guyanese (Top RAC1P among Guyanese Ancestry, PUMS 2023)")
ax.set_xlabel("Race category (RAC1P)")
ax.set_ylabel("People (weighted)")
ax.grid(axis="y", alpha=0.25)
format_thousands(ax)
annotate_bars(ax)
add_caption("RAC1P is ACS detailed race. Labels shown only when unambiguous; others kept as codes to avoid mislabeling.")
save_fig("04_race_codes_bar.png")

# ---------- Figure 5: NY counties bar (horizontal + labels) ----------
ny_file = OUT / "acs5_county_ancestry_2023_36.csv"
if ny_file.exists():
    ny = pd.read_csv(ny_file).sort_values("guyanese_total_ancestry", ascending=False).head(15).copy()
    if "NAME" in ny.columns:
        ny["label"] = ny["NAME"].str.replace(" County, New York", "", regex=False)
    else:
        ny["label"] = ny.get("county_fips", pd.Series(range(len(ny)))).astype(str)
    fig, ax = plt.subplots(figsize=(9,6.5))
    ny = ny.sort_values("guyanese_total_ancestry", ascending=True)  # for horizontal barh
    ax.barh(ny["label"], ny["guyanese_total_ancestry"])
    ax.set_title("NYC Epicenter: Guyanese Ancestry by County (ACS 5-year 2023)")
    ax.set_xlabel("People")
    ax.set_ylabel("County")
    ax.grid(axis="x", alpha=0.25)
    format_thousands(ax, axis='x')
    annotate_bars(ax, horizontal=True)
    add_caption("Self-reported Guyanese ancestry within New York State (5-year ACS for county stability).")
    save_fig("05_ny_counties_bar.png")
else:
    print("[warn] Skipping NYC counties bar: run `python guyanese_us_data_pipeline.py acs --geo county --state-fips 36` to generate.")

# ---------- Figure 6: NY counties choropleth ----------

def download_ny_county_shapes() -> Path | None:
    """Download Census cartographic county shapes (500k) and return path to the extracted .shp."""
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    zpath = RAW / "cb_2023_us_county_500k.zip"
    shp_dir = RAW / "cb_2023_us_county_500k"
    shp_dir.mkdir(exist_ok=True)

    if not zpath.exists():
        print(f"[dl] {url}")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(shp_dir)

    for p in shp_dir.glob("*.shp"):
        if p.name.startswith("cb_2023_us_county_500k"):
            return p
    return None


def ny_choropleth():
    if not HAS_GPD:
        print("[warn] GeoPandas not available; skipping choropleth. Install with: pip install geopandas pyproj shapely fiona")
        return
    if not ny_file.exists():
        print("[warn] NY county CSV missing; run: python guyanese_us_data_pipeline.py acs --geo county --state-fips 36")
        return

    shp_path = download_ny_county_shapes()
    if shp_path is None:
        print("[warn] Could not locate county shapefile after download.")
        return

    gdf = gpd.read_file(shp_path)
    ny_shapes = gdf[gdf["STATEFP"] == "36"].copy()

    ny = pd.read_csv(ny_file)[["state_fips","county_fips","guyanese_total_ancestry","NAME"]].copy()
    ny["county_fips"] = ny["county_fips"].astype(str).str.zfill(3)

    m = ny_shapes.merge(ny, left_on="COUNTYFP", right_on="county_fips", how="left")

    fig, ax = plt.subplots(figsize=(8.5,8))
    m.plot(column="guyanese_total_ancestry", legend=True, ax=ax)
    ax.set_axis_off()
    ax.set_title("Guyanese Ancestry — New York Counties (ACS 5-year 2023)")
    add_caption("Cartographic boundaries: U.S. Census Bureau (2023). Values: total reporting Guyanese ancestry (single + multiple).")
    save_fig("06_ny_counties_choropleth.png")

# Build the map
try:
    ny_choropleth()
except Exception as e:
    warnings.warn(f"Choropleth skipped due to error: {e}")
