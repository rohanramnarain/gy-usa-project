# GY‑USA‑Project

Data analysis and visualization of the Guyanese (GY) diaspora in the United States.  
The repository contains pipelines to pull, clean, aggregate, and visualize data from U.S. Census/ACS & PUMS, IPEDS, SEVIS‑like exports, and locality shapefiles to produce figures and interactive choropleths.

> **Highlights**
> - Reproducible Python pipelines for national and education‑focused views
> - Ready‑made figures (PNG) and interactive maps (HTML)
> - Clear folder structure separating raw/processed data, scripts, and outputs

---

## Table of Contents
- [Quickstart](#quickstart)
- [Repository Layout](#repository-layout)
- [Data Inputs & Outputs](#data-inputs--outputs)
- [Common Workflows](#common-workflows)
  - [Run the full, national pipeline](#run-the-full-national-pipeline)
  - [Education pipeline](#education-pipeline)
  - [Generate figures](#generate-figures)
  - [Build locality choropleths](#build-locality-choropleths)
- [Project Notes](#project-notes)
- [Contributing](#contributing)
- [License](#license)
- [Maintainer](#maintainer)

---

## Quickstart

### 1) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
Place source files under `data/raw/` and education‑specific CSVs under `education/` as appropriate.
The pipelines will write processed files to the `data/outputs/` and `education/outputs/` folders.

> Tip: Large Census shape files (e.g., `cb_2023_us_county_500k.zip`) and PUMS exports (`pums_2023_1y_persons.zip`) go under `data/raw/`.

---

## Repository Layout

```
GY-USA-PROJECT/
├─ .venv/                         # Local virtual environment (not committed)
├─ .gitignore
├─ requirements.txt               # Python dependencies
├─ viz.py                         # Convenience script for generating high‑level figures
├─ guyanese_us_data_pipeline.py   # National pipeline entry point
├─ guyanese_pipelines_auto.py     # Orchestrator/automation wrapper
├─ generate_maps.py               # Map generation driver
├─ build_zip_from_locals.py       # Helper to assemble ZIP‑level locality artifacts
│
├─ data/
│  ├─ raw/
│  │  ├─ cb_2023_us_county_500k/              # Unzipped Census geometries
│  │  ├─ cb_2023_us_county_500k.zip
│  │  └─ pums_2023_1y_persons.zip
│  └─ outputs/
│     ├─ acs5_state_ancestry_2023.csv
│     ├─ acs5_county_ancestry_2023_36.csv     # Example: NY (FIPS 36)
│     ├─ acs5_state_birthplace_guyana_2023.csv
│     ├─ pums_2023_guyanese_persons.csv
│     ├─ pums_2023_nativity_weighted_counts.csv
│     ├─ pums_2023_race_weighted_counts.csv
│     └─ pums_2023_state_weighted_counts.csv
│
├─ figures/                       # High‑level visuals exported by viz.py
│  ├─ 01_top_states_bar.png
│  ├─ 02_nativity_pie.png
│  ├─ 03_visibility_scatter.png
│  ├─ 03a_visibility_scatter_symlog.png
│  ├─ 03b_visibility_scatter_inset.png
│  ├─ 03c_visibility_ratio.png
│  ├─ 04_race_codes_bar.png
│  ├─ 05_ny_counties_bar.png
│  └─ 06_ny_counties_choropleth.png
│
├─ locality/
│  ├─ data/
│  │  ├─ tract_12/                # Example tract‑level inputs
│  │  ├─ tract_36/
│  │  ├─ zcta/
│  │  ├─ nyc_2020_tract_to_nta.csv
│  │  └─ nyc_ntas_2020.geojson
│  ├─ maps/                       # Map artifacts written here
│  │  ├─ nta_choropleth.html/.png
│  │  └─ zip_choropleth.html/.png
│  └─ outputs/                    # Any locality‑level tabular outputs
│
└─ education/
   ├─ edu_pipeline.py
   ├─ filter.py
   ├─ ipeds_guyana_and_completions.py
   ├─ sevis_bulk_to_csv.py
   ├─ sevis_page_to_csv.py
   ├─ october_2025_stem.csv
   ├─ october_2025_stem_guyana_by_state.csv
   ├─ opendoors_like_guyana.csv
   ├─ outputs/
   │  ├─ guayana_by_institution.csv
   │  ├─ guayana_by_state.csv
   │  ├─ pums_guyanese_educ_attainment_by_nativity.csv
   │  └─ pums_guyanese_educ_attainment_national.csv
   └─ figures/
      ├─ edu_attainment_national.png
      └─ edu_attainment_by_nativity.png
```

> The exact files may change as data updates; the structure above reflects the working layout seen in the project.

---

## Data Inputs & Outputs

**Key inputs (under `data/raw/` & `education/`):**
- **Census/ACS/PUMS**: population, ancestry, nativity, and race distributions.
- **Geometries**: `cb_2023_us_county_500k` (county boundaries), NYC NTA (`nyc_ntas_2020.geojson`), tract‑to‑NTA crosswalk.
- **Education**: IPEDS extracts and curated CSVs (e.g., `opendoors_like_guyana.csv`, `october_2025_stem*.csv`).
- **SEVIS‑like**: page/bulk exports converted to CSV via helper scripts.

**Primary outputs:**
- **Processed tables** in `data/outputs/` and `education/outputs/`.
- **Figures (PNG)** in `/figures` and `education/figures/`.
- **Interactive maps (HTML)** in `locality/maps/` including:
  - `nta_choropleth.html` (NYC NTA level)
  - `zip_choropleth.html` (ZIP level)

---

## Common Workflows

### Run the full, national pipeline
Processes ACS/PUMS to produce national/state/county‑level summaries and figures.

```bash
python guyanese_us_data_pipeline.py
# or (if you prefer the orchestrator)
python guyanese_pipelines_auto.py
```

Artifacts will be written to `data/outputs/` and `/figures`.

### Education pipeline
Aggregates education datasets and produces education‑specific CSVs and charts.

```bash
python education/edu_pipeline.py
# helpers
python education/ipeds_guyana_and_completions.py
python education/filter.py
# SEVIS‑like converters
python education/sevis_bulk_to_csv.py
python education/sevis_page_to_csv.py
```

Outputs land in `education/outputs/` and `education/figures/`.

### Generate figures
Top‑level plots (e.g., Top States bar chart, nativity pie, visibility scatter, NY counties bar/choropleth).

```bash
python viz.py
```

### Build locality choropleths
ZIP and NTA‑level interactive maps for NYC (and extendable to other regions).

```bash
# Prepare locality inputs or bundle ZIP‑level artifacts
python build_zip_from_locals.py

# Then render maps (writes to locality/maps/*.html and *.png)
python generate_maps.py
```

> Expect outputs: `locality/maps/nta_choropleth.html` and `locality/maps/zip_choropleth.html` plus matching PNG snapshots.

---

## Project Notes

- **Reproducibility**: keep raw data immutable in `data/raw/`; all transformations write to `outputs/` folders.
- **Large files**: consider Git LFS or excluding heavyweight zips via `.gitignore`.
- **Geospatial**: ensure `geopandas`, `pyproj`, and `shapely` versions in `requirements.txt` align for your platform.
- **Extensibility**: copy a pipeline and change filters to add new geographies (e.g., other states/NTAs/ZCTAs).

---

## Contributing

1. Create a feature branch:  
   `git checkout -b feature/short-name`
2. Commit small, well‑scoped changes with clear messages.
3. Open a Pull Request describing scope, inputs, and outputs touched.

---

## License

Specify a license (e.g., MIT) or “All rights reserved.” If you add a `LICENSE` file, reference it here.

---

## Maintainer

**Rohan Ramnarain** <rohan@pharmachute.com>

---

_This README was generated from the current repo layout and is intended to be kept in sync as scripts and data evolve._
