# Labor & Economy: Guyanese Context – Data Scripts

This repo contains four scripts to build a reproducible data pack on labor and business context for communities with a Guyanese presence.

## 1) CPS (IPUMS CPS API) — Guyanese Labor Outcomes
**File:** `cps_guyanese_indicators.py`  
**What it does:**  
- Submits an **IPUMS CPS** microdata extract (default: ASEC samples `cpsYYYY_03s`), asks for CSV output, downloads and loads the file.  
- Filters **age 16+** where **birthplace = Guyana** OR **ancestry = Guyanese** (when ancestry is available).  
- Computes weighted **LFP rate**, **unemployment rate**, **avg weekly hours (UHRSWORKT)** among employed, plus **occupation/industry distributions**.  
- Saves: `cps_guyanese_micro_subset.csv`, `cps_guyanese_summary.csv`, `cps_guyanese_occ_distribution.csv`, `cps_guyanese_ind_distribution.csv`, and `README_cps_extract.json`.

**Weights:** Uses **ASECWT** for ASEC (fallback to WTFINL if needed).  
**Employment coding:** IPUMS CPS **EMPSTAT** general codes 1=Employed, 2=Unemployed; **LABFORCE** 2=In labor force (used for denominator).  
**Refs:** IPUMS microdata API v2 supports CSV; CPS ASEC sample IDs like `cps2019_03s`; EMPSTAT/LABFORCE coding per IPUMS docs. :contentReference[oaicite:4]{index=4}

> ⚠️ Joining to OEWS SOC: CPS uses harmonized occupation codes (e.g., `OCC2010`). Join to SOC using a crosswalk as needed.

## 2) OEWS Wages — Metro Median by SOC
**File:** `oews_metro_median_wages.py`  
**What it does:**  
- Reads **OEWS time-series text files** (`oe.series`, `oe.data.1.AllData`, `oe.datatype`, `oe.area`) from the BLS download server.  
- For given **MSA/CBSA codes** (e.g., 35620, 33100, 16980) and **year**, extracts **median wages** by **SOC** (prefers **annual** median; falls back to **hourly** median if annual missing).  
- Saves: `oews_metro_median_wages_{year}.csv` with `soc_code` join key + `wage_unit`.

**Refs:** Official “Text Files for OEWS”; structure documented in `oe.txt`; files hosted at `download.bls.gov/pub/time.series/oe/`. :contentReference[oaicite:5]{index=5}

## 3) County Business Patterns & Nonemployer Statistics
**File:** `cbp_nes_by_naics.py`  
**What it does:**  
- Calls **Census CBP API** (e.g., 2023) and **NES timeseries API** (e.g., 2022) for selected counties (5-digit FIPS).  
- Auto-infers the correct **count variable** (CBP: establishments; NES: number of firms) from the dataset’s variable metadata.  
- Produces a summary by **NAICS** and flags **retail, food services, transportation/warehousing** as **context sectors**.  
- Saves: raw pulls + `cbp_nes_summary_{cbp_year}_{nes_year}.csv`.

**Refs:** Census CBP 2023 dataset page; Nonemployer (timeseries) dataset docs. :contentReference[oaicite:6]{index=6}  
> ⚠️ These are **context indicators** only — **not nationality-specific**.

## 4) IRS SOI County Migration — Net Matrix & Top Flows
**File:** `irs_soi_county_migration.py`  
**What it does:**  
- Scrapes the IRS **Migration Data Downloads** pages for **county inflow/outflow** CSVs for a specified **year label** (e.g., “2010-2011”, “2021-2022”).  
- Normalizes differing schemas across years, then computes **net migration matrices** among your selected counties and **top origins/destinations** lists.  
- Saves: `irs_flows_{label}.csv`, `irs_net_matrix_{label}.csv`, `irs_top_flows_{label}.csv`, `irs_net_summary_{label}.csv`.

**Refs:** IRS SOI migration downloads & county-to-county files pages. :contentReference[oaicite:7]{index=7}

---

### Setup
```bash
# Python 3.9+ recommended
pip install ipumspy pandas requests beautifulsoup4
export IPUMS_API_KEY="YOUR_IPUMS_KEY"
