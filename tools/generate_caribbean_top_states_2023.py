#!/usr/bin/env python3
"""Generate top-state weighted ancestry bar charts from ACS 1-year PUMS 2023.

Creates matching charts for:
- Trinidadian Tobagonian (ANC code 314)
- Dutch West Indian (used here as Suriname proxy in 2023 ANC coding) (ANC code 310)
- Jamaican (ANC code 308)

Outputs:
- data/outputs/caribbean_top_states_2023/*.csv
- figures/caribbean_top_states_2023/*.png
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

PUMS_1Y_PERSONS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip"

RAW_ZIP = Path("data/raw/pums_2023_1y_persons.zip")
OUT_DIR = Path("data/outputs/caribbean_top_states_2023")
FIG_DIR = Path("figures/caribbean_top_states_2023")

# Matches the existing top-states visual style used in viz.py.
plt.rcParams.update({
    "figure.dpi": 180,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FIPS_TO_POSTAL = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"
}

TARGETS = [
    {
        "key": "trinidad",
        "label": "Trinidadian Tobagonian",
        "code": 314,
        "title": "Presence of Trinidad Ancestry in the US",
        "caption_label": "Trinidadian Tobagonian",
    },
    {
        "key": "suriname",
        "label": "Suriname (Dutch West Indian proxy)",
        "code": 310,
        "title": "Presence of Suriname Ancestry in the US",
        "caption_label": "Dutch West Indian (proxy for Suriname in 2023 PUMS ANC labels)",
    },
    {
        "key": "jamaica",
        "label": "Jamaican",
        "code": 308,
        "title": "Presence of Jamaican Ancestry in the US",
        "caption_label": "Jamaican",
    },
]


def ensure_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[skip] Found {path}")
        return

    print(f"[download] {PUMS_1Y_PERSONS_URL}")
    with requests.get(PUMS_1Y_PERSONS_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    print(f"[ok] Downloaded {path}")


def iter_person_csv_members(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            lname = name.lower()
            if lname.endswith(".csv") and "psam_pus" in lname:
                with zf.open(name) as f:
                    yield io.BytesIO(f.read())


def compute_weighted_states(zip_path: Path) -> dict[str, pd.DataFrame]:
    sums_by_target: dict[str, dict[str, float]] = {t["key"]: {} for t in TARGETS}

    for fh in iter_person_csv_members(zip_path):
        for chunk in pd.read_csv(fh, dtype=str, chunksize=250_000):
            for needed in ("STATE", "ANC1P", "ANC2P", "PWGTP"):
                if needed not in chunk.columns:
                    raise RuntimeError(f"Missing expected column: {needed}")

            chunk["STATE"] = chunk["STATE"].astype(str).str.zfill(2)
            chunk["ANC1P"] = pd.to_numeric(chunk["ANC1P"], errors="coerce")
            chunk["ANC2P"] = pd.to_numeric(chunk["ANC2P"], errors="coerce")
            chunk["PWGTP"] = pd.to_numeric(chunk["PWGTP"], errors="coerce")

            for target in TARGETS:
                code = target["code"]
                mask = (chunk["ANC1P"] == code) | (chunk["ANC2P"] == code)
                if not mask.any():
                    continue

                grouped = (
                    chunk.loc[mask, ["STATE", "PWGTP"]]
                    .groupby("STATE", as_index=False)["PWGTP"]
                    .sum()
                )
                bucket = sums_by_target[target["key"]]
                for _, row in grouped.iterrows():
                    st = str(row["STATE"]).zfill(2)
                    bucket[st] = bucket.get(st, 0.0) + float(row["PWGTP"])

    out: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        key = target["key"]
        rows = [{"STATE": state, "weighted_count": val} for state, val in sums_by_target[key].items()]
        df = pd.DataFrame(rows)
        if df.empty:
            out[key] = df
            continue
        df = df.sort_values("weighted_count", ascending=False).reset_index(drop=True)
        df["state_name"] = df["STATE"].map(FIPS_TO_POSTAL).fillna(df["STATE"])
        out[key] = df

    return out


def save_tables(weighted_states: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        key = target["key"]
        out_file = OUT_DIR / f"pums_2023_{key}_state_weighted_counts.csv"
        weighted_states[key].to_csv(out_file, index=False)
        print(f"[ok] Wrote {out_file}")


def save_chart(df: pd.DataFrame, target: dict[str, str | int], top_n: int = 12) -> None:
    if df.empty:
        print(f"[warn] No records found for {target['label']} ({target['code']}).")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top_df = df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(top_df["state_name"], top_df["weighted_count"])
    ax.set_title(str(target["title"]))
    ax.set_xlabel("State (postal)")
    ax.set_ylabel("People (weighted)")
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    for rect in ax.patches:
        y = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        ax.annotate(f"{y:,.0f}", (x, y), xytext=(0, 3), textcoords="offset points", va="bottom", ha="center", fontsize=9)

    plt.figtext(
        0.01,
        -0.06,
        f"Source: ACS 1-year PUMS 2023 (ancestry code = {target['caption_label']}). Weights (PWGTP) applied.",
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    out_file = FIG_DIR / f"top_states_{target['key']}_pums2023.png"
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Wrote {out_file}")


def main() -> None:
    ensure_zip(RAW_ZIP)
    weighted_states = compute_weighted_states(RAW_ZIP)
    save_tables(weighted_states)
    for target in TARGETS:
        save_chart(weighted_states[target["key"]], target)


if __name__ == "__main__":
    main()
