#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guyanese diaspora insights — one file end-to-end.

Inputs (default filenames in CWD):
  - osm_pois.csv         (from your harvester)
  - gdelt_articles.csv   (from your harvester)
Optional:
  - --use-nominatim      (reverse geocode missing OSM city/state; rate-limited & cached)
  - --metro-pop-file     CSV with columns: metro,population (int)
  - --cities-file        CSV with columns: city,state,metro (to augment the built-in map)

Outputs (in --outdir):
  OSM:
    - osm_enriched.csv
    - osm_state_counts.csv, osm_city_counts.csv
    - osm_culinary_counts.csv
    - osm_dbscan_clusters.csv
    - osm_scatter.png, osm_clusters.png
  GDELT:
    - gdelt_deduped.csv
    - gdelt_sentiment_by_month.csv, gdelt_sentiment_by_domain.csv
    - gdelt_topics_doc.csv, gdelt_topics_monthly.csv, gdelt_topics_terms.csv
    - gdelt_keyword_counts.csv, gdelt_seasonality_monthly.csv
    - gdelt_geotag_mentions.csv
    - gdelt_monthly_counts.png, gdelt_sentiment_timeseries.png, gdelt_top_domains.png
  Presence:
    - presence_index_by_metro.csv  (with/without per capita)

Notes:
  - Uses only official/public endpoints; optional Nominatim reverse geocoder (OSM) with polite rate limiting & cache.
  - Designed to degrade gracefully if optional libs are missing (spaCy, rapidfuzz).
"""

import os, sys, re, json, time, math, argparse, string, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# Sentiment (VADER)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Fuzzy dedupe (optional)
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# spaCy NER (optional)
SPACY_OK = False
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_OK = True
    except Exception:
        SPACY_OK = False
except Exception:
    SPACY_OK = False

# Optional reverse geocoding (Nominatim)
try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False

USER_AGENT = "PharmachuteDiasporaInsights/1.0 (contact: rohan@pharmachute.com)"

# -------------------- Utilities --------------------

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path):
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[!] Missing file: {path}", file=sys.stderr)
        return pd.DataFrame()
    return pd.read_csv(path)

def normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"https?://\S+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_seendate(s):
    s = str(s)
    # Try ISO
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        pass
    # Try GDELT numeric format
    try:
        return pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def minhash_like_ratio(a: str, b: str) -> int:
    """Fallback similarity if rapidfuzz unavailable."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0
    return int(100 * len(sa & sb) / len(sa | sb))

def similarity(a: str, b: str) -> int:
    if RAPIDFUZZ_OK:
        return fuzz.token_set_ratio(a, b)
    return minhash_like_ratio(a, b)

def monthify(dt: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return dt.dt.to_period("M").astype(str)

def safe_lower(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower()

def mode_or_blank(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    return vals.mode().iloc[0] if not vals.empty else ""

# -------------------- Built-in city/metro helpers --------------------

# A compact built-in map of common US places relevant here; you can augment via --cities-file.
CITY_TO_METRO = {
    # NYC area
    ("richmond hill","ny"): "New York-Newark-Jersey City",
    ("jamaica","ny"): "New York-Newark-Jersey City",
    ("queens","ny"): "New York-Newark-Jersey City",
    ("brooklyn","ny"): "New York-Newark-Jersey City",
    ("bronx","ny"): "New York-Newark-Jersey City",
    ("manhattan","ny"): "New York-Newark-Jersey City",
    ("new york","ny"): "New York-Newark-Jersey City",
    # NJ
    ("newark","nj"): "New York-Newark-Jersey City",
    ("jersey city","nj"): "New York-Newark-Jersey City",
    # FL
    ("miramar","fl"): "Miami-Fort Lauderdale-West Palm Beach",
    ("miami","fl"): "Miami-Fort Lauderdale-West Palm Beach",
    ("fort lauderdale","fl"): "Miami-Fort Lauderdale-West Palm Beach",
    ("orlando","fl"): "Orlando-Kissimmee-Sanford",
    ("tampa","fl"): "Tampa-St. Petersburg-Clearwater",
    # DC/MD/VA
    ("washington","dc"): "Washington-Arlington-Alexandria",
    ("silver spring","md"): "Washington-Arlington-Alexandria",
    ("hyattsville","md"): "Washington-Arlington-Alexandria",
    # TX
    ("houston","tx"): "Houston-The Woodlands-Sugar Land",
    ("dallas","tx"): "Dallas-Fort Worth-Arlington",
    # GA
    ("atlanta","ga"): "Atlanta-Sandy Springs-Alpharetta",
    # CA
    ("los angeles","ca"): "Los Angeles-Long Beach-Anaheim",
    ("oakland","ca"): "San Francisco-Oakland-Berkeley",
}

# For regex city matching in titles
CITY_REGEX = re.compile(r"\b(" + "|".join({
    "richmond hill","jamaica","queens","brooklyn","bronx","manhattan","new york",
    "newark","jersey city","miami","fort lauderdale","orlando","tampa",
    "washington","silver spring","hyattsville","houston","dallas","atlanta",
    "los angeles","oakland"
}) + r")\b", flags=re.I)

def load_city_overrides(path: Path):
    """Augment CITY_TO_METRO via CSV with columns: city,state,metro."""
    if not path or not path.exists():
        return
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        c = str(r.get("city","")).strip().lower()
        s = str(r.get("state","")).strip().lower()
        m = str(r.get("metro","")).strip()
        if c and s and m:
            CITY_TO_METRO[(c, s)] = m

# -------------------- Optional Nominatim reverse geocode --------------------

def reverse_geocode(lat, lon, session, cache: dict):
    key = f"{lat:.6f},{lon:.6f}"
    if key in cache:
        return cache[key]
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat, "lon": lon,
        "format": "jsonv2",
        "zoom": 14,
        "addressdetails": 1
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    addr = j.get("address", {})
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("borough") or ""
    state = addr.get("state") or ""
    postcode = addr.get("postcode") or ""
    cache[key] = {"city": city, "state": state, "postcode": postcode}
    return cache[key]

# -------------------- OSM pipeline --------------------

def process_osm(osm_df: pd.DataFrame,
                outdir: Path,
                use_nominatim=False,
                max_geocode=30,
                cities_file: Path=None):

    ensure_outdir(outdir)
    # Clean basics
    for c in ["addr:state","addr:city","name","cuisine"]:
        if c in osm_df.columns:
            osm_df[c] = osm_df[c].fillna("").astype(str).str.strip()
    if "addr:state" in osm_df.columns:
        osm_df["addr:state"] = osm_df["addr:state"].str.upper()

    # Optional reverse geocoding for missing city/state
    if use_nominatim and REQUESTS_OK:
        print("[i] Reverse geocoding missing OSM city/state via Nominatim (polite rate limit)...")
        sess = requests.Session()
        sess.headers.update({"User-Agent": USER_AGENT})
        cache_path = outdir / "nominatim_cache.json"
        cache = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                cache = {}

        need = osm_df[
            (osm_df["addr:city"].eq("") | osm_df["addr:state"].eq(""))
            & osm_df["lat"].notna() & osm_df["lon"].notna()
        ].copy()

        count = 0
        for idx, row in need.iterrows():
            if count >= max_geocode: break
            lat, lon = float(row["lat"]), float(row["lon"])
            try:
                res = reverse_geocode(lat, lon, sess, cache)
                if osm_df.at[idx, "addr:city"] == "":
                    osm_df.at[idx, "addr:city"] = res["city"]
                if osm_df.at[idx, "addr:state"] == "":
                    # Convert state names to USPS if possible (simple map)
                    st = res["state"]
                    USPS = {
                        "New York":"NY","New Jersey":"NJ","Florida":"FL","District of Columbia":"DC",
                        "Maryland":"MD","Virginia":"VA","Texas":"TX","Georgia":"GA","California":"CA",
                        "Oregon":"OR"
                    }
                    osm_df.at[idx, "addr:state"] = USPS.get(st, st)[:2].upper()
                count += 1
                time.sleep(1.0)  # be polite
            except Exception as e:
                print("[!] Nominatim error:", e)
                time.sleep(1.0)
        cache_path.write_text(json.dumps(cache))

    # Summaries
    state_counts = (osm_df.groupby("addr:state", dropna=False).size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False))
    city_counts = (osm_df.groupby(["addr:state","addr:city"], dropna=False).size()
                   .reset_index(name="count")
                   .sort_values(["count","addr:state","addr:city"], ascending=[False,True,True]))

    state_counts.to_csv(outdir / "osm_state_counts.csv", index=False)
    city_counts.to_csv(outdir / "osm_city_counts.csv", index=False)

    # Culinary footprint
    text_cols = []
    if "name" in osm_df.columns: text_cols.append("name")
    if "cuisine" in osm_df.columns: text_cols.append("cuisine")

    culinary_words = [
        "roti","pepperpot","chow mein","bake","saltfish","black pudding","metemgee",
        "cookup","pepper sauce","pholourie","dhal puri","curry","sada","bara","doubles"
    ]
    low = safe_lower(osm_df[text_cols].apply(lambda r: " | ".join(r.values), axis=1)) if text_cols else pd.Series([], dtype=str)
    counts = [{"keyword": k, "count": int(low.str.contains(re.escape(k)).sum())} for k in culinary_words]
    pd.DataFrame(counts).sort_values("count", ascending=False).to_csv(outdir / "osm_culinary_counts.csv", index=False)

    # DBSCAN clustering on coordinates
    clusters_path = outdir / "osm_dbscan_clusters.csv"
    if {"lat","lon"}.issubset(osm_df.columns):
        pts = osm_df.dropna(subset=["lat","lon"]).copy()
        if not pts.empty:
            # haversine DBSCAN (lat/lon in radians). eps ~ 1km
            R = 6371.0088
            coords = np.radians(pts[["lat","lon"]].astype(float).values)
            kms = 1.0
            eps = kms / R
            db = DBSCAN(eps=eps, min_samples=2, metric="haversine").fit(coords)
            pts["cluster_id"] = db.labels_

            # Name clusters by modal city if present; else use centroid
            cluster_rows = []
            for cid, g in pts.groupby("cluster_id"):
                if cid == -1:  # noise
                    continue
                name = mode_or_blank(safe_lower(g["addr:city"]))
                if not name:
                    # centroid
                    clat, clon = g["lat"].astype(float).mean(), g["lon"].astype(float).mean()
                    name = f"cluster@({clat:.3f},{clon:.3f})"
                cluster_rows.append({
                    "cluster_id": int(cid),
                    "n_pois": int(len(g)),
                    "name": name,
                    "center_lat": float(g["lat"].astype(float).mean()),
                    "center_lon": float(g["lon"].astype(float).mean()),
                    "states": ",".join(sorted(set(safe_lower(g["addr:state"]))))
                })
            pd.DataFrame(cluster_rows).sort_values("n_pois", ascending=False).to_csv(clusters_path, index=False)

            # Plot clusters
            plt.figure()
            for cid, g in pts.groupby("cluster_id"):
                plt.scatter(g["lon"], g["lat"], s=18, alpha=0.7, label=f"c{cid}")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.title("OSM POIs — DBSCAN clusters")
            plt.legend(loc="best", fontsize=8, ncols=2)
            savefig(outdir / "osm_clusters.png")

            # Simple scatter
            plt.figure()
            plt.scatter(pts["lon"], pts["lat"], s=18, alpha=0.6)
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.title("OSM POIs — US scatter")
            savefig(outdir / "osm_scatter.png")

    # Augment with metro mapping (from city/state)
    load_city_overrides(cities_file) if cities_file else None
    def city_to_metro(row):
        c = str(row.get("addr:city","")).strip().lower()
        s = str(row.get("addr:state","")).strip().lower()
        return CITY_TO_METRO.get((c, s), "")
    if "addr:city" in osm_df.columns and "addr:state" in osm_df.columns:
        osm_df["metro"] = osm_df.apply(city_to_metro, axis=1)

    # Save enriched
    osm_df.to_csv(outdir / "osm_enriched.csv", index=False)

    # Return metro counts (for presence index)
    metro_pois = (osm_df.groupby("metro").size().reset_index(name="pois"))
    return metro_pois

# -------------------- GDELT pipeline --------------------

def dedupe_gdelt(df: pd.DataFrame, sim_threshold=92):
    # Normalize titles
    df = df.copy()
    df["title_norm"] = safe_lower(df.get("title", pd.Series([""]*len(df)))).map(normalize_title)
    df["domain_norm"] = safe_lower(df.get("domain", pd.Series([""]*len(df))))
    # First pass: exact (domain, title_norm)
    df["group_a"] = df["domain_norm"] + " | " + df["title_norm"]
    # Build fuzzy groups inside each domain for very similar titles
    keep = []
    seen = set()
    for dom, g in df.groupby("domain_norm"):
        idxs = list(g.index)
        titles = g["title_norm"].tolist()
        for i, ii in enumerate(idxs):
            if ii in seen: continue
            group = [ii]
            ti = titles[i]
            for j, jj in enumerate(idxs[i+1:], i+1):
                if jj in seen: continue
                tj = titles[j]
                if similarity(ti, tj) >= sim_threshold:
                    group.append(jj)
            # keep the earliest (or first) seen date if available
            best = g.loc[group]
            if "seendate" in best.columns:
                # choose earliest URL to represent; otherwise first
                rep = best.sort_values("seendate").index[0]
            else:
                rep = group[0]
            keep.append(rep)
            seen.update(group)
    return df.loc[sorted(set(keep))].copy()

def sentiment_vader(df: pd.DataFrame):
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    text = df.get("title", pd.Series([""]*len(df))).fillna("").astype(str)
    scores = text.apply(lambda t: sia.polarity_scores(t)["compound"])
    return scores

def topic_model(df: pd.DataFrame, n_topics=6, random_state=42):
    docs = df.get("title", pd.Series([""]*len(df))).fillna("").astype(str).tolist()
    if len([d for d in docs if d.strip()]) < 5:
        return None, None, None  # not enough text
    vec = TfidfVectorizer(min_df=2, max_df=0.6, ngram_range=(1,2))
    X = vec.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vec.get_feature_names_out())
    # top terms per topic
    top_terms = []
    for k in range(n_topics):
        top = terms[np.argsort(H[k])[::-1][:12]]
        top_terms.append({"topic": k, "terms": ", ".join(top)})
    # dominant topic per doc
    doc_topic = W.argmax(axis=1)
    return pd.Series(doc_topic, name="topic"), pd.DataFrame(top_terms), pd.DataFrame(W)

def geotag_titles(df: pd.DataFrame):
    titles = df.get("title", pd.Series([""]*len(df))).fillna("").astype(str).tolist()
    matches = []
    if SPACY_OK:
        print("[i] spaCy available — using NER for GPE + fallback regex.")
        for t in titles:
            mset = set()
            if t.strip():
                doc = nlp(t)
                for ent in doc.ents:
                    if ent.label_ in ("GPE","LOC"):
                        mset.add(ent.text.lower())
            # add regex city hits
            for m in CITY_REGEX.findall(t):
                mset.add(m.lower())
            matches.append(sorted(mset))
    else:
        print("[i] spaCy not available — using regex city matcher only.")
        for t in titles:
            mset = set(m.lower() for m in CITY_REGEX.findall(t))
            matches.append(sorted(mset))
    # Map city->metro when we can (state unknown in titles; infer by uniqueness)
    metro_hits = []
    for ms in matches:
        metros = set()
        for city in ms:
            # try both NY vs DC ambiguity if city equals 'washington' etc.
            # rough heuristic: pick the first state mapping we have
            for (c,s), metro in CITY_TO_METRO.items():
                if c == city:
                    metros.add(metro)
        metro_hits.append(sorted(metros))
    out = pd.DataFrame({
        "title": df.get("title"),
        "domain": df.get("domain"),
        "seen_dt": df.get("seen_dt"),
        "cities_found": [";".join(m) for m in matches],
        "metros_found": [";".join(m) for m in metro_hits]
    })
    return out

def gdelt_pipeline(gdelt_df: pd.DataFrame, outdir: Path, n_topics=6, random_state=42):
    ensure_outdir(outdir)
    # Parse dates
    gdelt_df = gdelt_df.copy()
    if "seendate" in gdelt_df.columns:
        gdelt_df["seen_dt"] = gdelt_df["seendate"].apply(parse_seendate)
    else:
        gdelt_df["seen_dt"] = pd.NaT

    # Deduplicate (de-syndicate)
    print("[i] De-syndicating GDELT stories…")
    dd = dedupe_gdelt(gdelt_df)
    dd.to_csv(outdir / "gdelt_deduped.csv", index=False)

    # Monthly counts
    monthly = (dd.assign(month=monthify(dd["seen_dt"]))
                 .groupby("month").size().reset_index(name="count")
                 .sort_values("month"))
    monthly.to_csv(outdir / "gdelt_monthly_counts.csv", index=False)

    if not monthly.empty:
        plt.figure()
        plt.plot(pd.to_datetime(monthly["month"]), monthly["count"])
        plt.title("GDELT mentions — Monthly (de-syndicated)")
        plt.xlabel("Month"); plt.ylabel("Articles")
        savefig(outdir / "gdelt_monthly_counts.png")

    # Top domains (de-syndicated)
    if "domain" in dd.columns:
        top_dom = (safe_lower(dd["domain"]).value_counts()
                   .reset_index().rename(columns={"index":"domain","domain":"count"}))
        top_dom.head(50).to_csv(outdir / "gdelt_top_domains.csv", index=False)
        if not top_dom.empty:
            plt.figure()
            plt.bar(top_dom["domain"].head(30), top_dom["count"].head(30))
            plt.xticks(rotation=70, ha="right")
            plt.xlabel("Domain"); plt.ylabel("Count")
            plt.title("GDELT — Top domains (de-syndicated)")
            savefig(outdir / "gdelt_top_domains.png")

    # Sentiment (VADER) on titles
    print("[i] VADER sentiment on titles…")
    dd["sentiment"] = sentiment_vader(dd)
    sent_month = (dd.assign(month=monthify(dd["seen_dt"]))
                    .groupby("month")["sentiment"].mean().reset_index())
    sent_dom = (dd.groupby(safe_lower(dd["domain"]))["sentiment"]
                  .mean().reset_index().rename(columns={"domain":"domain","sentiment":"avg_sentiment"})
                  .sort_values("avg_sentiment"))
    sent_month.to_csv(outdir / "gdelt_sentiment_by_month.csv", index=False)
    sent_dom.to_csv(outdir / "gdelt_sentiment_by_domain.csv", index=False)

    if not sent_month.empty:
        plt.figure()
        plt.plot(pd.to_datetime(sent_month["month"]), sent_month["sentiment"])
        plt.xlabel("Month"); plt.ylabel("Avg compound sentiment")
        plt.title("GDELT — Sentiment over time (titles)")
        savefig(outdir / "gdelt_sentiment_timeseries.png")

    # Topics (TF-IDF + NMF)
    print("[i] Topic modeling (NMF)…")
    topic_series, topic_terms, W = topic_model(dd, n_topics=n_topics, random_state=random_state)
    if topic_series is not None:
        dd["topic"] = topic_series
        dd[["title","domain","seen_dt","topic","sentiment"]].to_csv(outdir / "gdelt_topics_doc.csv", index=False)
        topic_terms.to_csv(outdir / "gdelt_topics_terms.csv", index=False)
        topics_monthly = (dd.dropna(subset=["seen_dt"])
                            .assign(month=monthify(dd["seen_dt"]))
                            .groupby(["month","topic"]).size().reset_index(name="count"))
        topics_monthly.to_csv(outdir / "gdelt_topics_monthly.csv", index=False)

    # Keyword probes + seasonality
    keywords = [
        "richmond hill","little guyana","queens","brooklyn","jamaica","new york",
        "miami","orlando","tampa","newark","jersey city","washington","houston","dallas","atlanta",
        "festival","parade","mashramani","independence","diwali","eid","cricket","soca","chutney","restaurant","bakery","roti","pepperpot"
    ]
    title_lc = safe_lower(dd["title"])
    kw_counts = [{"keyword":k, "count": int(title_lc.str.contains(re.escape(k)).sum())} for k in keywords]
    pd.DataFrame(kw_counts).sort_values("count", ascending=False).to_csv(outdir / "gdelt_keyword_counts.csv", index=False)

    # Simple seasonality around likely festive periods (month-based)
    dd["month"] = monthify(dd["seen_dt"])
    dd["mnum"] = pd.to_datetime(dd["month"], errors="coerce").dt.month
    seasonal = dd.assign(
        is_feb = dd["mnum"].eq(2),
        is_may = dd["mnum"].eq(5),
        is_octnov = dd["mnum"].isin([10,11])
    ).groupby("month")[["is_feb","is_may","is_octnov"]].mean().reset_index()
    seasonal.to_csv(outdir / "gdelt_seasonality_monthly.csv", index=False)

    # Geotag titles -> metros
    geo = geotag_titles(dd)
    geo.to_csv(outdir / "gdelt_geotag_mentions.csv", index=False)

    # Return metro mention counts for presence index
    metro_counts = []
    for _, r in geo.iterrows():
        mets = [m for m in str(r["metros_found"]).split(";") if m]
        for m in mets:
            metro_counts.append(m)
    metro_mentions = (pd.Series(metro_counts).value_counts().reset_index()
                      .rename(columns={"index":"metro",0:"mentions"}))
    return dd, metro_mentions

# -------------------- Presence index --------------------

def presence_index(metro_pois: pd.DataFrame,
                   metro_mentions: pd.DataFrame,
                   outdir: Path,
                   metro_pop_file: Path=None,
                   alpha=0.5):
    # Merge sparse frames
    base = pd.DataFrame({"metro":[],"pois":[]}).append(metro_pois, ignore_index=True) if not metro_pois.empty else pd.DataFrame({"metro":[],"pois":[]})
    base = base.rename(columns={"0":"pois"})
    base2 = pd.DataFrame({"metro":[],"mentions":[]}).append(metro_mentions, ignore_index=True) if not metro_mentions.empty else pd.DataFrame({"metro":[],"mentions":[]})
    base2 = base2.rename(columns={"0":"mentions"})
    df = pd.merge(base, base2, on="metro", how="outer").fillna(0)
    if df.empty:
        (outdir / "presence_index_by_metro.csv").write_text("")
        return df

    # Normalized (0..1)
    scaler = MinMaxScaler()
    df[["pois_norm","mentions_norm"]] = scaler.fit_transform(df[["pois","mentions"]].astype(float))
    df["presence_index"] = 100.0 * (alpha*df["pois_norm"] + (1-alpha)*df["mentions_norm"])

    # Optional per-capita
    if metro_pop_file and Path(metro_pop_file).exists():
        popdf = pd.read_csv(metro_pop_file)
        popdf.columns = [c.strip().lower() for c in popdf.columns]
        if {"metro","population"}.issubset(set(popdf.columns)):
            df = df.merge(popdf[["metro","population"]], on="metro", how="left")
            df["pois_per_million"] = df.apply(lambda r: (1e6*r["pois"]/r["population"]) if r.get("population",0)>0 else np.nan, axis=1)
            df["mentions_per_million"] = df.apply(lambda r: (1e6*r["mentions"]/r["population"]) if r.get("population",0)>0 else np.nan, axis=1)
            # normalized per million index
            sub = df[["pois_per_million","mentions_per_million"]].fillna(0.0).astype(float)
            if (sub.max() > 0).any():
                sub_norm = scaler.fit_transform(sub)
                df["presence_per_million_index"] = 100.0 * (alpha*sub_norm[:,0] + (1-alpha)*sub_norm[:,1])

    df.sort_values("presence_index", ascending=False).to_csv(outdir / "presence_index_by_metro.csv", index=False)
    return df

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Guyanese diaspora insights (venues, media, sentiment, topics, presence).")
    ap.add_argument("--osm", default="osm_pois.csv", help="Path to osm_pois.csv")
    ap.add_argument("--gdelt", default="gdelt_articles.csv", help="Path to gdelt_articles.csv")
    ap.add_argument("--outdir", default="output", help="Output directory")
    ap.add_argument("--use-nominatim", action="store_true", help="Reverse geocode missing city/state for OSM (rate-limited)")
    ap.add_argument("--max-geocode", type=int, default=30, help="Max Nominatim reverse geocodes (default 30)")
    ap.add_argument("--cities-file", default=None, help="Optional CSV (city,state,metro) to augment metro mapping")
    ap.add_argument("--metro-pop-file", default=None, help="Optional CSV (metro,population) for per-capita presence")
    ap.add_argument("--n-topics", type=int, default=6, help="Number of topics for NMF (default 6)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Presence index blend for POIs vs Mentions (0..1, default 0.5)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = ap.parse_args()
    outdir = Path(args.outdir); ensure_outdir(outdir)

    # Load data
    osm_df = read_csv_safe(Path(args.osm))
    gdelt_df = read_csv_safe(Path(args.gdelt))
    if osm_df.empty and gdelt_df.empty:
        print("[!] No input data found. Provide --osm and/or --gdelt", file=sys.stderr)
        sys.exit(1)

    # OSM
    print("[i] Processing OSM…")
    metro_pois = pd.DataFrame(columns=["metro","pois"])
    if not osm_df.empty:
        metro_pois = process_osm(
            osm_df,
            outdir=outdir,
            use_nominatim=args.use_nominatim,
            max_geocode=args.max_geocode,
            cities_file=Path(args.cities_file) if args.cities_file else None
        )

    # GDELT
    print("[i] Processing GDELT…")
    metro_mentions = pd.DataFrame(columns=["metro","mentions"])
    if not gdelt_df.empty:
        dd, metro_mentions = gdelt_pipeline(
            gdelt_df, outdir=outdir, n_topics=args.n_topics, random_state=args.random_state
        )

    # Presence index
    print("[i] Computing presence index by metro…")
    presence = presence_index(
        metro_pois=metro_pois,
        metro_mentions=metro_mentions,
        outdir=outdir,
        metro_pop_file=Path(args.metro_pop_file) if args.metro_pop_file else None,
        alpha=args.alpha
    )

    print("[✓] Done. Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
