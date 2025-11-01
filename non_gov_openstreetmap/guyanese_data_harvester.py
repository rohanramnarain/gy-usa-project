#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guyanese diaspora (US) data harvester — one file.
Sources:
  - OpenStreetMap via Overpass API (venues/POIs)
  - Wikidata SPARQL (orgs, restaurants, items)
  - GDELT DOC 2.0 API (US media mentions)
Optional (set environment variables to enable):
  - Eventbrite API (EVENTBRITE_TOKEN)
  - YouTube Data API v3 (YOUTUBE_API_KEY)

Notes:
  - Uses only official/public endpoints. No scraping of Google Maps or sites that prohibit it.
  - Respects rate limits; please be considerate (sleep, retries).
  - Outputs CSVs in --outdir (default: ./output).

CLI:
  python guyanese_data_harvester.py --query "Guyanese" --outdir output

Env (optional):
  export EVENTBRITE_TOKEN="YOUR_EVENTBRITE_OAUTH_TOKEN"
  export YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

USER_AGENT = "PharmachuteDiasporaHarvester/1.0 (+contact: rohan@pharmachute.com)"

# ---------- Helpers ----------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def session_with_retries() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def write_csv(path: str, rows: List[Dict], field_order: Optional[List[str]] = None) -> None:
    if not rows:
        # create an empty file with timestamp + message
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["_note"])
            w.writerow(["No records found at " + now_iso()])
        return
    fields = field_order or sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def clean(s: Optional[str]) -> str:
    return (s or "").strip()

# ---------- Overpass (OSM) ----------

def fetch_osm_pois(query_term: str, area: str = "US", timeout_sec: int = 90) -> List[Dict]:
    """
    Find Guyanese-related POIs in the United States.
    Strategy:
      - area by ISO3166-1=US (admin_level=2)
      - any n/w/r in that area where:
           * cuisine =~ "guyanese" (case-insensitive) OR
           * name   =~ "guyana|guyanese" (case-insensitive)
      - return tags + center lat/lon (for ways/relations)
    """
    overpass = "https://overpass-api.de/api/interpreter"
    q = f"""
[out:json][timeout:{timeout_sec}];
area["ISO3166-1"="{area}"][admin_level=2]->.us;
(
  nwr(area.us)["cuisine"~"guyanese",i];
  nwr(area.us)["name"~"guyana|guyanese",i];
);
out center tags;
"""
    s = session_with_retries()
    resp = s.post(overpass, data=q.encode("utf-8"))
    resp.raise_for_status()
    data = resp.json()
    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            # ways/relations use "center"
            center = el.get("center") or {}
            lat = center.get("lat")
            lon = center.get("lon")
        out.append({
            "source": "OSM",
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "name": tags.get("name"),
            "category_guess": tags.get("amenity") or tags.get("shop") or "",
            "cuisine": tags.get("cuisine", ""),
            "addr:housenumber": tags.get("addr:housenumber", ""),
            "addr:street": tags.get("addr:street", ""),
            "addr:city": tags.get("addr:city", ""),
            "addr:state": tags.get("addr:state", ""),
            "addr:postcode": tags.get("addr:postcode", ""),
            "phone": tags.get("phone") or tags.get("contact:phone", ""),
            "website": tags.get("website") or tags.get("contact:website", ""),
            "lat": lat,
            "lon": lon,
            "matched": "cuisine" if "guyanese" in (tags.get("cuisine","").lower()) else "name" if ("guyana" in (tags.get("name","").lower()) or "guyanese" in (tags.get("name","").lower())) else "",
            "fetched_at": now_iso(),
        })
    return out

# ---------- Wikidata SPARQL ----------

def fetch_wikidata_items(query_term: str) -> List[Dict]:
    """
    Pulls US-located items with Guyanese-related labels or cuisine field.
    Strategy:
      - Restaurants with cuisine (P2012) whose cuisine label contains 'guyanese'
      - OR any item with English label containing 'Guyanese' and country=United States (P17=Q30) or located in admin territory (P131*) within US
      - Returns QID, label, description, coordinates, item type (restaurant/organization/etc.)
    """
    endpoint = "https://query.wikidata.org/sparql"
    s = session_with_retries()
    s.headers.update({"Accept": "application/sparql-results+json"})

    # SPARQL that avoids hardcoding the Q-id for "Guyanese cuisine" by filtering cuisine label text
    sparql = """
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?typeLabel ?coord ?cityLabel ?stateLabel WHERE {
  VALUES ?us { wd:Q30 }
  ?item wdt:P31 ?type .
  OPTIONAL { ?item wdt:P625 ?coord . }
  OPTIONAL { ?item wdt:P131 ?city . OPTIONAL { ?city wdt:P131 ?state . } }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

  # US filter: either country=US or located-in-admin-entity chain ends in US
  FILTER(
    EXISTS { ?item wdt:P17 ?c . FILTER(?c = wd:Q30) } ||
    EXISTS { ?item (wdt:P131/wdt:P131*) ?top . FILTER(?top = wd:Q30) }
  )

  # Guyanese match: either cuisine label matches, or English label matches
  FILTER(
    EXISTS {
      ?item wdt:P2012 ?cuisine .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?cuisine rdfs:label ?cLabel. }
      FILTER(CONTAINS(LCASE(?cLabel), "guyanese"))
    }
    ||
    (BOUND(?itemLabel) && CONTAINS(LCASE(?itemLabel), "guyanese"))
  )
}
LIMIT 1000
"""
    resp = s.get(endpoint, params={"query": sparql})
    resp.raise_for_status()
    j = resp.json()
    rows = []
    for b in j.get("results", {}).get("bindings", []):
        def val(key):
            return b.get(key, {}).get("value")
        qid = val("item")
        if qid and qid.startswith("http"):
            qid = qid.rsplit("/", 1)[-1]
        rows.append({
            "source": "Wikidata",
            "qid": qid,
            "label": val("itemLabel"),
            "description": val("itemDescription"),
            "type": val("typeLabel"),
            "city": val("cityLabel"),
            "state": val("stateLabel"),
            "coord": val("coord"),
            "fetched_at": now_iso(),
        })
    return rows

# ---------- GDELT DOC 2.0 (news/media mentions) ----------

def fetch_gdelt_us_mentions(query_term: str, maxrecords: int = 250) -> List[Dict]:
    """
    GDELT DOC 2.0 full-text search across US sources.
    Note: 'sourceCountry:United States' or 'sourcecountry:US' both used historically; we use sourcecountry:US.
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    # Sort newest first; pull JSON for simplicity
    params = {
        "query": f"{query_term} sourcecountry:US",
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": str(maxrecords),
        "sort": "DateDesc",
    }
    s = session_with_retries()
    r = s.get(base, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    arts = j.get("articles", []) or []
    out = []
    for a in arts:
        out.append({
            "source": "GDELT",
            "url": a.get("url"),
            "title": a.get("title"),
            "seendate": a.get("seendate"),
            "domain": a.get("domain"),
            "language": a.get("language"),
            "sourcecountry": a.get("sourcecountry"),
            "socialimage": a.get("socialimage"),
            "fetched_at": now_iso(),
        })
    return out

# ---------- Eventbrite (optional) ----------

def fetch_eventbrite_events(query_term: str, country: str = "United States", limit_pages: int = 3) -> List[Dict]:
    token = os.environ.get("EVENTBRITE_TOKEN")
    if not token:
        return []
    s = session_with_retries()
    s.headers.update({"Authorization": f"Bearer {token}"})
    url = "https://www.eventbriteapi.com/v3/events/search/"
    params = {
        "q": query_term,
        "location.address": country,
        "expand": "venue,organizer",
        "sort_by": "date",
    }
    all_rows = []
    page = 1
    while page <= limit_pages:
        params["page"] = page
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        j = r.json()
        events = j.get("events", []) or []
        for e in events:
            venue = e.get("venue") or {}
            all_rows.append({
                "source": "Eventbrite",
                "event_id": e.get("id"),
                "name": e.get("name", {}).get("text"),
                "start": e.get("start", {}).get("utc"),
                "end": e.get("end", {}).get("utc"),
                "status": e.get("status"),
                "online_event": e.get("online_event"),
                "url": e.get("url"),
                "venue_name": venue.get("name"),
                "venue_address": (venue.get("address") or {}).get("localized_address_display"),
                "lat": venue.get("latitude"),
                "lon": venue.get("longitude"),
                "capacity": e.get("capacity"),
                "is_free": e.get("is_free"),
                "fetched_at": now_iso(),
            })
        pagination = j.get("pagination") or {}
        if not pagination.get("has_more_items"):
            break
        page += 1
        time.sleep(0.3)
    return all_rows

# ---------- YouTube Data API (optional) ----------

def fetch_youtube_search(query_term: str, region_code: str = "US", max_pages: int = 3) -> List[Dict]:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return []
    base = "https://www.googleapis.com/youtube/v3/search"
    s = session_with_retries()
    params = {
        "key": key,
        "q": query_term,
        "regionCode": region_code,
        "maxResults": "50",
        "type": "video",
        "part": "snippet",
        "order": "date",
        "safeSearch": "none",
    }
    rows = []
    page_count = 0
    next_page_token = None
    while page_count < max_pages:
        if next_page_token:
            params["pageToken"] = next_page_token
        r = s.get(base, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        j = r.json()
        for item in j.get("items", []):
            id_ = (item.get("id") or {}).get("videoId")
            sn = item.get("snippet") or {}
            rows.append({
                "source": "YouTube",
                "video_id": id_,
                "published_at": sn.get("publishedAt"),
                "channel_id": sn.get("channelId"),
                "channel_title": sn.get("channelTitle"),
                "title": sn.get("title"),
                "description": sn.get("description"),
                "url": f"https://www.youtube.com/watch?v={id_}" if id_ else "",
                "fetched_at": now_iso(),
            })
        next_page_token = j.get("nextPageToken")
        if not next_page_token:
            break
        page_count += 1
        time.sleep(0.25)
    return rows

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Harvest public diaspora data for 'Guyanese' in the US.")
    ap.add_argument("--query", default="Guyanese", help="Keyword to search (default: Guyanese)")
    ap.add_argument("--outdir", default="output", help="Directory to write CSV files")
    ap.add_argument("--no-osm", action="store_true", help="Skip OpenStreetMap Overpass")
    ap.add_argument("--no-wikidata", action="store_true", help="Skip Wikidata SPARQL")
    ap.add_argument("--no-gdelt", action="store_true", help="Skip GDELT")
    ap.add_argument("--no-eventbrite", action="store_true", help="Skip Eventbrite (even if token present)")
    ap.add_argument("--no-youtube", action="store_true", help="Skip YouTube (even if API key present)")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    print(f"[i] Output dir: {args.outdir}")

    total_written = []

    # OSM
    if not args.no_osm:
        try:
            print("[i] Fetching OSM POIs via Overpass …")
            osm_rows = fetch_osm_pois(args.query)
            path = os.path.join(args.outdir, "osm_pois.csv")
            write_csv(path, osm_rows, field_order=[
                "source","osm_type","osm_id","name","category_guess","cuisine","addr:housenumber",
                "addr:street","addr:city","addr:state","addr:postcode","phone","website","lat","lon","matched","fetched_at"
            ])
            print(f"[✓] OSM: {len(osm_rows)} rows → {path}")
            total_written.append(("OSM", len(osm_rows), path))
        except Exception as e:
            print(f"[!] OSM error: {e}")

    # Wikidata
    if not args.no_wikidata:
        try:
            print("[i] Fetching Wikidata items …")
            wd_rows = fetch_wikidata_items(args.query)
            path = os.path.join(args.outdir, "wikidata_items.csv")
            write_csv(path, wd_rows, field_order=[
                "source","qid","label","description","type","city","state","coord","fetched_at"
            ])
            print(f"[✓] Wikidata: {len(wd_rows)} rows → {path}")
            total_written.append(("Wikidata", len(wd_rows), path))
        except Exception as e:
            print(f"[!] Wikidata error: {e}")

    # GDELT
    if not args.no_gdelt:
        try:
            print("[i] Fetching GDELT US media mentions …")
            gd_rows = fetch_gdelt_us_mentions(args.query)
            path = os.path.join(args.outdir, "gdelt_articles.csv")
            write_csv(path, gd_rows, field_order=[
                "source","seendate","title","url","domain","language","sourcecountry","socialimage","fetched_at"
            ])
            print(f"[✓] GDELT: {len(gd_rows)} rows → {path}")
            total_written.append(("GDELT", len(gd_rows), path))
        except Exception as e:
            print(f"[!] GDELT error: {e}")

    # Eventbrite (optional)
    if not args.no_eventbrite:
        try:
            if os.environ.get("EVENTBRITE_TOKEN"):
                print("[i] Fetching Eventbrite events (token detected) …")
                eb_rows = fetch_eventbrite_events(args.query)
                path = os.path.join(args.outdir, "eventbrite_events.csv")
                write_csv(path, eb_rows, field_order=[
                    "source","event_id","name","start","end","status","online_event","is_free",
                    "url","venue_name","venue_address","lat","lon","capacity","fetched_at"
                ])
                print(f"[✓] Eventbrite: {len(eb_rows)} rows → {path}")
                total_written.append(("Eventbrite", len(eb_rows), path))
            else:
                print("[i] EVENTBRITE_TOKEN not set — skipping Eventbrite.")
        except Exception as e:
            print(f"[!] Eventbrite error: {e}")

    # YouTube (optional)
    if not args.no_youtube:
        try:
            if os.environ.get("YOUTUBE_API_KEY"):
                print("[i] Fetching YouTube metadata (key detected) …")
                yt_rows = fetch_youtube_search(args.query)
                path = os.path.join(args.outdir, "youtube_videos.csv")
                write_csv(path, yt_rows, field_order=[
                    "source","video_id","published_at","channel_id","channel_title","title","description","url","fetched_at"
                ])
                print(f"[✓] YouTube: {len(yt_rows)} rows → {path}")
                total_written.append(("YouTube", len(yt_rows), path))
            else:
                print("[i] YOUTUBE_API_KEY not set — skipping YouTube.")
        except Exception as e:
            print(f"[!] YouTube error: {e}")

    # Summary
    summary_path = os.path.join(args.outdir, "SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Harvest summary @ {now_iso()}\n")
        for name, n, path in total_written:
            f.write(f"- {name}: {n} → {path}\n")
    print(f"[✓] Wrote summary → {summary_path}")
    print("[done]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Aborted by user.")
        sys.exit(1)
