
#!/usr/bin/env python3
"""
guyanese_comments_2025_scraper.py  (v1.1)

Fixes & improvements vs v1.0
----------------------------
- **GitHub 422 fix**: Corrected search qualifier. We now use
  'updated:>=START updated:<END' instead of trying to hack an exclusive end
  with string slicing. We *still* time‑filter individual comments precisely.
- **Pushshift 403 resilience**: Added fallback to a public mirror
  (api.pullpush.io). Also tries legacy path format. Gentle backoff on 429/5xx.
- **Better logging**: Shows per‑source counts and the actual GitHub search
  query string for debugging.
- **Polite retries**: `polite_get` now supports basic exponential backoff
  for transient HTTP errors.

Ethics & scope remain the same: public endpoints/APIs only, no scraping of
private/logged‑in/paywalled content, modest rate limits, and a contactable
User‑Agent via CONTACT_EMAIL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


DEFAULT_QUERY = "guyanese"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-01-01"  # exclusive upper bound to capture full 2025
DEFAULT_MAX_PER_SOURCE = 500
DEFAULT_SLEEP = 1.0  # seconds between requests to be polite
MAX_RETRIES = 4


# ---------- Utilities ----------

def iso_to_epoch(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def epoch_to_iso(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def make_user_agent() -> str:
    contact = os.getenv("CONTACT_EMAIL", "").strip()
    ua = "EthicalCommentCollector/1.1 (+https://example.org/ethics)"
    if contact:
        ua += f" (contact: {contact})"
    return ua


def polite_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
               sleep: float = DEFAULT_SLEEP, timeout: float = 20.0) -> requests.Response:
    hdrs = {"User-Agent": make_user_agent()}
    if headers:
        hdrs.update(headers)

    # Simple retry policy with exponential backoff for transient errors.
    delay = sleep
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                # transient
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            time.sleep(sleep)  # politeness after successful request
            return resp
        except requests.RequestException as e:
            last_exc = e
            # If client error other than rate limit, don't spin too much
            if isinstance(e, requests.HTTPError) and e.response is not None:
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    break
            time.sleep(delay)
            delay *= 2
    # If we get here, we failed all attempts
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request failure")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class Comment:
    source: str
    id: str
    author: Optional[str]
    text: str
    url: str
    created_at: str
    extra: Dict[str, object]

    def to_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["extra"] = json.dumps(d["extra"], ensure_ascii=False)
        return d


# ---------- Fetchers ----------

def fetch_hn_comments(query: str, start_ts: int, end_ts: int, limit: int) -> Iterable[Comment]:
    base = "https://hn.algolia.com/api/v1/search_by_date"
    page = 0
    got = 0
    while got < limit:
        params = {
            "query": query,
            "tags": "comment",
            "numericFilters": f"created_at_i>={start_ts},created_at_i<{end_ts}",
            "page": page,
            "hitsPerPage": 100,
        }
        r = polite_get(base, params=params)
        data = r.json()
        hits = data.get("hits", [])
        if not hits:
            break
        for h in hits:
            if got >= limit:
                break
            text = (h.get("comment_text") or "") or ""
            if query.lower() not in text.lower():
                continue
            author = h.get("author")
            created_i = h.get("created_at_i")
            created_at = epoch_to_iso(created_i) if created_i else ""
            hn_link = f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            yield Comment(
                source="hackernews",
                id=str(h.get("objectID")),
                author=author,
                text=text,
                url=hn_link,
                created_at=created_at,
                extra={
                    "story_title": h.get("story_title"),
                    "story_url": h.get("story_url"),
                    "parent_id": h.get("parent_id"),
                },
            )
            got += 1
        nb_pages = data.get("nbPages", page + 1)
        page += 1
        if page >= nb_pages:
            break


def _pushshift_endpoints() -> List[str]:
    # Try primary, then public mirror, then legacy path
    return [
        "https://api.pushshift.io/reddit/comment/search",
        "https://api.pullpush.io/reddit/comment/search",
        "https://api.pushshift.io/reddit/search/comment/",
    ]


def fetch_reddit_pushshift(query: str, start_ts: int, end_ts: int, limit: int) -> Iterable[Comment]:
    size = 100
    got = 0
    after = start_ts

    endpoints = _pushshift_endpoints()
    ep_index = 0

    while got < limit and ep_index < len(endpoints):
        base = endpoints[ep_index]
        try:
            # Use ascending order and advance 'after' cursor.
            while got < limit:
                params = {
                    "q": query,
                    "after": after,
                    "before": end_ts,
                    "size": size,
                    "sort": "asc",
                }
                r = polite_get(base, params=params)
                data = r.json()
                items = data.get("data") or data.get("results") or []
                if not items:
                    break
                advanced = False
                for it in items:
                    if got >= limit:
                        break
                    body = it.get("body", "") or ""
                    if query.lower() not in body.lower():
                        continue
                    created = it.get("created_utc")
                    created_at = epoch_to_iso(created) if created else ""
                    cid = it.get("id")
                    author = it.get("author")
                    permalink = it.get("permalink")
                    if not permalink:
                        link_id = (it.get("link_id") or "").replace("t3_", "")
                        permalink = f"/r/{it.get('subreddit')}/comments/{link_id}/_/{cid}/"
                    url = f"https://www.reddit.com{permalink}" if permalink else ""
                    yield Comment(
                        source="reddit",
                        id=str(cid),
                        author=author,
                        text=body,
                        url=url,
                        created_at=created_at,
                        extra={
                            "subreddit": it.get("subreddit"),
                            "score": it.get("score"),
                        },
                    )
                    got += 1
                    if created:
                        after = max(after, created)
                        advanced = True
                if not advanced:
                    # safety to avoid infinite loop
                    after += 1
        except requests.HTTPError as e:
            # If forbidden/rate-limited, try next endpoint
            if e.response is not None and e.response.status_code in (403, 429, 500, 502, 503, 504):
                ep_index += 1
                continue
            else:
                raise
        except Exception:
            ep_index += 1
            continue
        # finished this endpoint (no more items); stop
        break


def fetch_github_comments(query: str, start_iso: str, end_iso: str, limit: int) -> Iterable[Comment]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": make_user_agent(),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    search_url = "https://api.github.com/search/issues"
    page = 1
    per_page = 50
    got = 0

    # Use 'updated' to surface issues/PRs that had 2025 activity,
    # then filter individual comments by exact timestamp window.
    q = f"{query} in:comments updated:>={start_iso} updated:<{end_iso}"

    while got < limit:
        params = {"q": q, "per_page": per_page, "page": page}
        resp = polite_get(search_url, params=params, headers=headers, sleep=DEFAULT_SLEEP)
        data = resp.json()
        items = data.get("items", []) or []
        if not items:
            break

        for issue in items:
            if got >= limit:
                break
            comments_url = issue.get("comments_url")
            if not comments_url:
                continue

            # Paginate through comments
            cpage = 1
            while got < limit:
                cparams = {"per_page": 100, "page": cpage}
                cr = polite_get(comments_url, params=cparams, headers=headers, sleep=DEFAULT_SLEEP)
                clist = cr.json() or []
                if not clist:
                    break
                for c in clist:
                    if got >= limit:
                        break
                    body = (c.get("body") or "")
                    if query.lower() not in body.lower():
                        continue
                    created_at_iso = c.get("created_at")
                    if not created_at_iso:
                        continue
                    created_dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
                    # Strict window: [start_iso, end_iso)
                    start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
                    end_dt = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
                    if not (start_dt <= created_dt < end_dt):
                        continue
                    url = c.get("html_url") or issue.get("html_url") or ""
                    yield Comment(
                        source="github",
                        id=str(c.get("id")),
                        author=(c.get("user", {}) or {}).get("login"),
                        text=body,
                        url=url,
                        created_at=created_dt.astimezone(timezone.utc).isoformat(),
                        extra={
                            "repo": issue.get("repository_url", "").split("repos/")[-1],
                            "issue_number": issue.get("number"),
                            "issue_title": issue.get("title"),
                        },
                    )
                    got += 1
                cpage += 1
        page += 1
        if "next" not in (resp.headers.get("Link") or ""):
            # rely on items empty to terminate
            pass


# ---------- Orchestration ----------

def collect_comments(
    query: str,
    start_iso_date: str,
    end_iso_date: str,
    max_per_source: int,
) -> List[Comment]:
    start_ts = iso_to_epoch(start_iso_date)
    end_ts = iso_to_epoch(end_iso_date)
    results: List[Comment] = []
    seen: Set[str] = set()

    hn_count = 0
    reddit_count = 0
    gh_count = 0

    # HN
    try:
        for c in fetch_hn_comments(query, start_ts, end_ts, max_per_source):
            key = sha1(f"{c.source}|{c.id}|{c.text}")
            if key not in seen:
                seen.add(key)
                results.append(c)
                hn_count += 1
    except Exception as e:
        print(f"[WARN] Hacker News fetch failed: {e}", file=sys.stderr)

    # Reddit via Pushshift (with fallback)
    try:
        for c in fetch_reddit_pushshift(query, start_ts, end_ts, max_per_source):
            key = sha1(f"{c.source}|{c.id}|{c.text}")
            if key not in seen:
                seen.add(key)
                results.append(c)
                reddit_count += 1
    except Exception as e:
        print(f"[WARN] Reddit/Pushshift fetch failed: {e}", file=sys.stderr)

    # GitHub
    try:
        for c in fetch_github_comments(query, start_iso_date, end_iso_date, max_per_source):
            key = sha1(f"{c.source}|{c.id}|{c.text}")
            if key not in seen:
                seen.add(key)
                results.append(c)
                gh_count += 1
    except Exception as e:
        print(f"[WARN] GitHub fetch failed: {e}", file=sys.stderr)

    # Stable sort by created_at
    def sort_key(c: Comment) -> Tuple:
        try:
            return (datetime.fromisoformat(c.created_at.replace("Z", "+00:00")), c.source, c.id)
        except Exception:
            return (datetime.min.replace(tzinfo=timezone.utc), c.source, c.id)

    results.sort(key=sort_key)

    print(f"[INFO] Per-source counts: HN={hn_count}, Reddit={reddit_count}, GitHub={gh_count}")
    return results


def write_jsonl(path: str, comments: List[Comment]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def write_csv(path: str, comments: List[Comment]) -> None:
    headers = ["source", "id", "author", "text", "url", "created_at", "extra"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for c in comments:
            writer.writerow(c.to_row())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect public comments mentioning a keyword in 2025 using ethical sources/APIs.")
    p.add_argument("--query", default=DEFAULT_QUERY, help="Substring to find in comments (case-insensitive). Default: %(default)s")
    p.add_argument("--start", default=DEFAULT_START, help="Inclusive start date (YYYY-MM-DD). Default: %(default)s")
    p.add_argument("--end", default=DEFAULT_END, help="Exclusive end date (YYYY-MM-DD). Default: %(default)s")
    p.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE, help="Max comments to fetch per source. Default: %(default)s")
    p.add_argument("--out-jsonl", default="guyanese_comments_2025.jsonl", help="Path to write JSONL output. Default: %(default)s")
    p.add_argument("--out-csv", default="guyanese_comments_2025.csv", help="Path to write CSV output. Default: %(default)s")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Seconds to sleep between requests. Default: %(default)s")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global DEFAULT_SLEEP
    DEFAULT_SLEEP = max(0.0, float(args.sleep))

    # Basic validation
    try:
        iso_to_epoch(args.start)
        iso_to_epoch(args.end)
    except Exception as e:
        print(f"[ERROR] Invalid date(s): {e}", file=sys.stderr)
        return 2

    print("[INFO] Starting collection...")
    print(f"       query={args.query!r} window=[{args.start} .. {args.end}) max_per_source={args.max_per_source}")
    print(f"       outputs: jsonl={args.out_jsonl}, csv={args.out_csv}")
    print(f"       user-agent: {make_user_agent()}")
    print(f"       github-q: {'%s in:comments updated:>=' % args.query + args.start + ' updated:<' + args.end}")

    comments = collect_comments(
        query=args.query,
        start_iso_date=args.start,
        end_iso_date=args.end,
        max_per_source=args.max_per_source,
    )
    print(f"[INFO] Collected {len(comments)} comments. Writing outputs...")

    write_jsonl(args.out_jsonl, comments)
    write_csv(args.out_csv, comments)

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
