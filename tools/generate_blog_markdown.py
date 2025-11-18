import os
import shutil
from pathlib import Path
from fnmatch import fnmatch
import json
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ASSETS = SITE / "assets"
SOURCES_FILE = ROOT / "visual_sources.json"

CATEGORY_SPECS = [
    {
        "name": "Overview",
        "slug": "overview",
        "globs": [
            "figures/*.png",
            "locality/maps/*choropleth*.png",
        ],
        "limit": 8,
        "link_html": [],
    },
    {
        "name": "General Figures",
        "slug": "figures",
        "globs": [
            "figures/*.png",
        ],
        "limit": None,
        "link_html": [],
    },
    {
        "name": "Education",
        "slug": "education",
        "globs": [
            "education/figures/*.png",
            "education/outputs/makechartsjs/out/*.png",
        ],
        "limit": None,
        "link_html": [],
    },
    {
        "name": "Economy & Healthcare",
        "slug": "economy-healthcare",
        "globs": [
            "economy_healthcare/outputs/*.png",
            "economy_healthcare/outputs/graphsinjs/out/*.png",
        ],
        "limit": None,
        "link_html": [],
    },
    {
        "name": "Locality & Maps",
        "slug": "locality",
        "globs": [
            "locality/maps/*.png",
        ],
        "limit": None,
        "link_html": [
            "locality/maps/*.html",
        ],
    },
    {
        "name": "Online Presence",
        "slug": "online-presence",
        "globs": [
            "online_presence/analysis_out/*.png",
        ],
        "limit": None,
        "link_html": [],
    },
    {
        "name": "Open Data (OSM/GDELT)",
        "slug": "open-data",
        "globs": [
            "non_gov_openstreetmap/output/*.png",
        ],
        "limit": None,
        "link_html": [],
    },
]


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def humanize(filename: str) -> str:
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ")
    # compact double spaces
    name = " ".join(name.split())
    return name.title()


def find_files(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(sorted(ROOT.glob(pat)))
    # de-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        if p.exists() and p.is_file():
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                uniq.append(p)
    return uniq


def copy_into_assets(files: List[Path], dest_dir: Path) -> List[Tuple[Path, Path]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pairs: List[Tuple[Path, Path]] = []
    for src in files:
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        pairs.append((src, dst))
    return pairs

def load_source_map():
    if SOURCES_FILE.exists():
        try:
            return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def find_source_for(path: Path, source_map) -> Tuple[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    for rule in source_map:
        pat = rule.get("pattern")
        if pat and fnmatch(rel, pat):
            return rule.get("source", ""), rule.get("url", "")
    return "", ""


def write_category_page(name: str, slug: str, asset_pairs: List[Tuple[Path, Path]], html_links: List[Path], source_map):
    page = SITE / f"{slug}.md"
    lines: List[str] = []
    lines.append(f"# {name}\n")

    if html_links:
        # Copy HTML files into the assets folder so they are deployed with the site
        html_dest_dir = ASSETS / slug
        html_dest_dir.mkdir(parents=True, exist_ok=True)

        lines.append("Interactive Visualizations")
        lines.append("")
        for html in html_links:
            copied_html = html_dest_dir / html.name
            shutil.copy2(html, copied_html)
            rel = os.path.relpath(copied_html, SITE)
            lines.append(f"- [{html.name}]({rel})")
        lines.append("")
        lines.append("---\n")

    if not asset_pairs and not html_links:
        lines.append("No visualizations found yet in this category.\n")
    else:
        for src, local in asset_pairs:
            rel = os.path.relpath(local, SITE)
            caption = humanize(local.name)
            lines.append(f"![{caption}]({rel})")
            # Attempt to cite a data source
            src_text, src_url = find_source_for(src, source_map)
            if src_text:
                if src_url:
                    lines.append(f"_Source: [{src_text}]({src_url})_")
                else:
                    lines.append(f"_Source: {src_text}_")
            lines.append("")

    page.write_text("\n".join(lines), encoding="utf-8")


def write_index(categories_summary: List[Tuple[str, str, List[Path]]]):
    idx = SITE / "index.md"
    lines: List[str] = []
    lines.append("# Guyanese in the USA — Visual Insights\n")
    lines.append("A curated gallery of figures, maps, and analyses generated from this repository. Each section links to a page with all visuals.")
    lines.append("")

    for name, slug, thumbs in categories_summary:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"[Open full gallery →]({slug}.md)")
        lines.append("")
        if thumbs:
            # show a row of thumbnails
            for t in thumbs[:8]:
                rel = os.path.relpath(t, SITE)
                alt = humanize(Path(t).name)
                lines.append(f"![{alt}]({rel})")
            lines.append("")
        lines.append("---\n")

    idx.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_clean_dir(SITE)
    ASSETS.mkdir(parents=True, exist_ok=True)

    categories_summary: List[Tuple[str, str, List[Path]]] = []

    source_map = load_source_map()

    for spec in CATEGORY_SPECS:
        name = spec["name"]
        slug = spec["slug"]
        limit = spec.get("limit")
        image_globs = spec["globs"]
        html_globs = spec.get("link_html", [])

        src_images = find_files(image_globs)
        if limit:
            src_images = src_images[:limit]
        dest_dir = ASSETS / slug
        copied = copy_into_assets(src_images, dest_dir)

        html_links = find_files(html_globs)

        write_category_page(name, slug, copied, html_links, source_map)
        thumbs = [dst for _, dst in copied][:8]
        categories_summary.append((name, slug, thumbs))

    write_index(categories_summary)
    print(f"Wrote site to: {SITE}")


if __name__ == "__main__":
    main()
