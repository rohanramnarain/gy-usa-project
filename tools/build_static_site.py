import re
import shutil
from pathlib import Path
from typing import Optional

import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PUBLIC = ROOT / "public"

TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"/>
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin/>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap\" rel=\"stylesheet\"/>
  <link rel=\"stylesheet\" href=\"/styles.css\" />
</head>
<body>
  <header>
    <div class=\"container\">
      <h1><a href=\"/index.html\">Guyanese in the USA — Visual Insights</a></h1>
      <nav>
        <a href=\"/index.html\">Home</a>
        <a href=\"/figures.html\">Figures</a>
        <a href=\"/education.html\">Education</a>
        <a href=\"/economy-healthcare.html\">Economy & Healthcare</a>
        <a href=\"/locality.html\">Locality</a>
        <a href=\"/online-presence.html\">Online</a>
        <a href=\"/open-data.html\">Open Data</a>
      </nav>
    </div>
  </header>
  <main class=\"container\">{content}</main>
  <footer>
    <div class=\"container\">Built from Markdown on {built_from} • <a href=\"https://firebase.google.com/products/hosting\" target=\"_blank\">Firebase Hosting</a></div>
  </footer>
</body>
</html>"""

CSS = """:root{--bg:#ffffff;--panel:#ffffff;--text:#111827;--muted:#6b7280;--accent:#2563eb}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
header{background:var(--panel);border-bottom:1px solid #e5e7eb}
header .container{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px 16px;padding:16px}
header h1{font-size:18px;margin:0;font-weight:600;white-space:nowrap}
header nav{display:flex;flex-wrap:wrap;gap:14px;margin-left:auto}
header nav a{color:var(--muted);white-space:nowrap}
.container{max-width:980px;margin:0 auto;padding:24px}
main img{max-width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb;margin:12px 0}
main h1{font-size:32px;margin-top:0}
main h2{margin-top:32px}
main p,main li{color:#374151}
hr, main hr{border:none;border-top:1px solid #e5e7eb;margin:28px 0}
footer{border-top:1px solid #e5e7eb;color:var(--muted)}
footer .container{padding:16px}
"""


def extract_title(html: str, fallback: str) -> str:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if m:
        # strip tags inside h1 if any
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return fallback


def convert_md_to_html(md_path: Path, out_path: Path):
    text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["extra", "tables", "toc"])  # type: ignore
    # Rewrite markdown page links (*.md) to their built HTML equivalents (*.html)
    html = re.sub(r"href=\"([^\"]+)\.md\"", lambda m: f"href=\"{m.group(1)}.html\"", html)
    title = extract_title(html, md_path.stem.title())
    wrapped = TEMPLATE.format(title=title, content=html, built_from=md_path.name)
    out_path.write_text(wrapped, encoding="utf-8")


def main():
    if not SITE.exists():
        raise SystemExit("Run tools/generate_blog_markdown.py first to create the site/ directory.")

    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    # Write CSS
    (PUBLIC / "styles.css").write_text(CSS, encoding="utf-8")

    # Copy assets
    assets_src = SITE / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, PUBLIC / "assets")

    # Convert markdown pages
    for md in sorted(SITE.glob("*.md")):
        out_html = PUBLIC / f"{md.stem}.html"
        convert_md_to_html(md, out_html)

    print(f"Built static site at: {PUBLIC}")


if __name__ == "__main__":
    main()
