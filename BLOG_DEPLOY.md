# Visual Blog: Build & Deploy

This repository now contains a Markdown-driven visual blog generated from all figures and map outputs found in the project.

## Content sources
- `figures/*.png`
- `education/figures/*.png`, `education/outputs/makechartsjs/out/*.png`
- `economy_healthcare/outputs/*.png`, `economy_healthcare/outputs/graphsinjs/out/*.png`
- `locality/maps/*.png` plus interactive `*.html` maps (copied into the site)
- `online_presence/analysis_out/*.png`
- `non_gov_openstreetmap/output/*.png`

## Generate site (Markdown)
This step scans the repo and writes Markdown pages into `site/` and copies assets under `site/assets/`.

```zsh
/Users/rohanramnarain/Documents/gy-usa-project/.venv/bin/python tools/generate_blog_markdown.py
```

## Build static site (HTML)
Converts Markdown pages into a styled static site in `public/` suitable for Firebase Hosting.

```zsh
/Users/rohanramnarain/Documents/gy-usa-project/.venv/bin/python tools/build_static_site.py
```

The output pages are:
- `public/index.html` (overview)
- `public/figures.html`
- `public/education.html`
- `public/economy-healthcare.html`
- `public/locality.html`
- `public/online-presence.html`
- `public/open-data.html`

Assets are in `public/assets/`.

## Deploy to Firebase Hosting
Prereqs: Node.js and Firebase CLI. Install the CLI if needed:

```zsh
npm install -g firebase-tools
```

Log in and select your Firebase project (once):

```zsh
firebase login
firebase projects:list
firebase use --add   # pick your project and name this alias (e.g. 'prod')
```

Deploy the static site:

```zsh
firebase deploy --only hosting
```

The hosting configuration is defined in `firebase.json` with `public` set to `public/`.

## Updating the blog
1. Re-run the generator to capture new figures:
   ```zsh
   /Users/rohanramnarain/Documents/gy-usa-project/.venv/bin/python tools/generate_blog_markdown.py
   ```
2. Rebuild the static site:
   ```zsh
   /Users/rohanramnarain/Documents/gy-usa-project/.venv/bin/python tools/build_static_site.py
   ```
3. Deploy again:
   ```zsh
   firebase deploy --only hosting
   ```

## Notes
- Interactive HTML maps from `locality/maps/*.html` are copied into `public/assets/locality/` and linked from the Locality page.
- Content organization can be adjusted in `tools/generate_blog_markdown.py` by editing the category specs.
 - Per-graph sources are controlled via `visual_sources.json` (glob → citation). Edit or add patterns and rebuild to update the source lines below each image.
