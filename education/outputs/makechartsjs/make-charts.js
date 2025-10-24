// make-charts.js — ESM (works with "type":"module")
// Purpose-built for the two PUMS files.
// Maps NATIVITY codes: 1 → "US Native", 2 → "Foreign-born".
// Produces high-res transparent PNGs with dark, LARGE category labels.

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse/sync";
import { ChartJSNodeCanvas } from "chartjs-node-canvas";
import Chart from "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";

// ---------- Global chart styling (dark text + bigger fonts) ----------
Chart.register(ChartDataLabels);
const TEXT = "#0b0c10";
Chart.defaults.color = TEXT;
Chart.defaults.font.family =
  'Inter, -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial';

// BIG category labels + readable value ticks
const CAT_TICK_FONT_SIZE = 28; // beside/under each bar
const VAL_TICK_FONT_SIZE = 20; // numbers on axis

// ---------- Paths ----------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FILES = [
  path.resolve(__dirname, "pums_guyanese_educ_attainment_by_nativity.csv"),
  path.resolve(__dirname, "pums_guyanese_educ_attainment_national.csv"),
];

// File-specific titles; forced horizontal bars for readability
const FILE_CONFIG = [
  { match: /by[_-]?nativity/i, title: "Educational Attainment by Nativity — Guyanese (PUMS)", indexAxis: "y" },
  { match: /national/i,        title: "Educational Attainment — Guyanese (National, PUMS)",   indexAxis: "y" },
];

// ---------- Output ----------
const OUT_DIR = path.resolve(__dirname, "out");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

// ---------- Visuals / sizing ----------
const PALETTE = [
  "#7C3AED","#06B6D4","#22C55E","#F59E0B","#EC4899",
  "#10B981","#A855F7","#14B8A6","#F97316","#6366F1",
  "#84CC16","#FB7185"
];
const BASE_W = 2200; // hi-res width
const BASE_H = 1400; // base height; horizontal charts expand with bar count

// ---------- Helpers ----------
const fmtThousands = (x) => {
  const n = Number(x);
  if (Number.isNaN(n)) return "";
  return Number.isInteger(n)
    ? n.toLocaleString()
    : n.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

// NATIVITY mapping
const NATIVITY_MAP = new Map([
  ["1", "US Native"],
  ["2", "Foreign-born"],
]);
const mapNativity = (v) => {
  const s = String(v ?? "").trim();
  return NATIVITY_MAP.get(s) ?? s;
};

// Hard-biased detection tuned for edu attainment tables
function pickColumns(headers, rows) {
  const low = headers.map(h => h.toLowerCase());

  // Education/nativity category
  const catPrefs = [
    "nativity", "education", "educational_attainment", "edu", "level",
    "category", "label", "name"
  ];
  // Value column (prefer percent/share; fallback to count)
  const valPrefs = [
    "percent", "percentage", "share", "rate", "count", "number", "total"
  ];

  let cat = null, val = null;

  for (const pref of catPrefs) {
    const i = low.findIndex(h => h.includes(pref));
    if (i !== -1) { cat = headers[i]; break; }
  }
  for (const pref of valPrefs) {
    const i = low.findIndex(h => h.includes(pref));
    if (i !== -1) { val = headers[i]; break; }
  }

  // Fallbacks
  if (!cat) {
    outer: for (const h of headers) {
      for (const r of rows) {
        const v = r[h];
        if (v != null && v !== "") {
          if (isNaN(parseFloat(v))) { cat = h; break outer; }
        }
      }
    }
  }
  if (!val) {
    let best = { col: null, sum: -Infinity };
    for (const h of headers) {
      let s = 0, ok = false;
      for (const r of rows) {
        const v = parseFloat(r[h]);
        if (!isNaN(v)) { ok = true; s += Math.abs(v); }
      }
      if (ok && s > best.sum) best = { col: h, sum: s };
    }
    val = best.col;
  }

  if (!cat || !val) {
    throw new Error(`Could not infer columns. Headers: ${headers.join(", ")}`);
  }
  return { cat, val };
}

function isPercent(values, valName = "") {
  const hint = /percent|percentage|share|rate/i.test(valName);
  const fin = values.filter(Number.isFinite);
  if (!fin.length) return hint;
  const max = Math.max(...fin), min = Math.min(...fin);
  return hint || (max <= 1 && min >= 0) || (hint && max <= 100);
}

// Wrap long labels to multiple lines (Chart.js supports arrays)
function wrapLabel(s, width = 26) {
  const words = String(s).split(/\s+/);
  let line = "", out = [];
  for (const w of words) {
    if ((line + " " + w).trim().length > width) { out.push(line.trim()); line = w; }
    else line = (line + " " + w).trim();
  }
  if (line) out.push(line);
  return out;
}

function readCSV(p) {
  const txt = fs.readFileSync(p, "utf8");
  const rows = parse(txt, { columns: true, skip_empty_lines: true });
  if (!rows.length) throw new Error(`No rows in ${p}`);
  return { headers: Object.keys(rows[0]), rows };
}

function sortHighLow(labels, values) {
  const z = labels.map((l,i)=>({ l, v: values[i] })).sort((a,b)=>b.v - a.v);
  return { labels: z.map(z=>z.l), values: z.map(z=>z.v) };
}

// ---------- Rendering ----------
async function renderChart({ title, labels, values, outPath }) {
  // Force horizontal bars (long labels)
  const indexAxis = "y";
  const H = Math.min(2600, Math.max(BASE_H, 62 * labels.length + 260));

  const chartCanvas = new ChartJSNodeCanvas({
    width: BASE_W,
    height: H,
    backgroundColour: "transparent"
  });

  // Percent detection & formatting
  const assumePercent = isPercent(values, "percent");
  const displayValues = values.map(v => (assumePercent && v <= 1 ? v * 100 : v));
  const valueFmt = v => assumePercent ? `${v.toFixed(1)}%` : fmtThousands(v);

  // Extra left padding so big category labels don’t clip
  const layout = { top: 30, right: 28, bottom: 36, left: 280 };

  const config = {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: displayValues,
        backgroundColor: labels.map((_, i) => PALETTE[i % PALETTE.length]),
        borderColor: "rgba(0,0,0,0.10)",
        borderWidth: 1,
        borderRadius: 10,
        barPercentage: 0.75,
        categoryPercentage: 0.82
      }]
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      indexAxis,
      layout,
      plugins: {
        legend: { display: false, labels: { color: TEXT } },
        title: {
          display: true,
          text: title,
          color: TEXT,
          font: { size: 30, weight: "700" },
          padding: { top: 10, bottom: 14 }
        },
        tooltip: {
          backgroundColor: "rgba(255,255,255,0.94)",
          titleColor: TEXT,
          bodyColor: TEXT,
          borderColor: "rgba(0,0,0,0.12)",
          borderWidth: 1,
          displayColors: false,
          callbacks: {
            title: (items) => (items?.[0]?.label || "").replace(/\n/g, " "),
            label: (item) => valueFmt(item.parsed.x)
          }
        },
        datalabels: {
          color: TEXT,
          backgroundColor: "rgba(255,255,255,0.85)", // set to 'transparent' to remove the pill
          borderRadius: 6,
          padding: { top: 4, right: 6, bottom: 4, left: 6 },
          font: { weight: "600" },
          align: "right",
          anchor: "end",
          offset: 6,
          formatter: (v) => valueFmt(v)
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(0,0,0,0.08)" },
          ticks: { color: TEXT, font: { size: VAL_TICK_FONT_SIZE }, callback: valueFmt }
        },
        y: {
          grid: { display: false },
          ticks: { color: TEXT, font: { size: CAT_TICK_FONT_SIZE, weight: "700" } }
        }
      }
    }
  };

  const buffer = await chartCanvas.renderToBuffer(config, "image/png");
  fs.writeFileSync(outPath, buffer);
}

// ---------- Main ----------
(async () => {
  for (const csvPath of FILES) {
    if (!fs.existsSync(csvPath)) { console.warn(`⚠️ Missing: ${csvPath}`); continue; }

    const { headers, rows } = readCSV(csvPath);
    const { cat, val } = pickColumns(headers, rows);

    // Raw labels + NATIVITY mapping (only if column name or filename indicates nativity, or values are just 1/2)
    const lowerCatName = cat.toLowerCase();
    const isNativityFile = /by[_-]?nativity/i.test(path.basename(csvPath)) || /nativity/.test(lowerCatName);

    const rawLabels = rows.map(r => r[cat]);
    const nativityLike = isNativityFile ||
      (new Set(rawLabels.map(v => String(v).trim()))).size <= 3 &&
      rawLabels.every(v => ["1","2"," 1"," 2",1,2].includes(v));

    const mappedLabels = nativityLike ? rawLabels.map(mapNativity) : rawLabels;

    const rawValues = rows.map(r => parseFloat(r[val])).filter(v => Number.isFinite(v));

    // Wrap labels wide for clarity
    const wrappedLabels = mappedLabels.map(s => wrapLabel(s, 26));

    // Sort high → low
    const { labels, values } = sortHighLow(wrappedLabels, rawValues);

    // Title + config by filename
    let title = path.basename(csvPath).replace(/\.[^.]+$/,"").replace(/_/g," ").replace(/\b\w/g, m => m.toUpperCase());
    for (const cfg of FILE_CONFIG) {
      if (cfg.match.test(path.basename(csvPath))) {
        if (cfg.title) title = cfg.title;
      }
    }

    const outName = path.basename(csvPath).replace(/\.[^.]+$/, "") + ".png";
    const outPath = path.join(OUT_DIR, outName);

    await renderChart({ title, labels, values, outPath });
    console.log(`✅ Saved: ${outPath}`);
  }
})().catch(err => {
  console.error("❌ Error:", err);
  process.exit(1);
});
