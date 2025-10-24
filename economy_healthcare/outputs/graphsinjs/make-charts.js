// make-charts.js — ESM (works with "type":"module")
// Generates high-res, transparent PNG bar charts for three CSVs (Canva-ready)
// Now with MUCH larger axis labels (category tick labels) without changing image size.

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse/sync";
import { ChartJSNodeCanvas } from "chartjs-node-canvas";
import Chart from "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";

// --- Register plugins & set dark text defaults ---
Chart.register(ChartDataLabels);
const TEXT = "#0b0c10"; // dark text everywhere
Chart.defaults.color = TEXT;
Chart.defaults.font.family =
  'Inter, -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial';

// >>> New: font sizes just for axis labels <<<
const CAT_TICK_FONT_SIZE = 26; // category-axis tick labels (the labels beside/under each bar)
const VAL_TICK_FONT_SIZE = 18; // value-axis tick labels (numbers on the scale)

// --- Paths ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Put your CSVs next to this script or edit these paths:
const FILES = [
  path.resolve(__dirname, "guyanese_insurance_mix_2023.csv"),
  path.resolve(__dirname, "guyanese_coverage_2023.csv"),
  path.resolve(__dirname, "guyanese_top_occupations_2023.csv"),
];

// Optional per-file overrides (title + orientation)
const FILE_CONFIG = [
  { match: /insurance_mix/i, title: "Guyanese Insurance Mix — 2023", indexAxis: "y" }, // horizontal
  { match: /coverage/i,      title: "Guyanese Coverage — 2023",      indexAxis: "x" }, // vertical
  { match: /occupations/i,   title: "Top Occupations — Guyanese (2023)", indexAxis: "y" }
];

// Output directory
const OUT_DIR = path.resolve(__dirname, "out");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

// --- Visuals / layout ---
const PALETTE = [
  "#7C3AED","#06B6D4","#22C55E","#F59E0B","#EC4899",
  "#10B981","#A855F7","#14B8A6","#F97316","#6366F1",
  "#84CC16","#FB7185"
];
const BASE_W = 2200;  // width in px (hi-res)
const BASE_H = 1400;  // base height; horizontal charts grow with #bars

// --- Helpers ---
const fmtThousands = (x) => {
  const n = Number(x);
  if (Number.isNaN(n)) return "";
  return Number.isInteger(n)
    ? n.toLocaleString()
    : n.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

function detectColumns(headers, rows) {
  const low = headers.map(h => h.toLowerCase());
  const catHints = ["category","label","name","segment","group","type","occupation","coverage","insurance","plan","class"];
  const valHints = ["value","count","number","total","amount","percent","percentage","share","rate"];

  let cat = null, val = null;

  for (const h of catHints) {
    const i = low.findIndex(x => x.includes(h));
    if (i !== -1) { cat = headers[i]; break; }
  }
  for (const h of valHints) {
    const i = low.findIndex(x => x.includes(h));
    if (i !== -1) { val = headers[i]; break; }
  }

  if (!cat) { // first non-numeric column
    outer: for (const h of headers) {
      for (const r of rows) {
        const v = r[h];
        if (v != null && v !== "") {
          if (isNaN(parseFloat(v))) { cat = h; break outer; }
        }
      }
    }
  }

  if (!val) { // numeric column with largest magnitude sum
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

  if ((!cat || !val) && headers.length === 2) {
    const [a,b] = headers;
    const aNum = rows.some(r => !isNaN(parseFloat(r[a])));
    const bNum = rows.some(r => !isNaN(parseFloat(r[b])));
    if (aNum && !bNum) { val = a; cat = b; }
    else if (bNum && !aNum) { val = b; cat = a; }
  }

  if (!cat || !val) {
    throw new Error("Could not infer category/value columns. Rename columns (e.g., 'occupation' and 'percent').");
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

function wrapLabel(s, width = 18) {
  const words = String(s).split(/\s+/);
  let line = "", out = [];
  for (const w of words) {
    if ((line + " " + w).trim().length > width) { out.push(line.trim()); line = w; }
    else line = (line + " " + w).trim();
  }
  if (line) out.push(line);
  return out; // Chart.js multi-line labels (array of lines)
}

function titleFrom(p) {
  const base = path.basename(p).replace(/\.[^.]+$/,"").replace(/_/g," ").trim();
  return base.replace(/\b\w/g, m => m.toUpperCase());
}

function autoAxis(cats) {
  const longest = cats.reduce((m, s) => Math.max(m, String(s || "").length), 0);
  return (cats.length > 6 || longest > 14) ? "y" : "x"; // 'y' = horizontal bars
}

function sortHighLow(labels, values) {
  const z = labels.map((l,i) => ({ l, v: values[i] })).sort((a,b) => b.v - a.v);
  return { labels: z.map(z => z.l), values: z.map(z => z.v) };
}

function readCSV(p) {
  const txt = fs.readFileSync(p, "utf8");
  const rows = parse(txt, { columns: true, skip_empty_lines: true });
  if (!rows.length) throw new Error(`No rows in ${p}`);
  return { headers: Object.keys(rows[0]), rows };
}

// --- Rendering ---
async function renderChart({ title, labels, values, indexAxis, percentMode, outPath }) {
  // Grow height for many horizontal bars (image size stays constant otherwise)
  const H = indexAxis === "y"
    ? Math.min(2600, Math.max(BASE_H, 62 * labels.length + 240))
    : BASE_H;

  const chartCanvas = new ChartJSNodeCanvas({
    width: BASE_W,
    height: H,
    backgroundColour: "transparent" // Canva-friendly
  });

  const asPercent = percentMode;
  const displayValues = values.map(v => (asPercent && v <= 1 ? v * 100 : v));
  const valueFmt = v => asPercent ? `${v.toFixed(1)}%` : fmtThousands(v);

  // >>> New: bigger padding on the category-axis side to fit larger tick labels (no image size change)
  const PAD_LEFT_FOR_Y = 260;   // left pad for horizontal (y-index) charts (category labels on left)
  const PAD_BOTTOM_FOR_X = 160; // bottom pad for vertical (x-index) charts (category labels below)

  const baseLayout = { top: 30, right: 24, bottom: 30, left: 24 };
  const layout =
    indexAxis === "y"
      ? { ...baseLayout, left: PAD_LEFT_FOR_Y }   // more room for big y-ticks
      : { ...baseLayout, bottom: PAD_BOTTOM_FOR_X }; // more room for big x-ticks

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
        categoryPercentage: 0.8
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
          font: { size: 28, weight: "700" },
          padding: { top: 10, bottom: 14 }
        },
        tooltip: {
          backgroundColor: "rgba(255,255,255,0.92)",
          titleColor: TEXT,
          bodyColor: TEXT,
          footerColor: TEXT,
          borderColor: "rgba(0,0,0,0.12)",
          borderWidth: 1,
          displayColors: false,
          callbacks: {
            title: (items) => (items?.[0]?.label || "").replace(/\n/g, " "),
            label: (item) => valueFmt(indexAxis === "y" ? item.parsed.x : item.parsed.y)
          }
        },
        datalabels: {
          color: TEXT,                             // value text stays dark
          backgroundColor: "rgba(255,255,255,0.85)", // set to 'transparent' to remove pill
          borderRadius: 6,
          padding: { top: 4, right: 6, bottom: 4, left: 6 },
          font: { weight: "600" },
          align: indexAxis === "y" ? "right" : "end",
          anchor: "end",
          offset: 6,
          formatter: (v) => valueFmt(v)
        }
      },
      // >>> Bigger tick font on the category axis; value axis uses a slightly smaller size
      scales: indexAxis === "y"
        ? {
            // Horizontal bars: x = value axis, y = category axis
            x: {
              grid: { color: "rgba(0,0,0,0.08)" },
              ticks: { color: TEXT, font: { size: VAL_TICK_FONT_SIZE }, callback: valueFmt }
            },
            y: {
              grid: { display: false },
              ticks: { color: TEXT, font: { size: CAT_TICK_FONT_SIZE, weight: "700" } } // BIG category labels
            }
          }
        : {
            // Vertical bars: x = category axis, y = value axis
            x: {
              grid: { display: false },
              ticks: {
                color: TEXT,
                font: { size: CAT_TICK_FONT_SIZE, weight: "700" } // BIG category labels
                // labels are multi-line arrays already; no rotation needed
              }
            },
            y: {
              grid: { color: "rgba(0,0,0,0.08)" },
              ticks: { color: TEXT, font: { size: VAL_TICK_FONT_SIZE }, callback: valueFmt }
            }
          }
    }
  };

  const buffer = await chartCanvas.renderToBuffer(config, "image/png");
  fs.writeFileSync(outPath, buffer);
}

// --- Main ---
(async () => {
  for (const csvPath of FILES) {
    if (!fs.existsSync(csvPath)) {
      console.warn(`⚠️ Missing: ${csvPath}`);
      continue;
    }

    const { headers, rows } = readCSV(csvPath);
    const { cat, val } = detectColumns(headers, rows);

    const rawLabels = rows.map(r => r[cat]);
    const rawValues = rows.map(r => parseFloat(r[val])).filter(v => Number.isFinite(v));

    const wrappedLabels = rawLabels.map(s => wrapLabel(s, 18));
    const { labels, values } = sortHighLow(wrappedLabels, rawValues);

    let title = titleFrom(csvPath);
    let indexAxis = autoAxis(rawLabels);
    const percentMode = isPercent(values, val);

    // Apply file-specific overrides
    for (const cfg of FILE_CONFIG) {
      if (cfg.match.test(path.basename(csvPath))) {
        if (cfg.title) title = cfg.title;
        if (cfg.indexAxis) indexAxis = cfg.indexAxis;
      }
    }

    const outName = path.basename(csvPath).replace(/\.[^.]+$/, "") + ".png";
    const outPath = path.join(OUT_DIR, outName);

    await renderChart({ title, labels, values, indexAxis, percentMode, outPath });
    console.log(`✅ Saved: ${outPath}`);
  }
})().catch(err => {
  console.error("❌ Error:", err);
  process.exit(1);
});
