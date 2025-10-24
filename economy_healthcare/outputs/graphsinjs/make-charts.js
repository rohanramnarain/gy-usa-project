// make-charts.js  (ESM version for "type":"module")
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse/sync";
import { ChartJSNodeCanvas } from "chartjs-node-canvas";
import Chart from "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";

// Register plugin for Chart.js v4
Chart.register(ChartDataLabels);

// Resolve this file's folder
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- INPUT FILES (put your CSVs next to this script, or change paths) ----
const FILES = [
  path.resolve(__dirname, "guyanese_insurance_mix_2023.csv"),
  path.resolve(__dirname, "guyanese_coverage_2023.csv"),
  path.resolve(__dirname, "guyanese_top_occupations_2023.csv"),
];

// Optional per-file overrides (title + orientation)
const FILE_CONFIG = [
  { match: /insurance_mix/i, title: "Guyanese Insurance Mix — 2023", indexAxis: "y" },
  { match: /coverage/i,      title: "Guyanese Coverage — 2023",      indexAxis: "x" },
  { match: /occupations/i,   title: "Top Occupations — Guyanese (2023)", indexAxis: "y" },
];

// ---- OUTPUT DIR ----
const OUT_DIR = path.resolve(__dirname, "out");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

// ---- Visuals ----
const PALETTE = ["#7C3AED","#06B6D4","#22C55E","#F59E0B","#EC4899",
                 "#10B981","#A855F7","#14B8A6","#F97316","#6366F1",
                 "#84CC16","#FB7185"];
const BASE_W = 2200; // high-res, Canva-ready
const BASE_H = 1400;

const fmtThousands = (x) => {
  const n = Number(x); if (Number.isNaN(n)) return "";
  return Number.isInteger(n) ? n.toLocaleString() : n.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

function detectColumns(headers, rows) {
  const low = headers.map(h => h.toLowerCase());
  const catHints = ["category","label","name","segment","group","type","occupation","coverage","insurance","plan","class"];
  const valHints = ["value","count","number","total","amount","percent","percentage","share","rate"];
  let cat = null, val = null;
  for (const h of catHints){ const i=low.findIndex(x=>x.includes(h)); if(i!==-1){cat=headers[i]; break;} }
  for (const h of valHints){ const i=low.findIndex(x=>x.includes(h)); if(i!==-1){val=headers[i]; break;} }
  if (!cat) { outer: for (const h of headers){ for (const r of rows){
      const v=r[h]; if (v!=null && v!==""){ if (isNaN(parseFloat(v))){ cat=h; break outer; } } } } }
  if (!val) {
    let best={ col:null, sum:-Infinity };
    for (const h of headers){
      let s=0, ok=false;
      for (const r of rows){ const v=parseFloat(r[h]); if(!isNaN(v)){ ok=true; s+=Math.abs(v); } }
      if (ok && s>best.sum) best={ col:h, sum:s };
    }
    val = best.col;
  }
  if ((!cat||!val) && headers.length===2){
    const [a,b]=headers;
    const aNum = rows.some(r => !isNaN(parseFloat(r[a])));
    const bNum = rows.some(r => !isNaN(parseFloat(r[b])));
    if (aNum && !bNum) { val=a; cat=b; }
    else if (bNum && !aNum) { val=b; cat=a; }
  }
  if (!cat || !val) throw new Error("Could not infer category/value columns.");
  return { cat, val };
}

function isPercent(values, valName="") {
  const hint = /percent|percentage|share|rate/i.test(valName);
  const fin = values.filter(Number.isFinite);
  if (!fin.length) return hint;
  const max = Math.max(...fin), min = Math.min(...fin);
  return hint || (max <= 1 && min >= 0) || (hint && max <= 100);
}

function wrapLabel(s, width=18) {
  const words = String(s).split(/\s+/); let line="", out=[];
  for (const w of words) {
    if ((line + " " + w).trim().length > width) { out.push(line.trim()); line=w; }
    else line = (line + " " + w).trim();
  }
  if (line) out.push(line);
  return out; // Chart.js multi-line labels
}

function titleFrom(p) {
  const base = path.basename(p).replace(/\.[^.]+$/,"").replace(/_/g," ").trim();
  return base.replace(/\b\w/g, m => m.toUpperCase());
}

function autoAxis(cats) {
  const longest = cats.reduce((m, s) => Math.max(m, String(s||"").length), 0);
  return (cats.length > 6 || longest > 14) ? "y" : "x"; // 'y' = horizontal bars
}

function sortHighLow(labels, values) {
  const z = labels.map((l,i)=>({l,v:values[i]})).sort((a,b)=>b.v - a.v);
  return { labels: z.map(z=>z.l), values: z.map(z=>z.v) };
}

async function renderChart({ title, labels, values, indexAxis, percentMode, outPath }) {
  const H = indexAxis === "y" ? Math.min(2400, Math.max(BASE_H, 62*labels.length + 240)) : BASE_H;

  // ChartJSNodeCanvas v5 works with Chart.js v4
  const chartCanvas = new ChartJSNodeCanvas({ width: BASE_W, height: H, backgroundColour: "transparent" });

  const asPercent = percentMode;
  const displayValues = values.map(v => (asPercent && v <= 1 ? v * 100 : v));
  const valueFmt = v => asPercent ? `${v.toFixed(1)}%` : fmtThousands(v);

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
      layout: { padding: { top: 30, right: 24, bottom: 30, left: 24 } },
      plugins: {
        legend: { display: false },
        title: {
          display: true, text: title, color: "#0f1115",
          font: { size: 28, weight: "700" }, padding: { top: 10, bottom: 14 }
        },
        tooltip: {
          backgroundColor: "rgba(17,18,24,.95)",
          borderColor: "rgba(255,255,255,.08)",
          borderWidth: 1,
          displayColors: false,
          callbacks: {
            title: (items) => (items?.[0]?.label || "").replace(/\n/g," "),
            label: (item) => valueFmt(indexAxis === "y" ? item.parsed.x : item.parsed.y)
          }
        },
        datalabels: {
          color: "#0B0C10",
          backgroundColor: "rgba(255,255,255,0.85)",
          borderRadius: 6,
          padding: { top:4,right:6,bottom:4,left:6 },
          font: { weight: "600" },
          align: indexAxis === "y" ? "right" : "end",
          anchor: "end",
          offset: 6,
          formatter: (v) => valueFmt(v)
        }
      },
      scales: indexAxis === "y"
        ? { x: { grid: { color: "rgba(0,0,0,0.08)" }, ticks: { callback: valueFmt } }, y: { grid: { display:false } } }
        : { x: { grid: { display:false } }, y: { grid: { color: "rgba(0,0,0,0.08)" }, ticks: { callback: valueFmt } } }
    }
  };

  const buffer = await chartCanvas.renderToBuffer(config, "image/png");
  fs.writeFileSync(outPath, buffer);
}

function readCSV(p) {
  const txt = fs.readFileSync(p, "utf8");
  const rows = parse(txt, { columns: true, skip_empty_lines: true });
  if (!rows.length) throw new Error(`No rows in ${p}`);
  return { headers: Object.keys(rows[0]), rows };
}

(async () => {
  for (const csvPath of FILES) {
    if (!fs.existsSync(csvPath)) { console.warn(`⚠️ Missing: ${csvPath}`); continue; }
    const { headers, rows } = readCSV(csvPath);
    const { cat, val } = detectColumns(headers, rows);

    const rawLabels = rows.map(r => r[cat]);
    const rawValues = rows.map(r => parseFloat(r[val])).filter(v => Number.isFinite(v));

    const wrappedLabels = rawLabels.map(s => wrapLabel(s, 18));
    const { labels, values } = sortHighLow(wrappedLabels, rawValues);

    let title = titleFrom(csvPath);
    let indexAxis = autoAxis(rawLabels);
    const percentMode = isPercent(values, val);

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
