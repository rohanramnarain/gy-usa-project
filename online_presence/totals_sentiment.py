
#!/usr/bin/env python3
"""
totals_sentiment.py

Purpose
-------
Load the scraper outputs (JSONL/CSV), standardize sources, run open-source
sentiment on every comment, and produce **totals by source** (counts + average
sentiment, etc.). No time series.

Default sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest
(robust, open-source). Optionally, you can use VADER for a lightweight lexicon
method via --method vader.

Install
-------
    pip install pandas transformers torch tqdm
    # (optional for VADER)
    pip install nltk

Usage
-----
    python totals_sentiment.py \

      --jsonl out.jsonl \

      --csv out.csv \

      --method hf \

      --model cardiffnlp/twitter-roberta-base-sentiment-latest \

      --batch-size 32 \

      --outdir totals_out

Outputs
-------
- totals_out/totals_by_source.csv
    Columns: source, count, avg_sentiment, std_sentiment, pos_share, neu_share, neg_share
- totals_out/overall_summary.txt
- (optional) totals_out/comments_with_sentiment.csv (add --save-enriched to enable)

Notes
-----
- Numeric sentiment = P(positive) - P(negative) in [-1, 1].
- Label shares use argmax label per comment (positive/neutral/negative).
- Sources are normalized (lowercased/stripped) with a few aliases (hn->hackernews).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm


# ---------- I/O ----------

def load_inputs(jsonl_path: str | None, csv_path: str | None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if jsonl_path and Path(jsonl_path).exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        frames.append(pd.DataFrame.from_records(records))
    if csv_path and Path(csv_path).exists():
        frames.append(pd.read_csv(csv_path, encoding="utf-8"))
    if not frames:
        raise SystemExit("No input files found. Provide at least --jsonl or --csv that exists.")
    df = pd.concat(frames, ignore_index=True)

    # Required columns
    for col in ["source", "id", "author", "text", "url", "created_at"]:
        if col not in df.columns:
            df[col] = None
    if "extra" not in df.columns:
        df["extra"] = None

    # Deduplicate exact records across inputs
    dedupe_key = (
        df["source"].astype(str) + "|" +
        df["id"].astype(str) + "|" +
        df["url"].astype(str) + "|" +
        df["created_at"].astype(str) + "|" +
        df["text"].astype(str)
    )
    df = df.loc[~dedupe_key.duplicated()].copy()

    # Clean/normalize
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Standardize source labels
    def _norm_source(s: Any) -> str:
        x = str(s).strip().lower()
        aliases = {
            "hn": "hackernews",
            "hacker news": "hackernews",
            "news.ycombinator": "hackernews",
            "github.com": "github",
        }
        return aliases.get(x, x)

    df["source"] = df["source"].map(_norm_source)
    return df


# ---------- Sentiment (two methods) ----------

def run_sentiment_hf(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str | None
) -> Tuple[List[float], List[str]]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    # Resolve device
    device_index = -1  # CPU
    if device is not None:
        if device.lower() == "cpu":
            device_index = -1
        elif device.lower() in ("cuda", "gpu"):
            device_index = 0
    else:
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                device_index = 0
        except Exception:
            device_index = -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Safe max length
    max_len = 512
    try:
        ml = int(getattr(tokenizer, "model_max_length", 512))
        if 0 < ml < 10000:
            max_len = min(max_len, ml)
    except Exception:
        pass

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
    )

    # Map labels
    id2label = {}
    try:
        cfg_map = getattr(model.config, "id2label", None)
        if isinstance(cfg_map, dict) and cfg_map:
            id2label = {int(k): str(v) for k, v in cfg_map.items()}
        elif isinstance(cfg_map, list) and cfg_map:
            id2label = {i: str(v) for i, v in enumerate(cfg_map)}
    except Exception:
        id2label = {}

    def _norm_label(raw: str) -> str:
        s = str(raw).lower()
        if s in ("negative", "neutral", "positive"):
            return s
        if s.startswith("label_") and id2label:
            try:
                idx = int(s.split("_")[1])
                return id2label.get(idx, s).lower()
            except Exception:
                return s
        return s

    numeric_scores: List[float] = []
    labels: List[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment (HF)", unit="batch"):
        batch = texts[i:i+batch_size]
        out = clf(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            top_k=None,   # return all labels
        )
        if isinstance(out, dict):
            out = [out]
        for per_text in out:
            probs = { _norm_label(d["label"]): float(d["score"]) for d in per_text }
            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            numeric = pos - neg
            numeric_scores.append(numeric)
            labels.append(max(probs, key=probs.get) if probs else "unknown")

    return numeric_scores, labels


def run_sentiment_vader(texts: List[str]) -> Tuple[List[float], List[str]]:
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            # Try to load; if missing, download the lexicon.
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
    except Exception as e:
        raise SystemExit(
            "\n[ERROR] VADER not available. Install NLTK and try again:\n"
            "  pip install nltk\n"
            f"Original error: {e}\n"
        )

    numeric_scores: List[float] = []
    labels: List[str] = []
    for t in tqdm(texts, desc="Sentiment (VADER)", unit="text"):
        s = sia.polarity_scores(t)
        numeric = float(s["pos"]) - float(s["neg"])  # keep same convention
        numeric_scores.append(numeric)
        # Argmax using compound threshold similar to common practice
        comp = s.get("compound", 0.0)
        if comp >= 0.05:
            labels.append("positive")
        elif comp <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    return numeric_scores, labels


# ---------- Aggregation ----------

def aggregate_totals(df: pd.DataFrame) -> pd.DataFrame:
    # label shares
    label_counts = (
        df.groupby(["source", "sentiment_label"])
          .size()
          .unstack(fill_value=0)
          .rename_axis(None, axis=1)
    )
    for col in ["positive", "neutral", "negative"]:
        if col not in label_counts.columns:
            label_counts[col] = 0

    counts = df.groupby("source").size().rename("count")
    means = df.groupby("source")["sentiment_score"].mean().rename("avg_sentiment")
    stds = df.groupby("source")["sentiment_score"].std(ddof=0).rename("std_sentiment")

    out = pd.concat([counts, means, stds, label_counts], axis=1).fillna(0)

    # shares
    total_by_source = out["count"].replace(0, pd.NA)
    out["pos_share"] = (out.get("positive", 0) / total_by_source).astype(float)
    out["neu_share"] = (out.get("neutral", 0) / total_by_source).astype(float)
    out["neg_share"] = (out.get("negative", 0) / total_by_source).astype(float)

    # order columns
    cols = ["count", "avg_sentiment", "std_sentiment", "pos_share", "neu_share", "neg_share"]
    out = out[cols]
    out = out.sort_values("count", ascending=False)
    out.index.name = "source"
    return out.reset_index()


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute totals and sentiment per source (no time series).")
    ap.add_argument("--jsonl", default="out.jsonl", help="Path to JSONL file from scraper (default: out.jsonl)")
    ap.add_argument("--csv", default="out.csv", help="Path to CSV file from scraper (default: out.csv)")
    ap.add_argument("--method", choices=["hf", "vader"], default="hf", help="Sentiment method: hf (Transformers) or vader (lexicon).")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="HF model name (used when --method hf).")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for HF inference (default: 32)")
    ap.add_argument("--device", choices=["cpu", "cuda", "gpu"], default=None, help="Force device for HF (default: auto)")
    ap.add_argument("--outdir", default="totals_out", help="Directory to write outputs (default: totals_out)")
    ap.add_argument("--save-enriched", action="store_true", help="Also save per-comment sentiment CSV.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.jsonl, args.csv)
    if df.empty:
        raise SystemExit("No rows after load/clean.")

    texts = df["text"].tolist()

    if args.method == "hf":
        scores, labels = run_sentiment_hf(texts, model_name=args.model, batch_size=args.batch_size, device=args.device)
    else:
        scores, labels = run_sentiment_vader(texts)

    df["sentiment_score"] = scores
    df["sentiment_label"] = labels

    totals = aggregate_totals(df)
    totals_path = outdir / "totals_by_source.csv"
    totals.to_csv(totals_path, index=False)

    overall_avg = float(df["sentiment_score"].mean())
    summary_path = outdir / "overall_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total comments: {len(df)}\n")
        f.write(f"Overall average sentiment (Ppos-Pneg in [-1,1]): {overall_avg:.4f}\n")
        f.write("\nTotals by source:\n")
        f.write(totals.to_csv(index=False))

    if args.save_enriched:
        enriched_path = outdir / "comments_with_sentiment.csv"
        df.to_csv(enriched_path, index=False)

    print("[INFO] Done.")
    print(f"- Totals CSV: {totals_path}")
    print(f"- Summary: {summary_path}")
    if args.save_enriched:
        print(f"- Enriched comments CSV: {enriched_path}")


if __name__ == "__main__":
    main()
