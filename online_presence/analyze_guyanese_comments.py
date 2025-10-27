
#!/usr/bin/env python3
"""
analyze_guyanese_comments.py  (v1.1)

Fixes in v1.1
-------------
- Prevents long-text crashes by enforcing explicit truncation with a safe
  max_length (defaults to 512) when running the sentiment model.
- Removes `return_all_scores` deprecation by using `top_k=None`.
- More robust label handling (supports 'LABEL_0/1/2' and 'negative/neutral/positive').

Outputs
-------
- analysis_out/comments_with_sentiment.csv
- analysis_out/counts_by_time_source.csv
- analysis_out/sentiment_timeseries_by_source.csv
- analysis_out/overall_summary.txt
- analysis_out/counts_timeseries.png
- analysis_out/sentiment_timeseries.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


def _safe_import_transformers(model_name: str):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    except Exception as e:
        raise SystemExit(
            "\n[ERROR] The 'transformers' package is required.\n"
            "Install dependencies first: pip install transformers torch\n"
            f"Original import error: {e}\n"
        )
    return AutoTokenizer, AutoModelForSequenceClassification, pipeline


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

    # Normalize expected columns
    for col in ["source", "id", "author", "text", "url", "created_at"]:
        if col not in df.columns:
            df[col] = None
    if "extra" not in df.columns:
        df["extra"] = None

    # Deduplicate
    dedupe_key = (
        df["source"].astype(str) + "|" +
        df["id"].astype(str) + "|" +
        df["url"].astype(str) + "|" +
        df["created_at"].astype(str) + "|" +
        df["text"].astype(str)
    )
    df = df.loc[~dedupe_key.duplicated()].copy()

    # Parse datetime
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).reset_index(drop=True)

    # Text cleanup
    df["text"] = (df["text"].astype(str).fillna("").str.strip())
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    df["source"] = df["source"].astype(str).fillna("unknown")
    return df


def build_time_series_counts(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = (
        df
        .set_index("created_at")
        .groupby([pd.Grouper(freq=freq), "source"])  # bucket x source
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = grouped.pivot(index="created_at", columns="source", values="count").fillna(0).astype(int)
    return pivot


def plot_counts(pivot_counts: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in pivot_counts.columns:
        ax.plot(pivot_counts.index, pivot_counts[col], label=str(col))
    ax.set_title("Comment Counts Over Time by Source")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_sentiment(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    AutoTokenizer, AutoModelForSequenceClassification, pipe = _safe_import_transformers(model_name)

    # Device resolution
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

    # Determine a safe max length (cap at 512 to avoid RoBERTa 514-pos embedding mismatch).
    max_len = 512
    try:
        ml = int(getattr(tokenizer, "model_max_length", 512))
        if 0 < ml < 10000:
            max_len = min(max_len, ml)
    except Exception:
        pass

    clf = pipe(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        # NOTE: we pass truncation/max_length/top_k at call-time for clarity.
    )

    # Build id2label map (handles 'LABEL_0' etc.)
    id2label = {}
    try:
        cfg_map = getattr(model.config, "id2label", None)
        if isinstance(cfg_map, dict) and cfg_map:
            # keys may be str or int
            id2label = {int(k): str(v) for k, v in cfg_map.items()}
        elif isinstance(cfg_map, list) and cfg_map:
            id2label = {i: str(v) for i, v in enumerate(cfg_map)}
    except Exception:
        id2label = {}

    def normalize_label(raw: str) -> str:
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

    texts = df["text"].tolist()
    results: List[List[Dict[str, Any]]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment", unit="batch"):
        batch = texts[i:i+batch_size]
        out = clf(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            top_k=None,            # return all labels/scores (replaces return_all_scores=True)
        )
        if isinstance(out, dict):
            out = [out]
        results.extend(out)

    numeric_scores = []
    labels = []
    for per_text in results:
        probs = {normalize_label(d["label"]): float(d["score"]) for d in per_text}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        neu = probs.get("neutral", 0.0)
        numeric = pos - neg  # [-1, 1]
        numeric_scores.append(numeric)
        label = max(probs, key=probs.get) if probs else "unknown"
        labels.append(label)

    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = numeric_scores
    return df


def build_sentiment_timeseries(df_sent: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = (
        df_sent
        .set_index("created_at")
        .groupby([pd.Grouper(freq=freq), "source"])  # bucket x source
        ["sentiment_score"].mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="created_at", columns="source", values="sentiment_score").astype(float)
    return pivot


def plot_sentiment(pivot_sent: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in pivot_sent.columns:
        ax.plot(pivot_sent.index, pivot_sent[col], label=str(col))
    ax.set_title("Average Sentiment Over Time by Source (P(pos)-P(neg))")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Sentiment [-1, 1]")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze counts and sentiment for collected comments.")
    ap.add_argument("--jsonl", default="out.jsonl", help="Path to JSONL file from scraper (default: out.jsonl)")
    ap.add_argument("--csv", default="out.csv", help="Path to CSV file from scraper (default: out.csv)")
    ap.add_argument("--freq", default="W", help="Time bucket frequency (Pandas offset alias, e.g., D, W, M). Default: W")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="HF model name for sentiment.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for sentiment inference (default: 32)")
    ap.add_argument("--device", choices=["cpu", "cuda", "gpu"], default=None, help="Force device (default: auto)")
    ap.add_argument("--outdir", default="analysis_out", help="Directory to write outputs (default: analysis_out)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading inputs...")
    df = load_inputs(args.jsonl, args.csv)
    if df.empty:
        raise SystemExit("No rows after load/clean.")

    print("[INFO] Building count time series...")
    pivot_counts = build_time_series_counts(df, args.freq)
    counts_csv = outdir / "counts_by_time_source.csv"
    pivot_counts.to_csv(counts_csv, index=True)

    counts_png = outdir / "counts_timeseries.png"
    plot_counts(pivot_counts, counts_png)

    print("[INFO] Running sentiment model... This may download weights on first run.")
    df_sent = run_sentiment(df, model_name=args.model, batch_size=args.batch_size, device=args.device)

    enriched_csv = outdir / "comments_with_sentiment.csv"
    df_sent.to_csv(enriched_csv, index=False)

    print("[INFO] Building sentiment time series...")
    pivot_sent = build_sentiment_timeseries(df_sent, args.freq)
    sent_csv = outdir / "sentiment_timeseries_by_source.csv"
    pivot_sent.to_csv(sent_csv, index=True)

    sent_png = outdir / "sentiment_timeseries.png"
    plot_sentiment(pivot_sent, sent_png)

    overall_avg = float(df_sent["sentiment_score"].mean()) if not df_sent.empty else float("nan")
    summary_txt = outdir / "overall_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        total = len(df_sent)
        by_source = df_sent.groupby("source").size().to_dict()
        f.write(f"Total comments: {total}\n")
        f.write(f"Counts by source: {by_source}\n")
        f.write(f"Overall average sentiment (Ppos-Pneg in [-1,1]): {overall_avg:.4f}\n")

    print("[INFO] Done.")
    print(f"- Counts CSV: {counts_csv}")
    print(f"- Counts plot: {counts_png}")
    print(f"- Enriched comments CSV: {enriched_csv}")
    print(f"- Sentiment CSV: {sent_csv}")
    print(f"- Sentiment plot: {sent_png}")
    print(f"- Summary: {summary_txt}")


if __name__ == "__main__":
    main()
