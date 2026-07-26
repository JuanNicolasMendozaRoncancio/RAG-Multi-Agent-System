"""
Sentiment Benchmark
Compares three sentiment classifiers on the CURATED corpus:
 
  1. LogisticRegression on 384-dim embeddings (sklearn baseline)
  2. Groq   llama-3.1-8b-instant  (primary LLM)
  3. Gemini gemini-2.0-flash       (fallback LLM)
 
Why this design:
- No manual ground truth exists. CURATED labels were produced by Groq/Gemini
  during the sentiment pipeline. Using those same labels as ground truth for
  the LLM classifiers would be circular.
- LogisticRegression is evaluated with 5-fold cross-validation over CURATED
  embeddings+labels. This is a valid ML evaluation: it measures whether the
  384-dim embedding space carries enough signal to predict the LLM-assigned
  sentiment without an LLM. If LR F1 is high, the embedding space already
  encodes sentiment and the LLM adds little. If LR F1 is low, the LLM is
  doing non-trivial work.
- Groq vs Gemini are compared by re-classifying a held-out sample of 30
  articles and measuring inter-model agreement (Cohen's kappa + raw accuracy
  relative to the original CURATED label). This is the honest comparison:
  we cannot say which is "better" without human annotations, but we can
  measure consistency.
 
Output:
  - Console report with all metrics
  - docs/sentiment_benchmark.md — markdown table for the README
 
Run from project root:
    python -m nlp_worker.sentiment_benchmark
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from shared.db import get_db, COL_CURATED
from shared.llm_client import chat_complete

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SAMPLE_SIZE = 30
_RANDOM_SEED = 42
 
_SENTIMENT_SYSTEM_PROMPT = """You are a sentiment analysis engine for climate and energy journalism.
Respond ONLY with a JSON object. No preamble, no explanation, no markdown fences.
The JSON must contain exactly these four keys:
- "sentiment": one of "positive", "negative", "neutral"
- "intensity": a float between 0.0 and 1.0
- "principal_subject": a short string (max 10 words) naming the main topic
- "main_argument": one sentence summarising the article's central argument
"""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
async def _load_curated(db: Any) -> list[dict[str, Any]]:
    """
    Load all CURATED documents that have both 'sentiment' and 'embedding'.
 
    Why require embedding: LogisticRegression needs the 384-dim vector.
    Documents without embedding were not processed by the embedder and
    cannot participate in the LR evaluation.
 
    Why load from CURATED (not CLEAN): CURATED is the single collection
    that has both the LLM sentiment label and the article text.
    """
    docs = await db[COL_CURATED].find(
        {"sentiment": {"$exists": True}, "embedding": {"$exists": True}},
        {"url": 1, "title": 1, "text": 1, "sentiment": 1, "source": 1,
         "detected_language": 1, "embedding": 1, "_id": 0},
    ).to_list(length=None)

    logger.info("Loaded %d CURATED docs with embeddings.", len(docs))
    return docs

# ---------------------------------------------------------------------------
# Classifier 1 — LogisticRegression baseline
# ---------------------------------------------------------------------------
def _evaluate_logistic_regression(
    docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate LogisticRegression with 5-fold stratified cross-validation.
    """
    X = np.array([d["embedding"] for d in docs], dtype= np.float32)
    y_raw = [d["sentiment"] for d in docs]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    clf = LogisticRegression(max_iter=1000,
                             random_state= _RANDOM_SEED,
                             C = 1.0)
    cv = StratifiedKFold(n_splits=5,
                         shuffle= True,
                         random_state= _RANDOM_SEED)

    y_pred = cross_val_predict(clf, X, y, cv = cv)
    y_pred_labels = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_raw, y_pred_labels)
    f1_macro = f1_score(y_raw, y_pred_labels, average="macro")
    report = classification_report(y_raw, y_pred_labels, output_dict= True)

    unique, counts = np.unique(y_raw, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(
        "LogisticRegression CV — accuracy=%.3f, F1-macro=%.3f", accuracy, f1_macro
    )
    return {
        "classifier": "LogisticRegression (5-fold CV)",
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "class_distribution": distribution,
        "per_class_f1": {
            cls: round(report[cls]["f1-score"], 4)
            for cls in report
            if cls in ("positive", "negative", "neutral")
        },
        "n_samples": len(docs),
    }

# ---------------------------------------------------------------------------
# Classifier 2 & 3 — Groq and Gemini live re-classification
# ---------------------------------------------------------------------------
def _classify_with_provider(
        article: dict[str, Any],
        provider: str,
) -> str | None:
    """
    Re-classify one article with a specific LLM provider.
 
    Returns the sentiment string or None if the call fails.
    """
    os.environ["LLM_PROVDER"] = provider

    title = article.get("title") or ""
    text = article.get("text") or ""
    messages = [
        {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {title}\n\nText: {text[:1500]}"},
    ]

    try:
        response = chat_complete(messages, json_mode=True)
        raw = response.get("content", "")
        parsed = json.loads(raw)
        return parsed.get("sentiment")
    except Exception as exc:
        logger.warning("Provider %s failed for '%s': %s", provider, article.get("url"), exc)
        return None

def _evaluate_llm_provider(
    sample: list[dict[str, Any]],
    provider: str
) -> dict[str, Any]:
    """
    Re-classify the sample with the given provider and compute metrics
    against the original CURATED label (produced by the pipeline).
 
    Metric interpretation:
    - accuracy vs CURATED: how often does this provider agree with the
      original pipeline label? High agreement = consistent behavior.
    - Cohen's kappa: agreement corrected for chance. Kappa > 0.6 is
      considered substantial agreement.
    - Note: this does NOT measure correctness — it measures consistency
      with the original pipeline labels, which themselves have no ground truth.
    """
    original_labels = []
    predicted_labels = []
    failed = 0

    for doc in sample:
        pred = _classify_with_provider(doc, provider)
        if pred is None:
            failed += 1
            continue
        original_labels.append(doc["sentiment"])
        predicted_labels.append(pred)

    if not predicted_labels:
        logger.error("Provider %s failed on all %d articles.", provider, len(sample))
        return {
            "classifier": provider,
            "accuracy_vs_curated": None,
            "kappa_vs_curated": None,
            "f1_macro_vs_curated": None,
            "failed": failed,
            "n_samples": len(sample),
        }
 
    accuracy = accuracy_score(original_labels, predicted_labels)
    kappa = cohen_kappa_score(original_labels, predicted_labels)
    f1_macro = f1_score(
        original_labels, predicted_labels, average="macro", zero_division=0
    )
 
    logger.info(
        "Provider %s — accuracy_vs_curated=%.3f, kappa=%.3f, F1-macro=%.3f (failed=%d)",
        provider, accuracy, kappa, f1_macro, failed,
    )
    return {
        "classifier": provider,
        "accuracy_vs_curated": round(accuracy, 4),
        "kappa_vs_curated": round(kappa, 4),
        "f1_macro_vs_curated": round(f1_macro, 4),
        "failed": failed,
        "n_samples": len(sample),
    }

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown(
    lr_result: dict[str, Any],
    groq_result: dict[str, Any],
    gemini_result: dict[str, Any],
    docs: list[dict[str, Any]],
    sample_size: int,
) -> None:
    """
    Write docs/sentiment_benchmark.md with results table and methodology note.
 
    Why a separate markdown file and not just console output:
    The README links to this file. Recruiters inspecting the repo can read
    the benchmark without running code. It also serves as a permanent record
    of the evaluation conditions (corpus size, date, model versions).
    """
    Path("docs").mkdir(exist_ok=True)
 
    # Class distribution for context
    dist = lr_result["class_distribution"]
    total = sum(dist.values())
    dist_str = ", ".join(
        f"{k}: {v} ({v/total*100:.1f}%)" for k, v in sorted(dist.items())
    )
 
    lines = [
        "# Sentiment Benchmark — Climate & Energy Intelligence System",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  ",
        f"**Corpus**: {len(docs)} articles from CURATED  ",
        f"**Class distribution**: {dist_str}  ",
        f"**LLM sample size**: {sample_size} articles re-classified live  ",
        "",
        "## Methodology",
        "",
        "Three classifiers are compared:",
        "",
        "1. **LogisticRegression** — sklearn baseline trained on 384-dim "
        "multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2). "
        "Evaluated with 5-fold stratified cross-validation. Labels: the "
        "sentiment assigned by the LLM pipeline (stored in CURATED).",
        "",
        "2. **Groq** (llama-3.1-8b-instant) — re-classifies a random sample "
        f"of {sample_size} articles. Metrics are computed against the original "
        "CURATED label.",
        "",
        "3. **Gemini** (gemini-2.0-flash) — same sample, same prompt, "
        "independent classification.",
        "",
        "> **Note on ground truth**: No human-annotated labels exist. "
        "LogisticRegression is evaluated via CV on LLM-assigned labels — "
        "this measures whether the embedding space encodes sentiment signal, "
        "not absolute correctness. Groq/Gemini metrics measure inter-model "
        "consistency, not accuracy against a gold standard.",
        "",
        "## Results",
        "",
        "### LogisticRegression (Full Corpus, 5-Fold CV)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {lr_result['accuracy']:.4f} |",
        f"| F1-macro | {lr_result['f1_macro']:.4f} |",
        f"| N samples | {lr_result['n_samples']} |",
        "",
        "**Per-class F1:**",
        "",
        "| Class | F1 |",
        "|-------|----|",
    ]
 
    for cls, f1 in sorted(lr_result["per_class_f1"].items()):
        lines.append(f"| {cls} | {f1:.4f} |")
 
    lines += [
        "",
        "### LLM Providers vs CURATED Labels (Sample)",
        "",
        "| Classifier | Accuracy vs CURATED | F1-macro vs CURATED | Cohen's Kappa | Failed | N |",
        "|------------|--------------------|--------------------|---------------|--------|---|",
    ]
 
    for result in [groq_result, gemini_result]:
        acc = f"{result['accuracy_vs_curated']:.4f}" if result["accuracy_vs_curated"] is not None else "N/A"
        f1 = f"{result['f1_macro_vs_curated']:.4f}" if result["f1_macro_vs_curated"] is not None else "N/A"
        kappa = f"{result['kappa_vs_curated']:.4f}" if result["kappa_vs_curated"] is not None else "N/A"
        lines.append(
            f"| {result['classifier']} | {acc} | {f1} | {kappa} | {result['failed']} | {result['n_samples']} |"
        )
 
    lines += [
        "",
        "## Interpretation",
        "",
        "- If **LogisticRegression F1-macro > 0.70**: the 384-dim embedding "
        "space already carries strong sentiment signal. The LLM adds "
        "nuanced reasoning (intensity, principal subject) beyond what a "
        "linear classifier can provide.",
        "",
        "- If **Groq/Gemini kappa > 0.60**: both providers are substantially "
        "consistent — the fallback mechanism does not degrade output quality.",
        "",
        "- If **Groq/Gemini kappa < 0.40**: the two models interpret sentiment "
        "differently. Consider adding more specific instructions to the system "
        "prompt to reduce variance.",
        "",
        "## Models Used",
        "",
        "| Component | Model |",
        "|-----------|-------|",
        "| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (384 dims) |",
        "| Groq | llama-3.1-8b-instant |",
        "| Gemini | gemini-2.0-flash |",
        "| Baseline | LogisticRegression (C=1.0, max_iter=1000) |",
    ]
 
    content = "\n".join(lines) + "\n"
    Path("docs/sentiment_benchmark.md").write_text(content, encoding="utf-8")
    logger.info("Benchmark report written to docs/sentiment_benchmark.md")
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_benchmark() -> None:
    db = get_db()

    docs = await _load_curated(db)
    if len(docs) < 10:
        raise RuntimeError(
            f"Only {len(docs)} articles with embeddings found in CURATED. "
            "Run the full pipeline first."
        )

    # --- Classifier 1: LogisticRegression ---
    logger.info("=== Evaluating LogisticRegression baseline ===")
    lr_result = _evaluate_logistic_regression(docs)
 
    # --- Sample for LLM re-classification ---
    random.seed(_RANDOM_SEED)
    sample_size = min(_SAMPLE_SIZE, len(docs))
    sample = random.sample(docs, sample_size)
    logger.info("Sampled %d articles for LLM re-classification.", sample_size)
 
    # --- Classifier 2: Groq ---
    logger.info("=== Re-classifying sample with Groq ===")
    groq_result = _evaluate_llm_provider(sample, "groq")
 
    # --- Classifier 3: Gemini ---
    logger.info("=== Re-classifying sample with Gemini ===")
    gemini_result = _evaluate_llm_provider(sample, "gemini")
 
    # Restore default provider
    os.environ["LLM_PROVIDER"] = os.getenv("LLM_PROVIDER", "groq")
 
    # --- Console report ---
    print("\n" + "=" * 60)
    print("SENTIMENT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nCorpus: {len(docs)} articles | LLM sample: {sample_size} articles")
    print(f"Class distribution: {lr_result['class_distribution']}")
    print()
    print("--- LogisticRegression (5-fold CV, full corpus) ---")
    print(f"  Accuracy : {lr_result['accuracy']:.4f}")
    print(f"  F1-macro : {lr_result['f1_macro']:.4f}")
    print(f"  Per-class: {lr_result['per_class_f1']}")
    print()
    for result in [groq_result, gemini_result]:
        print(f"--- {result['classifier']} (vs CURATED labels, sample) ---")
        print(f"  Accuracy : {result['accuracy_vs_curated']}")
        print(f"  F1-macro : {result['f1_macro_vs_curated']}")
        print(f"  Kappa    : {result['kappa_vs_curated']}")
        print(f"  Failed   : {result['failed']}/{result['n_samples']}")
        print()
    print("=" * 60)
 
    # --- Write markdown ---
    _write_markdown(lr_result, groq_result, gemini_result, docs, sample_size)
 
 
if __name__ == "__main__":
    asyncio.run(run_benchmark())
 