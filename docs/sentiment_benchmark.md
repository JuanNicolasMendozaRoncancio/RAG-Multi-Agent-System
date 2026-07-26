# Sentiment Benchmark — Climate & Energy Intelligence System

**Date**: 2026-07-26  
**Corpus**: 222 articles from CURATED  
**Class distribution**: negative: 148 (66.7%), neutral: 22 (9.9%), positive: 52 (23.4%)  
**LLM sample size**: 30 articles re-classified live  

## Methodology

Three classifiers are compared:

1. **LogisticRegression** — sklearn baseline trained on 384-dim multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2). Evaluated with 5-fold stratified cross-validation. Labels: the sentiment assigned by the LLM pipeline (stored in CURATED).

2. **Groq** (llama-3.1-8b-instant) — re-classifies a random sample of 30 articles. Metrics are computed against the original CURATED label.

3. **Gemini** (gemini-2.0-flash) — same sample, same prompt, independent classification.

> **Note on ground truth**: No human-annotated labels exist. LogisticRegression is evaluated via CV on LLM-assigned labels — this measures whether the embedding space encodes sentiment signal, not absolute correctness. Groq/Gemini metrics measure inter-model consistency, not accuracy against a gold standard.

## Results

### LogisticRegression (Full Corpus, 5-Fold CV)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7432 |
| F1-macro | 0.4793 |
| N samples | 222 |

**Per-class F1:**

| Class | F1 |
|-------|----|
| negative | 0.8483 |
| neutral | 0.0000 |
| positive | 0.5895 |

### LLM Providers vs CURATED Labels (Sample)

| Classifier | Accuracy vs CURATED | F1-macro vs CURATED | Cohen's Kappa | Failed | N |
|------------|--------------------|--------------------|---------------|--------|---|
| groq | 0.9000 | 0.8360 | 0.7857 | 0 | 30 |
| gemini | 0.9333 | 0.8333 | 0.8438 | 0 | 30 |

## Interpretation

- If **LogisticRegression F1-macro > 0.70**: the 384-dim embedding space already carries strong sentiment signal. The LLM adds nuanced reasoning (intensity, principal subject) beyond what a linear classifier can provide.

- If **Groq/Gemini kappa > 0.60**: both providers are substantially consistent — the fallback mechanism does not degrade output quality.

- If **Groq/Gemini kappa < 0.40**: the two models interpret sentiment differently. Consider adding more specific instructions to the system prompt to reduce variance.

## Models Used

| Component | Model |
|-----------|-------|
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 (384 dims) |
| Groq | llama-3.1-8b-instant |
| Gemini | gemini-2.0-flash |
| Baseline | LogisticRegression (C=1.0, max_iter=1000) |
