"""
NLP pipeline: language detection, cleaning, lemmatization, and NER.
 
Consumes articles from the RAW MongoDB collection and writes enriched
documents to the CLEAN collection. The pipeline is idempotent: articles
already present in CLEAN (checked by URL) are skipped unconditionally.
 
Design overview
---------------
RAW article (url, title, text, source, ...)
    │
    ▼
langdetect  →  detected_language  (ISO 639-1: 'es' | 'en' | 'fr')
    │
    ▼
_clean_text()  →  normalised plain text  (strip HTML remnants, collapse whitespace)
    │
    ▼
spaCy model (language-specific)
    ├─ lemmatization  →  lemmatized_tokens: list[str]
    └─ NER            →  entities: list[{text, label}]
    │
    ▼
insert_clean_article()  →  CLEAN collection
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import spacy
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory


DetectorFactory.seed = 0

from shared.db import get_unprocessed_raw_urls, insert_clean_article, get_db, COL_RAW

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# spaCy model registry
# ---------------------------------------------------------------------------
# Why one model per language instead of a single multilingual model:
# spaCy's language-specific models (es_core_news_md, en_core_web_md,
# fr_core_news_md) are trained on corpora for that language and carry
# language-specific morphological rules. A multilingual model such as
# xx_ent_wiki_sm exists but only does NER, not lemmatization.
# Since we need both lemmatization AND NER, we need language-specific models.
#
# Why the _md (medium) variants instead of _sm (small) or _lg (large):
# _sm models omit word vectors entirely — they still do NER and
# lemmatization but with lower accuracy on ambiguous tokens. _lg models
# add 685k word vectors (~700MB), overkill for a pipeline that only needs
# lemmas and named entities. _md (20k–50k vectors, ~43MB each) is the
# correct trade-off: meaningfully better NER recall than _sm at a fraction
# of _lg's memory footprint.
#
# Models are loaded once at module import time (not on each article) because
# spaCy model loading takes ~0.5–2s per model. Loading inside process_article()
# would pay that cost for every single article.
_SPACY_MODELS: dict[str, str] = {
    "es":"es_core_news_md",
    "en":"en_core_web_md",
    "fr":"fr_core_news_md",
}

_nlp_cache: dict[str, spacy.language.Language] ={}

def _get_nlp(lang: str) -> spacy.language.Language:
    """
    Return the cached spaCy model for the given language code.
 
    Raises ValueError for unsupported languages so the pipeline can skip
    the article cleanly rather than crashing.
    """
    if lang not in _SPACY_MODELS:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {list(_SPACY_MODELS.keys())}"
        )
    if lang not in _nlp_cache:
        model_name = _SPACY_MODELS[lang]
        logger.info("Loading spaCy model '%s' (one-time cost)...", model_name)
        _nlp_cache[lang] = spacy.load(model_name)
        logger.info("Model '%s' loaded.", model_name)
    return _nlp_cache[lang]

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
 
# Regex compiled once at module level — compiling a regex is O(n) in pattern
# length but O(1) in usage; compiling inside the function would pay the
# compilation cost on every article call.
_WITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    """
    Normalise raw article text for NLP processing.
 
    Steps (in order):
    1. Strip residual HTML tags — trafilatura removes most boilerplate but
       some sources include inline <em>, <strong>, or <a> tags in the text
       layer.
    2. Collapse all whitespace sequences (spaces, tabs, newlines, non-breaking
       spaces) to a single space. This matters for spaCy's tokeniser: multiple
       spaces confuse sentence boundary detection in some models.
    3. Strip leading/trailing whitespace.
 
    Why not use BeautifulSoup here: BS4 parses a full HTML document. The
    text at this stage is already plain text with occasional inline tags —
    a simple regex is faster and avoids an extra dependency.
    """
    text = _HTML_TAG_RE.sub("", text)
    text = _WITESPACE_RE.sub(" ", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
 
def detect_language(text: str) -> str | None:
    """
    Detect the language of the article text.
 
    Returns an ISO 639-1 code ('es', 'en', 'fr') or None if detection
    fails or the detected language is not in the supported set.
 
    Why langdetect over spaCy's built-in language detection:
    spaCy does not include a language detector in its standard pipeline
    components — it expects you to know the language before loading a model.
    langdetect is the standard Python library for this task: it uses
    character n-gram profiles derived from Wikipedia and achieves >99%
    accuracy for the three languages in scope (ES/EN/FR) on article-length
    text. It is also significantly lighter than loading a full transformer
    for detection.
 
    Why run detection before selecting the spaCy model:
    The spaCy model must be selected based on language. There is no
    'universal' pipeline that lemmatises correctly across ES/EN/FR — they
    have fundamentally different morphological rules.
 
    Why return None instead of raising on unsupported language:
    The feed may occasionally pick up articles in other languages (PT, IT,
    DE) if a source publishes multilingual content. Returning None lets the
    pipeline skip these articles silently rather than crashing.
    """
    try:
        lang = detect(text[:2000])
    except LangDetectException:
        return None
 
    supported = set(_SPACY_MODELS.keys())
    return lang if lang in supported else None

# ---------------------------------------------------------------------------
# Core NLP processing
# ---------------------------------------------------------------------------
def process_article(raw_article: dict[str, Any]) -> dict[str, Any] | None:
    """
    Run the full NLP pipeline on a single raw article dict.
 
    Returns the enriched CLEAN article dict, or None if the article should
    be skipped (empty text, unsupported language, detection failure).
 
    The returned dict contains all RAW fields plus:
    - detected_language: str   ISO 639-1 code
    - lemmatized_tokens: list[str]
    - entities: list[dict]     each dict has 'text' and 'label' keys
 
    Why we pass the full raw_article dict and return the enriched version
    rather than only the NLP output: the CLEAN collection is a superset of
    RAW — downstream consumers (BERTopic, embeddings, agents) need the
    original fields (url, title, source, publication_date) alongside the
    NLP enrichment. Merging here keeps the pipeline a pure function easy
    to test.
    """
    text: str = raw_article.get("text") or ""
    if not text.strip():
        logger.warning("Skipping article '%s': empty text.", raw_article.get("url"))
        return None

    cleanned = _clean_text(text)

    lang = detect_language(cleanned)
    if lang is None:
        logger.warning(
            "Skipping article '%s': unsupported or undetectable language.",
            raw_article.get("url"),
        )
        return None

    try:
        nlp = _get_nlp(lang)
    except ValueError as exc:
        logger.warning("Skipping article '%s': %s", raw_article.get("url"), exc)
        return None

    doc = nlp(cleanned)
    # Lemmatization filter rationale:
    # - token.is_alpha: exclude punctuation, numbers, and symbols. These add
    #   noise to topic modeling and have no semantic value as lemmas.
    # - not token.is_stop: stopwords ('el', 'the', 'le') are the most
    #   frequent tokens but carry no discriminative information for BERTopic
    #   or for the /rag/search endpoint. Removing them reduces the token list
    #   size by ~40% without losing signal.
    # - token.lemma_.strip(): a few spaCy models produce lemmas with leading
    #   whitespace for tokens at sentence boundaries; strip() is defensive.
    lemmatized_tokens: list[str] = [
        token.lemma_.strip().lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 1
    ]

    # NER output format: list of dicts with 'text' and 'label'.
    # Why dicts instead of tuples: MongoDB stores BSON — tuples serialize
    # as arrays, which loses the key names and makes querying by label
    # (e.g. find all articles mentioning an ORG) impossible without positional
    # array indexing. Dicts serialize as BSON subdocuments with named fields,
    # enabling queries like {"entities.label": "ORG"}.
    entities: list[dict[str, str]] = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    # Build CLEAN document: all RAW fields + NLP enrichment.
    # We update detected_language here even if the RAW document already had
    # a value (scraper.py sets it to None). This is the single source of
    # truth for language — langdetect on the full text is more reliable than
    # any feed-level language hint.
    clean_article: dict[str, Any] = {
        **raw_article,
        "detected_language":lang,
        "lemmatized_tokens": lemmatized_tokens,
        "entities": entities,
    }

    # Remove MongoDB's internal _id field if present — insert_clean_article
    # will get a new _id on insert. Carrying the RAW _id would cause a
    # duplicate key error on the CLEAN collection's _id index.
    clean_article.pop("_id", None)

    return clean_article

async def run_nlp_pipeline() -> dict[str, int]:
    """
    Process all RAW articles that have no CLEAN counterpart.
 
    Returns a summary dict: {'processed': N, 'skipped': N, 'failed': N}
 
    Why async: this function is called both from the Prefect nocturnal flow
    (which runs in an async context) and from the FastAPI SSE endpoint
    (also async). Making it async allows both callers to await it without
    running it in a thread pool executor.
 
    Why fetch all unprocessed URLs first and then retrieve articles one by
    one, rather than streaming a cursor: get_unprocessed_raw_urls() returns
    the diff between RAW and CLEAN URL sets, which requires materialising
    both sets in memory. This is already done in shared/db.py. Fetching
    articles individually afterwards keeps memory usage bounded — we never
    hold the entire RAW collection in memory at once.
    """
    db = get_db()
    raw_col = db[COL_RAW]

    unprocessed_urls = await get_unprocessed_raw_urls()
    logger.info("NLP pipeline: %d articles to process.", len(unprocessed_urls))

    processed = skipped = failed = 0

    for url in unprocessed_urls:
        raw_article = await raw_col.find_one({"url":url})
        if raw_article is None:
            logger.warning("Article '%s' disappeared from RAW — skipping.", url)
            skipped += 1
            continue

        try:
            clean_article = process_article(raw_article)
        except Exception as exc:
            logger.error("NLP error on '%s': %s", url, exc, exc_info=True)
            failed += 1
            continue

        if clean_article is None:
            skipped += 1
            continue

        try:
            await insert_clean_article(clean_article)
            processed += 1
            logger.debug("Processed '%s' (lang=%s).", url, clean_article["detected_language"])
        except Exception as exc:
            logger.error("DB write error on '%s': %s", url, exc, exc_info=True)
            failed += 1

    summary = {"processed": processed, "skipped": skipped, "failed": failed}
    logger.info("NLP pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Entrypoint (called by Docker CMD: python -m nlp_worker.pipeline)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_nlp_pipeline())