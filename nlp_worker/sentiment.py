"""
Sentiment classification for the RAG pipeline.

Reads articles from CLEAN that have a topic_id (BERTopic step complete)
and no corresponding document in CURATED, calls chat_complete() with
json_mode=True to extract a structured sentiment block, and writes the
enriched document to CURATED.

Cache strategy: if a URL already exists in CURATED, the article is skipped
unconditionally — chat_complete() is never called for it.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from shared.db import get_db, COL_CLEAN, COL_CURATED, get_unclassified_clean_urls, insert_curated_article
from shared.llm_client import chat_complete

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a sentiment analysis engine for climate and energy journalism.
Respond ONLY with a JSON object. No preamble, no explanation, no markdown fences.
The JSON must contain exactly these four keys:
- "sentiment": one of "positive", "negative", "neutral"
- "intensity": a float between 0.0 and 1.0
- "principal_subject": a short string (max 10 words) naming the main topic
- "main_argument": one sentence summarising the article's central argument
"""

def _build_user_prompt(article: dict[str, Any]) -> str:
    """
    Build the user-turn prompt from an article dict.

    Why title + text[:1500] and not the full text:
    Groq's context window is large, but sentiment can be reliably extracted
    from the headline and the first ~1500 characters (roughly the lede and
    first two paragraphs).
    """
    title = article.get("title") or ""
    text = article.get("text") or ""
    return f"Title: {title}\n\nText: {text[:1500]}"

# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------
def _parse_sentiment_response(raw: str) ->  dict[str, Any] | None:
    """
    Parse and validate the LLM JSON response.

    Returns None if the response is not valid JSON or is missing required keys.
    Returning None (not raising) lets the caller count the article as failed
    and continue.

    Why validate required keys explicitly:
    json_mode guarantees valid JSON syntax but not schema correctness.
    The model might omit a key or use a different name. Explicit validation
    here prevents downstream KeyError crashes in the agents.
    """
    required_keys = {"sentiment","intensity","principal_subject","main_argument"}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON response: %s", raw[:200])
        return None

    if not required_keys.issubset(parsed.keys()):
        missing = required_keys - parsed.keys()
        logger.error("LLM response missing keys %s: %s", missing, raw[:200])
        return None

    return parsed # type: ignore[return-value]

async def classify_article(article: dict[str, Any]) -> bool:
    """
    Classify the sentiment of a single CLEAN article and write it to CURATED.

    Parameters
    ----------
    article:
        A CLEAN document dict. Must contain 'url', 'title', 'text', 'topic_id'.

    Returns
    -------
    bool
        True if the article was successfully classified and inserted into CURATED.
        False if the LLM response was malformed or chat_complete raised.

    Why this function is sync on the LLM call but async overall:
    chat_complete() is synchronous (it uses the openai SDK's blocking client).
    The function is still declared async because it calls await insert_curated_article().
    Keeping it async lets run_sentiment_pipeline() await it without running it
    in a thread pool executor.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(article)},
    ]

    try:
        response = chat_complete(messages, json_mode=True)
    except RuntimeError as exc:
        logger.error("chat_complete failed for '%s': %s", article.get("url"), exc)
        return False

    raw_content = str(response.get("content", ""))
    sentiment_block = _parse_sentiment_response(raw_content)

    if sentiment_block is None:
        return False

    curated: dict[str, Any] = {
        **article,
        **sentiment_block,
    }
    curated.pop("_id", None)

    await insert_curated_article(curated)
    logger.info(
        "Classified '%s' → %s (%.2f) via %s",
        article.get("url"),
        sentiment_block["sentiment"],
        sentiment_block["intensity"],
        response.get("provider"),
    )
    return True

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
async def run_sentiment_pipeline() -> dict[str, int]:
    """
    Classify all CLEAN articles that have a topic_id but no CURATED entry.

    Returns
    -------
    dict[str, int]
        Keys: 'classified', 'skipped_cache', 'failed'

    Why fetch unclassified URLs first and then retrieve articles one by one:
    Same pattern as run_nlp_pipeline() — avoids holding the entire CLEAN
    collection in memory. The URL diff is O(n) but produces a small list;
    individual find_one() calls are O(log n) via the URL index.
    """
    db = get_db()
    clean_col = db[COL_CLEAN]

    unclassified_urls = await get_unclassified_clean_urls()
    skipped_cache = await db[COL_CURATED].count_documents({})

    logger.info("Sentiment pipeline: %d articles to classify.", len(unclassified_urls))

    classified = failed = 0

    for url in unclassified_urls:
        article = await clean_col.find_one({"url": url})
        if article is None:
            logger.warning("Article '%s' disappeared from CLEAN — skipping.", url)
            failed += 1
            continue

        success = await classify_article(article)
        if success:
            classified += 1
        else:
            failed += 1

    summary = {
        "classified": classified,
        "skipped_cache": skipped_cache,
        "failed": failed,
    }
    logger.info("Sentiment pipeline complete: %s", summary)
    return summary