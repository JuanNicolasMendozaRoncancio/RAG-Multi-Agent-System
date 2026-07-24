"""
Embedding module for the RAG pipeline.
 
Fetches articles from the CLEAN collection that do not yet have an embedding,
calls the HF Serverless Inference API in batches to generate 384-dimensional
multilingual vectors, and writes the result back into each CLEAN document.
 
Design decisions:
- httpx.AsyncClient instead of requests: the entire stack is async (FastAPI,
  motor). A blocking HTTP call inside a coroutine would freeze the event loop.
- Batch size 32: HF Serverless CPU inference has a ~30s timeout per request.
  Batches of 32 stay safely under that limit while reducing round-trips by ~30x
  compared to one request per article.
- Embedding input = lemmatized_tokens joined as string, not raw text: stopwords
  and punctuation have already been removed; the lemmatized tokens carry the
  semantic signal without noise from function words.
- Cache via field presence: if 'embedding' already exists in the CLEAN document,
  the article is skipped. No separate tracking collection needed — the field
  itself is the idempotency marker.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from shared.db import COL_CLEAN, get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{_HF_MODEL}"
_BATCH_SIZE = 32
_REQUEST_TIMEOUT = 60.0  # cold start

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_hf_token() -> str:
    """
    Read the HF token from the environment.
 
    Fails loudly at call time rather than at import time so that tests that
    mock the HTTP layer can run without a real token set.
 
    Raises
    ------
    RuntimeError
        If HF_TOKEN is not set.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Create a Read token at https://huggingface.co/settings/tokens "
            "and add it to your .env file: HF_TOKEN=hf_..."
        )
    return token

def _tokens_to_text(doc: dict[str, Any]) -> str | None:
    """
    Convert a CLEAN document's lemmatized_tokens list to a single string
    suitable for embedding.
 
    Why join with spaces rather than passing the list directly:
    The HF feature-extraction pipeline expects plain strings. Joining the
    tokens with a single space reconstructs a normalised, stopword-free
    pseudo-sentence that the model's tokenizer can process correctly.
 
    Returns None if the token list is missing or empty — these articles
    produce no useful embedding and must be skipped.
    """
    tokens: list[str] | None = doc.get("lemmatized_tokens")
    if not tokens:
        return None
    return " ".join(tokens)

async def _call_hf_api(
        texts: list[str],
        client: httpx.AsyncClient,
        token: str,
) -> list[list[float]] | None:
    """
    Send a batch of texts to the HF Serverless Inference API.
 
    The feature-extraction pipeline returns a nested structure. For
    sentence-level embeddings (not token-level), the model returns either:
      - list[list[float]]  — one vector per input string (the expected case)
      - list[list[list[float]]]  — token-level embeddings (must be pooled)
 
    paraphrase-multilingual-MiniLM-L12-v2 returns sentence-level embeddings
    directly, so no pooling is needed. The check below handles the edge case
    defensively.
 
    Parameters
    ----------
    texts:
        List of lemmatized text strings (one per article).
    client:
        Shared AsyncClient instance (keeps TCP connection alive across batches).
    token:
        HF Bearer token.
 
    Returns
    -------
    list[list[float]] | None
        One 384-dimensional vector per input string, or None if the API call
        failed. Returning None (not raising) lets the caller skip the batch
        and continue with the next one — a single API error should not abort
        the entire run.
    """
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    try:
        response = await client.post(_HF_API_URL, json = payload, headers= headers)
        response.raise_for_status()
        result: list[Any] = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "HF API returned HTTP %d for batch of %d texts: %s",
            exc.response.status_code,
            len(texts),
            exc.response.text[:200],
        )
        return None
    except httpx.TransportError as exc:
        logger.error("HF API network error for batch of %d texts: %s", len(texts), exc)
        return None

    if result and isinstance(result[0][0], list):
        pooled = []
        for token_embenddings in result:
            n = len(token_embenddings)
            dim = len(token_embenddings[0])
            mean_vec = [
                sum(token_embenddings[t][d] for t in range(n)) / n
                for d in range(dim)
            ]
            pooled.append(mean_vec)
        return pooled

    return result # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def embed_articles(batch_size: int = _BATCH_SIZE) -> dict[str, int]:
    """
    Embed all CLEAN articles that do not yet have an 'embedding' field.
 
    Workflow:
      1. Fetch all CLEAN documents where 'embedding' does not exist.
      2. Split into batches of `batch_size`.
      3. For each batch, call the HF API and write the resulting vectors
         back into MongoDB using update_one with $set.
      4. Return a summary dict with counts.
 
    Why update_one with $set instead of replacing the whole document:
    The CLEAN document may be modified by later pipeline steps (sentiment,
    BERTopic topic assignment). Writing only the 'embedding' field avoids
    overwriting fields that a concurrent step may have set.
 
    Parameters
    ----------
    batch_size:
        Number of articles per HF API request. Default 32 is the empirically
        safe limit for HF CPU inference within the 60s timeout.
 
    Returns
    -------
    dict[str, int]
        Keys: 'embedded', 'skipped_no_tokens', 'skipped_api_error', 'already_cached'
    """
    token = _get_hf_token()
    db = get_db()
    col = db[COL_CLEAN]

    cursor = col.find(
        {"embedding": {"$exists": False}},
        {"_id":1, "url": 1, "lemmatized_tokens":1},
    )
    docs: list[dict[str, Any]] = await cursor.to_list(length= None)

    already_cached_count = await col.count_documents({"embedding": {"$exists": True}})

    if not docs:
        logger.info("ALL CLEAN alrticles already have embeddings")
        return{
            "embedded": 0,
            "skipped_no_tokens": 0,
            "skipped_api_error": 0,
            "already_cached": already_cached_count, 
        }

    logger.info(
        "Found %d articles without embeddings (%d already cached).",
        len(docs),
        already_cached_count,
    )

    embedded = 0
    skipped_no_tokens = 0
    skipped_api_error = 0

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        for batch_start in range(0, len(docs), batch_size):
            batch = docs[batch_start: batch_start + batch_size]

            pairs: list[tuple[dict[str, Any], str]] = []
            for doc in batch:
                text = _tokens_to_text(doc)
                if text is None:
                    skipped_no_tokens += 1
                    logger.debug("Skipping %s — no lemmatized_tokens.", doc.get("url"))
                else:
                    pairs.append((doc, text))

            if not pairs:
                continue

            texts = [text for _, text in pairs]
            vectors = await _call_hf_api(texts, client, token)

            if vectors is None:
                skipped_api_error += len(pairs)
                logger.warning(
                    "Batch %d–%d failed — %d articles skipped.",
                    batch_start,
                    batch_start + len(batch) - 1,
                    len(pairs),
                )
                continue

            if len(vectors) != len(pairs):
                logger.error(
                    "HF API returned %d vectors for %d inputs — skipping batch.",
                    len(vectors),
                    len(pairs),
                )
                skipped_api_error += len(pairs)
                continue

            for (doc, _), vector in zip(pairs, vectors):
                await col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": vector}},
                )
                embedded += 1

            logger.info(
                "Batch %d–%d: embedded %d articles.",
                batch_start,
                min(batch_start + batch_size, len(docs)) - 1,
                len(pairs),
            )

    summary = {
        "embedded": embedded,
        "skipped_no_tokens": skipped_no_tokens,
        "skipped_api_error": skipped_api_error,
        "already_cached": already_cached_count,
    }
    logger.info("Embedding run complete: %s", summary)
    return summary
