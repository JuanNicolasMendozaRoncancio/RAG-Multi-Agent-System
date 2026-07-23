from __future__ import annotations

import os
from typing import Any

import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase


COL_RAW = "raw"
COL_CLEAN = "clean"
COL_CURATED = "curated"
COL_SUMMARIES = "summaries"


_client: motor.motor_asyncio.AsyncIOMotorClient | None = None # type: ignore[type-arg]
_db: AsyncIOMotorDatabase | None = None # type: ignore[type-arg]


def get_db() -> AsyncIOMotorDatabase: # type: ignore[type-arg]
    """
    Return the singleton database handle, creating the connection if needed.
 
    Why not async: motor.motor_asyncio.AsyncIOMotorClient() does not perform
    any I/O at construction time — it only sets up the connection pool lazily.
    The first actual network call happens when a query is issued. This means
    get_db() can be synchronous and called freely from both sync and async
    contexts without blocking.
 
    Raises
    ------
    RuntimeError
        If MONGODB_URI is not set in the environment.
    """ 
    global _client, _db

    if _db is not None:
        return _db

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set. "
            "Add it to your .env file: "
            "MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/rag_climate"
        )


    _client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    _db = _client.get_default_database()
    return _db

def _col(name: str) -> AsyncIOMotorCollection:  # type: ignore[type-arg]
    """
    Return a collection handle by name.
 
    Private helper — callers use the public functions below instead of
    accessing collections directly. This keeps the collection namespace
    centralised and makes it easy to add cross-cutting concerns (logging,
    metrics) in one place.
    """
    return get_db()[name]

# ---------------------------------------------------------------------------
# RAW collection — ingestion layer
# ---------------------------------------------------------------------------

async def get_existing_hashes() -> set[str]:
    """
    Return all SHA-256 hashes currently stored in the RAW collection.

    Returns
    -------
    set[str]
        SHA-256 hex strings of all URLs already in RAW.
    """
    cursor = _col(COL_RAW).find({}, {"sha256": 1, "_id": 0})
    return {doc["sha256"] async for doc in cursor}

async def insert_raw_article(article: dict[str, Any]) -> bool:
    """
    Insert one article into RAW if its SHA-256 hash is not already present.
 
    Parameters
    ----------
    article:
        Dict produced by nlp_worker/scraper.py. Must contain a 'sha256' key.
 
    Returns
    -------
    bool
        True if the article was inserted, False if it was a duplicate.
    """
    sha = article.get("sha256")
    if not sha:
        raise ValueError("article dict must contain a 'sha256' field")
 
    existing = await _col(COL_RAW).find_one({"sha256": sha}, {"_id": 1})
    if existing:
        return False
 
    await _col(COL_RAW).insert_one(article)
    return True


async def insert_raw_articles_bulk(
    articles: list[dict[str, Any]],
    known_hashes: set[str],
) -> tuple[int, int]:
    """
    Insert multiple articles into RAW, skipping duplicates.
 
    Why a bulk variant in addition to insert_raw_article:
 
    Parameters
    ----------
    articles:
        List of article dicts, each with a 'sha256' field.
    known_hashes:
        Set of SHA-256 hashes already in RAW. Updated in-place with newly
        inserted hashes so subsequent calls in the same run stay consistent.
 
    Returns
    -------
    tuple[int, int]
        (inserted_count, skipped_count)
    """
    new_articles = [a for a in articles if a.get("sha256") not in known_hashes]
 
    if not new_articles:
        return 0, len(articles)
 
    result = await _col(COL_RAW).insert_many(new_articles, ordered=False)
    inserted = len(result.inserted_ids)
 
    for article in new_articles:
        known_hashes.add(article["sha256"])
 
    skipped = len(articles) - inserted
    return inserted, skipped

# ---------------------------------------------------------------------------
# CLEAN collection — NLP-processed layer
# ---------------------------------------------------------------------------
 
async def get_unprocessed_raw_urls() -> list[str]:
    """
    Return URLs in RAW that have no corresponding document in CLEAN.
 
    Why compare by URL and not by sha256:
    
    This function is a stub — it returns an empty list until Step 4 
    is implemented and CLEAN starts being populated.
    """
    raw_urls_cursor = _col(COL_RAW).find({}, {"url": 1, "_id": 0})
    raw_urls = {doc["url"] async for doc in raw_urls_cursor}
 
    clean_urls_cursor = _col(COL_CLEAN).find({}, {"url": 1, "_id": 0})
    clean_urls = {doc["url"] async for doc in clean_urls_cursor}
 
    return list(raw_urls - clean_urls)
 
 
async def insert_clean_article(article: dict[str, Any]) -> None:
    """
    Insert one NLP-processed article into CLEAN.
 
    Stub — called by the NLP pipeline in Step 4.
    The article dict is expected to contain all RAW fields plus:
    - lemmatized_tokens: list[str]
    - detected_language: str  (ISO 639-1: 'es', 'en', 'fr')
    - entities: list[dict]  (NER output from spaCy)
    """
    await _col(COL_CLEAN).insert_one(article)

# ---------------------------------------------------------------------------
# SUMMARIES collection — agent output layer
# ---------------------------------------------------------------------------
 
async def insert_summary(summary: dict[str, Any]) -> None:
    """
    Insert one agent-generated summary into SUMMARIES.
 
    Stub — called by the Synthesis Agent in Step 11.
    The summary dict is expected to contain:
    - text: str
    - timestamp: datetime
    - llm_provider: str
    - period_days: int
    """
    await _col(COL_SUMMARIES).insert_one(summary)

# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------
 
async def ensure_indexes() -> None:
    """Create all indexes required by the system. Idempotent — safe to call on every startup.
 
    Why a dedicated function instead of creating indexes inline:
    - Index creation in MongoDB is idempotent: if the index already exists with
      the same definition, the call is a no-op. Running this function on every
      startup guarantees the indexes exist without manual database administration.
    - Grouping all index definitions here makes it trivial to audit the full
      index strategy of the system in one place.
 
    Index rationale:
    - RAW.sha256 unique: enforces deduplication at the database level even if
      application-level checks are bypassed (e.g., concurrent workers). Unique
      indexes in MongoDB also make equality queries on that field use the index
      automatically (O(log n) instead of O(n) collection scan).
    - RAW.fuente + RAW.fecha_ingesta: the dashboard filters articles by source
      and sorts by ingestion date. A compound index on (fuente, fecha_ingesta)
      satisfies both operations in a single index scan.
    - CLEAN.url unique: the NLP worker checks whether a RAW article already has
      a CLEAN counterpart by URL. A unique index makes this lookup O(log n) and
      prevents duplicate processing if two workers race.
    - CLEAN.embedding: the Vector Search index is created separately through the
      Atlas UI (it uses a different index type — knnVector — not supported by
      the standard createIndex API). This function creates only the standard
      B-tree indexes.
    """
    raw = _col(COL_RAW)
    clean = _col(COL_CLEAN)
 
    # RAW indexes
    await raw.create_index("sha256", unique=True, name="raw_sha256_unique")
    await raw.create_index(
        [("source", 1), ("ingestion_date", -1)],
        name="raw_source_ingestion_date",
    )
 
    await clean.create_index("url", unique=True, name="clean_url_unique")
    await clean.create_index("source", name="clean_source")
    await clean.create_index("detected_language", name="clean_detected_language")
 
    # SUMMARIES indexes
    await _col(COL_SUMMARIES).create_index(
        "timestamp", name="summaries_timestamp"
    )

    await _col(COL_CURATED).create_index(
        [("source", 1), ("ingestion_date", -1)],
        name="curated_source_ingestion_date",
    )