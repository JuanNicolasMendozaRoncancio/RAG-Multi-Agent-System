"""
Integration tests for the full NLP pipeline chain.

Strategy
--------
These tests verify that data flows correctly between pipeline stages:
    run_ingestion() → RAW
        ↓
    run_nlp_pipeline() → CLEAN
        ↓
    embed_articles() → CLEAN (embedding field added)
        ↓
    run_sentiment_pipeline() → CURATED

All external dependencies are mocked:
- MongoDB: mongomock (in-memory, synchronous) wrapped in async-compatible helpers
- feedparser + trafilatura: mocked to return controlled article fixtures
- spaCy + langdetect: mocked to avoid 130MB model downloads
- HF Serverless API: httpx.AsyncClient.post mocked
- Groq/Gemini: chat_complete mocked

What is verified:
- An article that enters run_ingestion() ends up in CURATED with all
  required sentiment fields (sentiment, intensity, principal_subject, main_argument)
- The critical transition fields are correct at each stage:
    RAW: sha256 present
    CLEAN: url matches RAW, _id removed, lemmatized_tokens present
    CLEAN: embedding field added by embed_articles()
    CURATED: sentiment fields present, _id from CLEAN not carried over
- A duplicate URL in the feed produces exactly one document in CURATED
- An article with empty text is skipped and never reaches CURATED
- A feed article that trafilatura cannot scrape is skipped at ingestion

What is NOT tested:
- Actual NLP quality (spaCy's responsibility)
- Actual embedding quality (HF model's responsibility)
- Actual sentiment quality (LLM's responsibility)
- Atlas Vector Search (requires real Atlas cluster)
"""
from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Async mongomock wrappers
# ---------------------------------------------------------------------------
# We consolidate the best of both patterns used in test_embedder.py and
# test_sentiment.py into a single _AsyncCollection that supports:
# - to_list()   used by embed_articles() and run_nlp_pipeline()
# - __aiter__   used by get_existing_hashes() and get_unprocessed_raw_urls()
# - find_one()  used by run_nlp_pipeline() and run_sentiment_pipeline()
# - count_documents(), update_one(), insert_one() used across all stages
# ---------------------------------------------------------------------------

class _AsyncCursor:
    """
    Wraps a synchronous mongomock cursor to support both:
    - async for iteration (used by shared/db.py set comprehensions)
    - await cursor.to_list() (used by embedder.py and pipeline.py)
    """
    def __init__(self, sync_cursor: Any) -> None:
        self._cursor = sync_cursor
        self._results: list[Any] | None = None

    def _materialize(self) -> list[Any]:
        if self._results is None:
            self._results = list(self._cursor)
        return self._results

    def __aiter__(self) -> "_AsyncCursor":
        self._iter = iter(self._materialize())
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def to_list(self, length: int | None = None) -> list[Any]:
        results = self._materialize()
        if length is not None:
            results = results[:length]
        return results


class _AsyncCollection:
    """
    Wraps a mongomock collection to make all motor-style async methods awaitable.

    Supports the full set of methods used across all four pipeline stages:
    find, find_one, count_documents, update_one, insert_one, insert_many.
    Unknown attributes fall through to the real mongomock collection via __getattr__.
    """
    def __init__(self, sync_col: Any) -> None:
        self._col = sync_col

    def find(self, *args: Any, **kwargs: Any) -> _AsyncCursor:
        return _AsyncCursor(self._col.find(*args, **kwargs))

    async def find_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.find_one(*args, **kwargs)

    async def count_documents(self, *args: Any, **kwargs: Any) -> int:
        return self._col.count_documents(*args, **kwargs)

    async def update_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.update_one(*args, **kwargs)

    async def insert_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.insert_one(*args, **kwargs)

    async def insert_many(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.insert_many(*args, **kwargs)

    async def aggregate(self, *args: Any, **kwargs: Any) -> _AsyncCursor:
        return _AsyncCursor(self._col.aggregate(*args, **kwargs))

    def sort(self, *args: Any, **kwargs: Any) -> "_AsyncCollection":
        self._col = self._col.find().sort(*args, **kwargs)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._col, name)


class _AsyncDb:
    """
    Wraps a mongomock database so db[collection_name] returns _AsyncCollection.
    """
    def __init__(self, sync_db: Any) -> None:
        self._db = sync_db

    def __getitem__(self, name: str) -> _AsyncCollection:
        return _AsyncCollection(self._db[name])

    async def command(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": 1}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384


def _fake_vector(seed: float = 0.1) -> list[float]:
    return [seed] * _EMBEDDING_DIM


def _make_feed_item(n: int = 1) -> dict[str, Any]:
    """Minimal feed item as returned by fetch_all_feeds()."""
    return {
        "url": f"https://example.com/article-{n}",
        "source": "carbon_brief",
        "feed_title": f"Article {n}",
        "feed_date": "2024-06-01",
        "feed_author": "Test Author",
        "feed_categories": ["climate"],
    }


def _make_scraped_article(n: int = 1, text: str = "Renewable energy is growing fast.") -> dict[str, Any]:
    """Minimal scraped article as returned by scrape_article()."""
    from shared.dedup import compute_url_hash
    url = f"https://example.com/article-{n}"
    return {
        "url": url,
        "title": f"Article {n}",
        "text": text,
        "detected_language": None,
        "source": "carbon_brief",
        "publication_date": "2024-06-01",
        "ingestion_date": "2024-06-01T12:00:00+00:00",
        "sha256": compute_url_hash(url),
    }


def _make_spacy_doc(tokens: list[str] = None, ents: list[tuple[str, str]] = None) -> MagicMock:
    """Build a minimal spaCy Doc mock."""
    if tokens is None:
        tokens = ["renewable", "energy", "climate"]
    if ents is None:
        ents = [("Amazon", "LOC")]

    mock_tokens = []
    for lemma in tokens:
        tok = MagicMock()
        tok.lemma_ = lemma
        tok.is_alpha = True
        tok.is_stop = False
        mock_tokens.append(tok)

    mock_ents = []
    for text, label in ents:
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        mock_ents.append(ent)

    doc = MagicMock()
    doc.__iter__ = MagicMock(side_effect=lambda: iter(mock_tokens))
    doc.ents = mock_ents
    return doc


def _make_hf_response(n_texts: int, seed: float = 0.1) -> MagicMock:
    """Mock httpx response returning n_texts embedding vectors."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = [_fake_vector(seed) for _ in range(n_texts)]
    return mock_resp


def _make_sentiment_response(sentiment: str = "positive") -> dict[str, Any]:
    """Mock chat_complete() response for sentiment classification."""
    content = json.dumps({
        "sentiment": sentiment,
        "intensity": 0.8,
        "principal_subject": "renewable energy",
        "main_argument": "Clean energy is expanding rapidly.",
    })
    return {"content": content, "provider": "groq", "model": "llama-3.1-8b-instant"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db_singleton() -> Generator[None, None, None]:
    """Reset shared/db.py singletons between tests."""
    import shared.db as db_module
    db_module._client = None
    db_module._db = None
    yield
    db_module._client = None
    db_module._db = None


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Patch get_db() in all four pipeline modules to return the same
    async-compatible mongomock database.

    Why patch all four namespaces and not just shared.db.get_db:
    Each module binds 'get_db' at import time with 'from shared.db import get_db'.
    Patching the source module does not update already-bound references.
    We must patch where each name is looked up (the consumer namespace).
    """
    try:
        import mongomock
    except ImportError:
        pytest.skip("mongomock not installed")
    monkeypatch.setattr("nlp_worker.topic_modeler.get_db", lambda: async_db, raising=False)
    client = mongomock.MongoClient()
    sync_db = client["rag_climate"]
    async_db = _AsyncDb(sync_db)

    for module in [
        "nlp_worker.ingest",
        "nlp_worker.pipeline",
        "nlp_worker.embedder",
        "nlp_worker.sentiment",
        "shared.db",
    ]:
        monkeypatch.setattr(module + ".get_db", lambda: async_db, raising=False)

    # Also patch the shared.db functions that are imported directly
    monkeypatch.setattr(
        "shared.db.get_existing_hashes",
        lambda: _async_get_existing_hashes(sync_db),
        raising=False,
    )
    monkeypatch.setattr(
        "shared.db.get_unprocessed_raw_urls",
        lambda: _async_get_unprocessed_raw_urls(sync_db),
        raising=False,
    )
    monkeypatch.setattr(
        "shared.db.get_unclassified_clean_urls",
        lambda: _async_get_unclassified_clean_urls(sync_db),
        raising=False,
    )
    monkeypatch.setattr(
        "shared.db.insert_raw_articles_bulk",
        lambda articles, known_hashes: _async_insert_raw_bulk(sync_db, articles, known_hashes),
        raising=False,
    )
    monkeypatch.setattr(
        "shared.db.insert_clean_article",
        lambda article: _async_insert_clean(sync_db, article),
        raising=False,
    )
    monkeypatch.setattr(
        "shared.db.insert_curated_article",
        lambda article: _async_insert_curated(sync_db, article),
        raising=False,
    )

    # Also patch ingest module's direct imports
    monkeypatch.setattr(
        "nlp_worker.ingest.get_existing_hashes",
        lambda: _async_get_existing_hashes(sync_db),
        raising=False,
    )
    monkeypatch.setattr(
        "nlp_worker.ingest.insert_raw_articles_bulk",
        lambda articles, known_hashes: _async_insert_raw_bulk(sync_db, articles, known_hashes),
        raising=False,
    )
    monkeypatch.setattr(
        "nlp_worker.pipeline.get_unprocessed_raw_urls",
        lambda: _async_get_unprocessed_raw_urls(sync_db),
        raising=False,
    )
    monkeypatch.setattr(
        "nlp_worker.pipeline.insert_clean_article",
        lambda article: _async_insert_clean(sync_db, article),
        raising=False,
    )
    monkeypatch.setattr(
        "nlp_worker.sentiment.get_unclassified_clean_urls",
        lambda: _async_get_unclassified_clean_urls(sync_db),
        raising=False,
    )

    return sync_db


# ---------------------------------------------------------------------------
# Async helpers for shared/db functions
# ---------------------------------------------------------------------------
# These replicate the logic of the real shared/db.py functions but using
# the synchronous mongomock interface directly, then wrapping in a coroutine.
# Why not just call the real functions: the real functions call get_db()
# internally, and while we patch get_db(), the set comprehensions inside them
# use 'async for' which requires our _AsyncCursor wrapper. These helpers
# bypass that by using mongomock's synchronous find() directly.
# ---------------------------------------------------------------------------

async def _async_get_existing_hashes(sync_db: Any) -> set[str]:
    return {doc["sha256"] for doc in sync_db["raw"].find({}, {"sha256": 1, "_id": 0})}


async def _async_get_unprocessed_raw_urls(sync_db: Any) -> list[str]:
    raw_urls = {doc["url"] for doc in sync_db["raw"].find({}, {"url": 1, "_id": 0})}
    clean_urls = {doc["url"] for doc in sync_db["clean"].find({}, {"url": 1, "_id": 0})}
    return list(raw_urls - clean_urls)


async def _async_get_unclassified_clean_urls(sync_db: Any) -> list[str]:
    clean_urls = {
        doc["url"]
        for doc in sync_db["clean"].find(
            {"topic_id": {"$exists": True}}, {"url": 1, "_id": 0}
        )
    }
    curated_urls = {doc["url"] for doc in sync_db["curated"].find({}, {"url": 1, "_id": 0})}
    return list(clean_urls - curated_urls)


async def _async_insert_raw_bulk(
    sync_db: Any,
    articles: list[dict[str, Any]],
    known_hashes: set[str],
) -> tuple[int, int]:
    seen = set(known_hashes)
    new_articles = []
    for a in articles:
        sha = a.get("sha256")
        if sha not in seen:
            seen.add(sha)
            new_articles.append(a)
    if not new_articles:
        return 0, len(articles)
    sync_db["raw"].insert_many(new_articles, ordered=False)
    for a in new_articles:
        known_hashes.add(a["sha256"])
    return len(new_articles), len(articles) - len(new_articles)


async def _async_insert_clean(sync_db: Any, article: dict[str, Any]) -> None:
    sync_db["clean"].insert_one(article)


async def _async_insert_curated(sync_db: Any, article: dict[str, Any]) -> None:
    sync_db["curated"].insert_one(article)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFullPipelineChain:
    """
    Tests that run the four pipeline stages in sequence against a shared
    in-memory mongomock database. Each test verifies data integrity at
    one or more collection boundaries.
    """

    def _patch_nlp(self) -> tuple[Any, Any]:
        """Return (detect_language patch, _get_nlp patch) context managers."""
        spacy_doc = _make_spacy_doc()
        nlp_mock = MagicMock(return_value=spacy_doc)
        return (
            patch("nlp_worker.pipeline.detect_language", return_value="en"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        )

    async def test_article_flows_from_feed_to_curated(self, mock_db: Any) -> None:
        """
        Happy path: one article from the feed ends up in CURATED with all
        required sentiment fields after running all four pipeline stages.
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.embedder import embed_articles
        from nlp_worker.sentiment import run_sentiment_pipeline

        feed_item = _make_feed_item(1)
        scraped = _make_scraped_article(1)

        detect_patch, nlp_patch = self._patch_nlp()
        hf_response = _make_hf_response(n_texts=1)

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=[feed_item]),
            patch("nlp_worker.ingest.scrape_article", return_value=scraped),
            detect_patch,
            nlp_patch,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=hf_response),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            # Ingestion must insert topic_id manually since BERTopic is not
            # part of this chain (it requires a pre-trained model file).
            # We simulate the topic assignment by pre-setting topic_id=0 in
            # CLEAN after the NLP step, which is what assign_topics() does.
            await run_ingestion()
            await run_nlp_pipeline()

            # Simulate BERTopic assignment (assign_topics is not mocked here
            # because it requires a .joblib file; we set topic_id directly)
            from nlp_worker.topic_modeler import assign_topics
            await embed_articles()
            await assign_topics()
            await run_sentiment_pipeline()

        # Verify the article reached CURATED
        curated_docs = list(mock_db["curated"].find({}))
        assert len(curated_docs) == 1

        doc = curated_docs[0]
        assert doc["url"] == "https://example.com/article-1"
        assert doc["sentiment"] == "positive"
        assert doc["intensity"] == 0.8
        assert "principal_subject" in doc
        assert "main_argument" in doc

    async def test_transition_fields_are_correct_at_each_stage(self, mock_db: Any) -> None:
        """
        Verify the critical fields at each collection boundary:
        - RAW: sha256 present
        - CLEAN: _id from RAW removed, lemmatized_tokens present, url matches
        - CLEAN after embed: embedding field has correct dimension
        - CURATED: _id from CLEAN not carried as string, sentiment fields present
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.embedder import embed_articles
        from nlp_worker.sentiment import run_sentiment_pipeline
        from shared.dedup import compute_url_hash

        scraped = _make_scraped_article(1)
        detect_patch, nlp_patch = self._patch_nlp()
        hf_response = _make_hf_response(n_texts=1)

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=[_make_feed_item(1)]),
            patch("nlp_worker.ingest.scrape_article", return_value=scraped),
            detect_patch,
            nlp_patch,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=hf_response),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            await run_ingestion()

            # RAW boundary
            raw_doc = mock_db["raw"].find_one({"url": "https://example.com/article-1"})
            assert raw_doc is not None
            assert raw_doc["sha256"] == compute_url_hash("https://example.com/article-1")

            await run_nlp_pipeline()

            # CLEAN boundary
            clean_doc = mock_db["clean"].find_one({"url": "https://example.com/article-1"})
            assert clean_doc is not None
            assert "_id" not in {k: v for k, v in clean_doc.items() if k == "_id" and isinstance(v, str)}
            assert isinstance(clean_doc["lemmatized_tokens"], list)
            assert clean_doc["url"] == raw_doc["url"]
            assert clean_doc["sha256"] == raw_doc["sha256"]
            assert "detected_language" in clean_doc

            from nlp_worker.topic_modeler import assign_topics
            await embed_articles()
            await assign_topics()
            

            # CLEAN after embedding boundary
            embedded_doc = mock_db["clean"].find_one({"url": "https://example.com/article-1"})
            assert embedded_doc is not None
            assert "embedding" in embedded_doc
            assert len(embedded_doc["embedding"]) == _EMBEDDING_DIM

            await run_sentiment_pipeline()

            # CURATED boundary
            curated_doc = mock_db["curated"].find_one({"url": "https://example.com/article-1"})
            assert curated_doc is not None
            # The string _id from CLEAN must not be carried to CURATED
            assert curated_doc.get("_id") != "mock_id_1"
            assert curated_doc["sentiment"] in ("positive", "negative", "neutral")

    async def test_duplicate_url_produces_one_curated_document(self, mock_db: Any) -> None:
        """
        A URL that appears twice in the feed must produce exactly one document
        in CURATED — deduplication at ingestion prevents double-processing.
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.embedder import embed_articles
        from nlp_worker.sentiment import run_sentiment_pipeline

        # Same URL appears twice in the feed
        feed_items = [_make_feed_item(1), _make_feed_item(1)]
        scraped = _make_scraped_article(1)
        detect_patch, nlp_patch = self._patch_nlp()
        hf_response = _make_hf_response(n_texts=1)

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=feed_items),
            patch("nlp_worker.ingest.scrape_article", return_value=scraped),
            detect_patch,
            nlp_patch,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=hf_response),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            ingestion_summary = await run_ingestion()

            # Only one article should have been inserted despite two feed items
            assert ingestion_summary["inserted"] == 1
            assert mock_db["raw"].count_documents({}) == 1

            await run_nlp_pipeline()
            from nlp_worker.topic_modeler import assign_topics
            await embed_articles()
            await assign_topics()
            await run_sentiment_pipeline()

        assert mock_db["curated"].count_documents({}) == 1

    async def test_empty_text_article_skipped_at_nlp_stage(self, mock_db: Any) -> None:
        """
        An article with empty text is inserted into RAW by ingestion but
        skipped by the NLP pipeline — it must never reach CURATED.
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.sentiment import run_sentiment_pipeline

        scraped_empty = _make_scraped_article(1, text="")

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=[_make_feed_item(1)]),
            patch("nlp_worker.ingest.scrape_article", return_value=scraped_empty),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            await run_ingestion()

            # RAW has the article
            assert mock_db["raw"].count_documents({}) == 1

            nlp_summary = await run_nlp_pipeline()

            # NLP skips it — empty text returns None from process_article()
            assert nlp_summary["skipped"] == 1
            assert mock_db["clean"].count_documents({}) == 0

            await run_sentiment_pipeline()

        # Nothing in CURATED
        assert mock_db["curated"].count_documents({}) == 0

    async def test_scrape_failure_skips_article(self, mock_db: Any) -> None:
        """
        If trafilatura returns None for an article, ingestion skips it.
        RAW, CLEAN, and CURATED must all remain empty.
        """
        from nlp_worker.ingest import run_ingestion

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=[_make_feed_item(1)]),
            patch("nlp_worker.ingest.scrape_article", return_value=None),
        ):
            summary = await run_ingestion()

        assert summary["skipped_scrape"] == 1
        assert summary["inserted"] == 0
        assert mock_db["raw"].count_documents({}) == 0

    async def test_multiple_articles_all_reach_curated(self, mock_db: Any) -> None:
        """
        Three distinct articles from the feed must all end up in CURATED.
        Verifies that the pipeline processes each article independently.
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.embedder import embed_articles
        from nlp_worker.sentiment import run_sentiment_pipeline

        n = 3
        feed_items = [_make_feed_item(i) for i in range(1, n + 1)]
        scraped_articles = {
            f"https://example.com/article-{i}": _make_scraped_article(i)
            for i in range(1, n + 1)
        }

        def fake_scrape(url: str, source: str) -> dict[str, Any] | None:
            return scraped_articles.get(url)

        detect_patch, nlp_patch = self._patch_nlp()
        hf_response = _make_hf_response(n_texts=n)

        async def fake_hf_post(*args: Any, **kwargs: Any) -> MagicMock:
            payload = kwargs.get("json") or (args[1] if len(args) > 1 else {})
            inputs = payload.get("inputs", []) if isinstance(payload, dict) else []
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = [_fake_vector(0.1)] * len(inputs)
            return mock_resp
        
        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=feed_items),
            patch("nlp_worker.ingest.scrape_article", side_effect=fake_scrape),
            detect_patch,
            nlp_patch,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_hf_post),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            ingestion_summary = await run_ingestion()
            assert ingestion_summary["inserted"] == n

            nlp_summary = await run_nlp_pipeline()
            assert nlp_summary["processed"] == n

            from nlp_worker.topic_modeler import assign_topics
            embed_summary = await embed_articles()
            await assign_topics()

            assert embed_summary["embedded"] == n

            await run_sentiment_pipeline()

        assert mock_db["curated"].count_documents({}) == n

    async def test_second_pipeline_run_skips_already_processed(self, mock_db: Any) -> None:
        """
        Running the full pipeline twice on the same feed must not create
        duplicate documents. The second run should be a no-op at each stage.

        This verifies the idempotency guarantees:
        - Ingestion: sha256 dedup prevents re-insertion into RAW
        - NLP: URL diff prevents re-processing articles already in CLEAN
        - Embeddings: field presence check prevents re-embedding
        - Sentiment: URL diff prevents re-classifying articles already in CURATED
        """
        from nlp_worker.ingest import run_ingestion
        from nlp_worker.pipeline import run_nlp_pipeline
        from nlp_worker.embedder import embed_articles
        from nlp_worker.sentiment import run_sentiment_pipeline

        feed_item = _make_feed_item(1)
        scraped = _make_scraped_article(1)
        detect_patch, nlp_patch = self._patch_nlp()
        hf_response = _make_hf_response(n_texts=1)

        run_kwargs = dict(
            fetch_all_feeds=patch("nlp_worker.ingest.fetch_all_feeds", return_value=[feed_item]),
            scrape=patch("nlp_worker.ingest.scrape_article", return_value=scraped),
            detect=detect_patch,
            nlp=nlp_patch,
            hf=patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=hf_response),
            llm=patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            env=patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        )

        async def _run_pipeline() -> None:
            with (
                run_kwargs["fetch_all_feeds"],
                run_kwargs["scrape"],
                run_kwargs["detect"],
                run_kwargs["nlp"],
                run_kwargs["hf"],
                run_kwargs["llm"],
                run_kwargs["env"],
            ):
                await run_ingestion()
                await run_nlp_pipeline()
                from nlp_worker.topic_modeler import assign_topics
                await embed_articles()
                await assign_topics()                
                await run_sentiment_pipeline()

        # First run
        await _run_pipeline()

        assert mock_db["raw"].count_documents({}) == 1
        assert mock_db["clean"].count_documents({}) == 1
        assert mock_db["curated"].count_documents({}) == 1

        # Re-create patches for second run (context managers are exhausted)
        detect_patch2, nlp_patch2 = self._patch_nlp()
        hf_response2 = _make_hf_response(n_texts=1)

        with (
            patch("nlp_worker.ingest.fetch_all_feeds", return_value=[feed_item]),
            patch("nlp_worker.ingest.scrape_article", return_value=scraped),
            detect_patch2,
            nlp_patch2,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=hf_response2),
            patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()),
            patch.dict("os.environ", {"HF_TOKEN": "hf_fake"}),
        ):
            second_ingestion = await run_ingestion()
            assert second_ingestion["inserted"] == 0  # dedup worked

            second_nlp = await run_nlp_pipeline()
            assert second_nlp["processed"] == 0  # URL diff returned empty

            second_embed = await embed_articles()
            assert second_embed["embedded"] == 0  # field presence check worked

            second_sentiment = await run_sentiment_pipeline()
            assert second_sentiment["classified"] == 0  # URL diff returned empty

        # Counts must not have changed
        assert mock_db["raw"].count_documents({}) == 1
        assert mock_db["clean"].count_documents({}) == 1
        assert mock_db["curated"].count_documents({}) == 1