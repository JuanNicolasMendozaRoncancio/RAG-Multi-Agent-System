"""
Unit tests for nlp_worker/sentiment.py.

Strategy:
- MongoDB replaced by mongomock wrapped in async-compatible helpers
  (same pattern as test_embedder.py).
- chat_complete is patched at the nlp_worker.sentiment namespace so the
  openai SDK never makes real HTTP calls.
- Tests are grouped by unit: _parse_sentiment_response, classify_article,
  and run_sentiment_pipeline.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_doc(
    n: int,
    *,
    has_topic: bool = True,
) -> dict[str, Any]:
    """Minimal CLEAN document ready for sentiment classification."""
    doc: dict[str, Any] = {
        "_id": f"mock_id_{n}",
        "url": f"https://example.com/article-{n}",
        "title": f"Article {n} about climate",
        "text": "Renewable energy is expanding rapidly across Latin America.",
        "detected_language": "en",
        "source": "carbon_brief",
        "sha256": f"{'a' * 60}{n:04d}",
        "lemmatized_tokens": ["renewable", "energy", "expand"],
        "entities": [{"text": "Latin America", "label": "LOC"}],
        "embedding": [0.1] * 384,
    }
    if has_topic:
        doc["topic_id"] = 3
    return doc


def _make_sentiment_response(
    sentiment: str = "positive",
    intensity: float = 0.8,
    subject: str = "renewable energy",
    argument: str = "Solar capacity is growing fast.",
) -> dict[str, Any]:
    """Valid LLM response dict as returned by chat_complete()."""
    content = json.dumps({
        "sentiment": sentiment,
        "intensity": intensity,
        "principal_subject": subject,
        "main_argument": argument,
    })
    return {"content": content, "provider": "groq", "model": "llama3-8b-8192"}


# ---------------------------------------------------------------------------
# Async mongomock wrappers (same pattern as test_embedder.py)
# ---------------------------------------------------------------------------

class _AsyncCursor:
    def __init__(self, sync_cursor: Any) -> None:
        self._cursor = sync_cursor

    def __aiter__(self) -> "_AsyncCursor":
        self._iter = iter(list(self._cursor))
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _AsyncCollection:
    def __init__(self, sync_col: Any) -> None:
        self._col = sync_col

    def find(self, *args: Any, **kwargs: Any) -> _AsyncCursor:
        return _AsyncCursor(self._col.find(*args, **kwargs))

    async def find_one(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return self._col.find_one(*args, **kwargs)

    async def count_documents(self, *args: Any, **kwargs: Any) -> int:
        return self._col.count_documents(*args, **kwargs)

    async def insert_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.insert_one(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._col, name)


class _AsyncDb:
    def __init__(self, sync_db: Any) -> None:
        self._db = sync_db

    def __getitem__(self, name: str) -> _AsyncCollection:
        return _AsyncCollection(self._db[name])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db_singleton() -> Generator[None, None, None]:
    import shared.db as db_module
    db_module._client = None
    db_module._db = None
    yield
    db_module._client = None
    db_module._db = None


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Patch get_db() in the sentiment module to return an async-compatible
    mongomock database.

    Why patch 'nlp_worker.sentiment.get_db' and not 'shared.db.get_db':
    sentiment.py imports get_db with 'from shared.db import get_db', binding
    the name in the nlp_worker.sentiment namespace at import time. Patching
    the name where it is looked up is the correct unittest.mock pattern.
    """
    try:
        import mongomock
    except ImportError:
        pytest.skip("mongomock not installed")

    client = mongomock.MongoClient()
    sync_db = client["rag_climate"]
    async_db = _AsyncDb(sync_db)

    monkeypatch.setattr("nlp_worker.sentiment.get_db", lambda: async_db)
    monkeypatch.setattr("shared.db.get_db", lambda: async_db)
    return sync_db


# ---------------------------------------------------------------------------
# Tests: _parse_sentiment_response
# ---------------------------------------------------------------------------

class TestParseSentimentResponse:

    def test_valid_json_returns_dict(self) -> None:
        from nlp_worker.sentiment import _parse_sentiment_response

        raw = json.dumps({
            "sentiment": "positive",
            "intensity": 0.9,
            "principal_subject": "solar energy",
            "main_argument": "Solar is growing.",
        })
        result = _parse_sentiment_response(raw)
        assert result is not None
        assert result["sentiment"] == "positive"
        assert result["intensity"] == 0.9

    def test_invalid_json_returns_none(self) -> None:
        from nlp_worker.sentiment import _parse_sentiment_response

        result = _parse_sentiment_response("not json at all")
        assert result is None

    def test_missing_key_returns_none(self) -> None:
        from nlp_worker.sentiment import _parse_sentiment_response

        # Missing 'main_argument'
        raw = json.dumps({
            "sentiment": "neutral",
            "intensity": 0.5,
            "principal_subject": "climate policy",
        })
        result = _parse_sentiment_response(raw)
        assert result is None

    def test_extra_keys_are_allowed(self) -> None:
        """LLM adding unexpected keys must not cause a failure."""
        from nlp_worker.sentiment import _parse_sentiment_response

        raw = json.dumps({
            "sentiment": "negative",
            "intensity": 0.7,
            "principal_subject": "deforestation",
            "main_argument": "Deforestation accelerates.",
            "unexpected_key": "some value",
        })
        result = _parse_sentiment_response(raw)
        assert result is not None

    def test_empty_string_returns_none(self) -> None:
        from nlp_worker.sentiment import _parse_sentiment_response

        assert _parse_sentiment_response("") is None


# ---------------------------------------------------------------------------
# Tests: classify_article
# ---------------------------------------------------------------------------

class TestClassifyArticle:

    async def test_happy_path_inserts_into_curated(self, mock_db: Any) -> None:
        """Valid LLM response → document written to CURATED with sentiment fields."""
        from shared.db import COL_CURATED
        from nlp_worker.sentiment import classify_article

        article = _make_clean_doc(1)
        response = _make_sentiment_response()

        with patch("nlp_worker.sentiment.chat_complete", return_value=response):
            success = await classify_article(article)

        assert success is True
        col = mock_db[COL_CURATED]
        docs = list(col.find({"url": article["url"]}))
        assert len(docs) == 1
        assert docs[0]["sentiment"] == "positive"
        assert docs[0]["intensity"] == 0.8

    async def test_malformed_json_returns_false(self, mock_db: Any) -> None:
        """Malformed LLM response → classify_article returns False, nothing inserted."""
        from shared.db import COL_CURATED
        from nlp_worker.sentiment import classify_article

        article = _make_clean_doc(1)
        bad_response = {"content": "not json", "provider": "groq", "model": "llama3"}

        with patch("nlp_worker.sentiment.chat_complete", return_value=bad_response):
            success = await classify_article(article)

        assert success is False
        assert mock_db[COL_CURATED].count_documents({}) == 0

    async def test_chat_complete_raises_returns_false(self, mock_db: Any) -> None:
        """If chat_complete raises RuntimeError, classify_article returns False gracefully."""
        from nlp_worker.sentiment import classify_article

        article = _make_clean_doc(1)

        with patch(
            "nlp_worker.sentiment.chat_complete",
            side_effect=RuntimeError("All LLM providers failed"),
        ):
            success = await classify_article(article)

        assert success is False

    async def test_mongodb_id_not_carried_to_curated(self, mock_db: Any) -> None:
        """The RAW/CLEAN _id must be stripped before inserting into CURATED."""
        from shared.db import COL_CURATED
        from nlp_worker.sentiment import classify_article

        article = _make_clean_doc(1)
        assert "_id" in article

        with patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()):
            await classify_article(article)

        docs = list(mock_db[COL_CURATED].find({"url": article["url"]}, {"_id": 0}))
        # mongomock always adds _id on insert; what we verify is that the
        # original string _id from CLEAN was not carried over
        inserted = mock_db[COL_CURATED].find_one({"url": article["url"]})
        assert inserted is not None
        assert inserted.get("_id") != "mock_id_1"

    async def test_all_sentiment_fields_present_in_curated(self, mock_db: Any) -> None:
        from shared.db import COL_CURATED
        from nlp_worker.sentiment import classify_article

        article = _make_clean_doc(1)

        with patch("nlp_worker.sentiment.chat_complete", return_value=_make_sentiment_response()):
            await classify_article(article)

        doc = mock_db[COL_CURATED].find_one({"url": article["url"]})
        assert doc is not None
        for key in ("sentiment", "intensity", "principal_subject", "main_argument"):
            assert key in doc


# ---------------------------------------------------------------------------
# Tests: run_sentiment_pipeline
# ---------------------------------------------------------------------------

class TestRunSentimentPipeline:

    async def test_classifies_unclassified_articles(self, mock_db: Any) -> None:
        """Happy path: 2 unclassified articles → both classified."""
        from nlp_worker.sentiment import run_sentiment_pipeline

        with (
            patch(
                "nlp_worker.sentiment.get_unclassified_clean_urls",
                new=AsyncMock(return_value=[
                    "https://example.com/article-1",
                    "https://example.com/article-2",
                ]),
            ),
            patch(
                "nlp_worker.sentiment.classify_article",
                new=AsyncMock(return_value=True),
            ) as mock_classify,
        ):
            # Simulate find_one returning a doc for each URL
            from shared.db import COL_CLEAN
            mock_db[COL_CLEAN].insert_many([_make_clean_doc(1), _make_clean_doc(2)])

            summary = await run_sentiment_pipeline()

        assert summary["classified"] == 2
        assert summary["failed"] == 0
        assert mock_classify.call_count == 2

    async def test_cache_count_reflects_existing_curated(self, mock_db: Any) -> None:
        """Articles already in CURATED are counted as skipped_cache."""
        from shared.db import COL_CURATED
        from nlp_worker.sentiment import run_sentiment_pipeline

        # Pre-populate CURATED with 3 already-classified articles
        mock_db[COL_CURATED].insert_many([
            {"url": f"https://example.com/old-{i}", "sentiment": "neutral"}
            for i in range(3)
        ])

        with (
            patch(
                "nlp_worker.sentiment.get_unclassified_clean_urls",
                new=AsyncMock(return_value=[]),
            ),
        ):
            summary = await run_sentiment_pipeline()

        assert summary["skipped_cache"] == 3
        assert summary["classified"] == 0

    async def test_failed_classify_counted_correctly(self, mock_db: Any) -> None:
        """classify_article returning False → counted as failed, pipeline continues."""
        from nlp_worker.sentiment import run_sentiment_pipeline

        with (
            patch(
                "nlp_worker.sentiment.get_unclassified_clean_urls",
                new=AsyncMock(return_value=["https://example.com/article-1"]),
            ),
            patch(
                "nlp_worker.sentiment.classify_article",
                new=AsyncMock(return_value=False),
            ),
        ):
            from shared.db import COL_CLEAN
            mock_db[COL_CLEAN].insert_one(_make_clean_doc(1))
            summary = await run_sentiment_pipeline()

        assert summary["failed"] == 1
        assert summary["classified"] == 0

    async def test_empty_queue_returns_zeros(self, mock_db: Any) -> None:
        from nlp_worker.sentiment import run_sentiment_pipeline

        with patch(
            "nlp_worker.sentiment.get_unclassified_clean_urls",
            new=AsyncMock(return_value=[]),
        ):
            summary = await run_sentiment_pipeline()

        assert summary["classified"] == 0
        assert summary["failed"] == 0

    async def test_disappeared_article_counted_as_failed(self, mock_db: Any) -> None:
        """
        Race condition: URL in unclassified list but find_one returns None.
        Must be counted as failed, not crash.
        """
        from nlp_worker.sentiment import run_sentiment_pipeline

        with (
            patch(
                "nlp_worker.sentiment.get_unclassified_clean_urls",
                new=AsyncMock(return_value=["https://example.com/ghost"]),
            ),
            patch(
                "nlp_worker.sentiment.classify_article",
                new=AsyncMock(return_value=True),
            ) as mock_classify,
        ):
            # CLEAN is empty — find_one will return None
            summary = await run_sentiment_pipeline()

        assert summary["failed"] == 1
        mock_classify.assert_not_called()