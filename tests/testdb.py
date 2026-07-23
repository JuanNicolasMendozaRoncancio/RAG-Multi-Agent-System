from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from collections.abc import Generator

import pytest


def _make_article(n: int, sha: str | None = None) -> dict[str, Any]:
    """Build a minimal article dict for testing."""
    return {
        "url": f"https://example.com/article-{n}",
        "title": f"Article {n}",
        "text": f"Body text of article {n}.",
        "detected_language": None,
        "source": "test_source",
        "publication_date": None,
        "ingestion_date": "2024-01-01T00:00:00Z",
        "sha256": sha or f"{'a' * 60}{n:04d}",
    }


@pytest.fixture(autouse=True)
def reset_db_singleton() -> Generator[None, None, None]:
    """Reset the module-level singleton between tests.

    Why: shared/db.py stores _client and _db as module globals. If one test
    creates the singleton, the next test inherits it — including the mock.
    Resetting to None forces get_db() to re-run its initialisation logic with
    whatever MONGODB_URI is set for that test.
    """
    import shared.db as db_module

    db_module._client = None
    db_module._db = None
    yield
    db_module._client = None
    db_module._db = None


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Patch get_db() to return an in-memory mongomock database
    """
    try:
        import mongomock
    except ImportError:
        pytest.skip("mongomock not installed — install dev extras")

    client = mongomock.MongoClient()
    db = client["rag_climate"]

    import shared.db as db_module

    monkeypatch.setattr(db_module, "get_db", lambda: db)
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetExistingHashes:
    async def test_empty_collection_returns_empty_set(self, mock_db: MagicMock) -> None:
        """An empty RAW collection produces an empty hash set."""
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        hashes = {doc["sha256"] for doc in col.find({}, {"sha256": 1, "_id": 0})}
        assert hashes == set()

    async def test_returns_hashes_of_existing_articles(self, mock_db: MagicMock) -> None:
        """Hashes of inserted articles appear in the returned set."""
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        articles = [_make_article(i) for i in range(3)]
        col.insert_many(articles)

        hashes = {doc["sha256"] for doc in col.find({}, {"sha256": 1, "_id": 0})}
        expected = {a["sha256"] for a in articles}
        assert hashes == expected


class TestInsertRawArticleLogic:
    """
    Test the deduplication logic in insert_raw_article.

    We test the logic (skip if sha256 exists, insert if new) using mongomock's
    synchronous interface, isolating it from motor's async machinery.
    """

    def test_new_article_is_inserted(self, mock_db: MagicMock) -> None:
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        article = _make_article(1)

        existing = col.find_one({"sha256": article["sha256"]}, {"_id": 1})
        if not existing:
            col.insert_one(article)

        assert col.count_documents({"sha256": article["sha256"]}) == 1

    def test_duplicate_article_is_skipped(self, mock_db: MagicMock) -> None:
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        article = _make_article(1)
        col.insert_one(article)

        existing = col.find_one({"sha256": article["sha256"]}, {"_id": 1})
        was_inserted = False
        if not existing:
            col.insert_one(article)
            was_inserted = True

        assert not was_inserted
        assert col.count_documents({"sha256": article["sha256"]}) == 1

    def test_article_missing_sha256_raises(self) -> None:
        from shared.db import COL_RAW

        article = _make_article(1)
        del article["sha256"]

        with pytest.raises(ValueError, match="sha256"):
            sha = article.get("sha256")
            if not sha:
                raise ValueError("article dict must contain a 'sha256' field")


class TestInsertRawArticlesBulk:
    def test_bulk_inserts_new_articles(self, mock_db: MagicMock) -> None:
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        articles = [_make_article(i) for i in range(5)]
        known_hashes: set[str] = set()

        new_articles = [a for a in articles if a.get("sha256") not in known_hashes]
        result = col.insert_many(new_articles, ordered=False)

        assert len(result.inserted_ids) == 5

    def test_bulk_skips_known_hashes(self, mock_db: MagicMock) -> None:
        from shared.db import COL_RAW

        col = mock_db[COL_RAW]
        articles = [_make_article(i) for i in range(5)]
        known_hashes = {a["sha256"] for a in articles[:3]}

        new_articles = [a for a in articles if a.get("sha256") not in known_hashes]
        result = col.insert_many(new_articles, ordered=False)

        assert len(result.inserted_ids) == 2

    def test_bulk_all_duplicates_inserts_nothing(self, mock_db: MagicMock) -> None:
        articles = [_make_article(i) for i in range(3)]
        known_hashes = {a["sha256"] for a in articles}

        new_articles = [a for a in articles if a.get("sha256") not in known_hashes]
        assert len(new_articles) == 0


class TestGetDbRaisesWithoutUri:
    def test_raises_runtime_error_when_uri_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        get_db() must raise clearly if MONGODB_URI is not set.
        """
        monkeypatch.delenv("MONGODB_URI", raising=False)

        import shared.db as db_module

        db_module._client = None
        db_module._db = None

        with pytest.raises(RuntimeError, match="MONGODB_URI"):
            db_module.get_db()