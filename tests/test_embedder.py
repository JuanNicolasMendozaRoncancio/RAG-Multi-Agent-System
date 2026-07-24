"""
Unit tests for nlp_worker/embedder.py.

Strategy:
- MongoDB is replaced by mongomock (in-memory, synchronous interface that mirrors
  motor's API at the collection level). This lets us verify reads and writes without
  a real Atlas connection.
- httpx.AsyncClient.post is patched with AsyncMock so that HF API calls never hit
  the network. Each test controls exactly what the "API" returns.
- HF_TOKEN is set via monkeypatch.setenv — tests never need a real token.

What is NOT tested here:
- The actual quality of the embeddings (that is the model's responsibility).
- Atlas Vector Search recall (integration test, needs a real Atlas cluster).
- Network retry behaviour (httpx handles that internally).
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

_EMBEDDING_DIM = 384


def _fake_vector(seed: float = 0.1) -> list[float]:
    """Return a deterministic 384-dimensional unit-ish vector for testing."""
    return [seed] * _EMBEDDING_DIM


def _make_clean_doc(
    n: int,
    *,
    has_tokens: bool = True,
    has_embedding: bool = False,
) -> dict[str, Any]:
    """
    Build a minimal CLEAN document.

    Parameters
    ----------
    n:
        Document index — used to make url and _id unique.
    has_tokens:
        If False, omits the lemmatized_tokens field to test the skip-no-tokens path.
    has_embedding:
        If True, includes a pre-existing embedding to test the cache path.
    """
    doc: dict[str, Any] = {
        "_id": f"mock_id_{n}",
        "url": f"https://example.com/article-{n}",
        "detected_language": "en",
        "lemmatized_tokens": [f"token{n}", "energy", "climate"] if has_tokens else [],
    }
    if has_embedding:
        doc["embedding"] = _fake_vector(seed=0.99)
    return doc


def _make_hf_response(texts: list[str], seed: float = 0.1) -> MagicMock:
    """
    Build a mock httpx.Response that returns one 384-dim vector per input text.
    """
    vectors = [_fake_vector(seed) for _ in texts]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()  # no-op — no HTTP error
    mock_resp.json.return_value = vectors
    return mock_resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake HF token so _get_hf_token() does not raise."""
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken123")


@pytest.fixture(autouse=True)
def reset_db_singleton() -> Generator[None, None, None]:
    """
    Reset shared/db.py module-level singletons between tests.

    Why: get_db() caches _client and _db as module globals. Without this reset,
    a mock set in one test leaks into the next test's get_db() call.
    """
    import shared.db as db_module

    db_module._client = None
    db_module._db = None
    yield
    db_module._client = None
    db_module._db = None


class _AsyncCursor:
    """
    Wraps a synchronous mongomock cursor to expose an async to_list() method.

    Why needed: embed_articles() calls 'await cursor.to_list(length=None)',
    which is a motor (async) API. mongomock returns a plain synchronous cursor.
    This wrapper makes the cursor awaitable without changing embedder.py.
    """

    def __init__(self, sync_cursor: Any) -> None:
        self._cursor = sync_cursor

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        results = list(self._cursor)
        if length is not None:
            results = results[:length]
        return results


class _AsyncCollection:
    """
    Wraps a mongomock collection to make find/count_documents/update_one async.

    embed_articles() uses three async collection methods:
    - col.find(...) → cursor with async to_list()
    - await col.count_documents(...)
    - await col.update_one(...)

    mongomock implements all three synchronously. This wrapper makes them
    awaitable so embed_articles() can be tested without a real motor/Atlas
    connection. All other attribute accesses fall through to the real mongomock
    collection (e.g. insert_many used by tests to set up fixtures).
    """

    def __init__(self, sync_col: Any) -> None:
        self._col = sync_col

    def find(self, *args: Any, **kwargs: Any) -> _AsyncCursor:
        return _AsyncCursor(self._col.find(*args, **kwargs))

    async def count_documents(self, *args: Any, **kwargs: Any) -> int:
        return self._col.count_documents(*args, **kwargs)

    async def update_one(self, *args: Any, **kwargs: Any) -> Any:
        return self._col.update_one(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (insert_many, insert_one, etc.) to mongomock
        return getattr(self._col, name)


class _AsyncDb:
    """
    Wraps a mongomock database so that db[COL_CLEAN] returns an _AsyncCollection.

    embed_articles() accesses the collection via db[COL_CLEAN]. This wrapper
    intercepts that subscript and returns our async-compatible wrapper instead
    of the raw mongomock collection.
    """

    def __init__(self, sync_db: Any) -> None:
        self._db = sync_db

    def __getitem__(self, name: str) -> _AsyncCollection:
        return _AsyncCollection(self._db[name])


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Patch get_db() to return an async-compatible mongomock database.

    Why patch 'nlp_worker.embedder.get_db' and not 'shared.db.get_db':
    embedder.py imports get_db with 'from shared.db import get_db', which
    binds the name 'get_db' in the nlp_worker.embedder namespace at import
    time. Patching shared.db.get_db replaces the attribute in the source
    module, but embedder.py's local reference still points to the original
    function. Patching the name where it is *used* (nlp_worker.embedder.get_db)
    is the correct unittest.mock pattern: patch where it's looked up, not
    where it's defined.

    The fixture returns the raw mongomock db (not _AsyncDb) so that test
    setup code (col.insert_many, col.find, col.count_documents) uses the
    synchronous mongomock interface directly. Only embed_articles() goes
    through get_db() → _AsyncDb → _AsyncCollection.
    """
    try:
        import mongomock
    except ImportError:
        pytest.skip("mongomock not installed — install dev extras")

    client = mongomock.MongoClient()
    sync_db = client["rag_climate"]
    async_db = _AsyncDb(sync_db)

    monkeypatch.setattr("nlp_worker.embedder.get_db", lambda: async_db)
    # Return the raw sync db so test setup (insert_many etc.) works normally
    return sync_db


# ---------------------------------------------------------------------------
# Tests: _tokens_to_text
# ---------------------------------------------------------------------------


class TestTokensToText:
    """Unit tests for the internal helper that prepares embedding input."""

    def test_joins_tokens_with_space(self) -> None:
        from nlp_worker.embedder import _tokens_to_text

        doc = {"lemmatized_tokens": ["solar", "energy", "climate"]}
        assert _tokens_to_text(doc) == "solar energy climate"

    def test_returns_none_for_empty_list(self) -> None:
        from nlp_worker.embedder import _tokens_to_text

        doc = {"lemmatized_tokens": []}
        assert _tokens_to_text(doc) is None

    def test_returns_none_for_missing_field(self) -> None:
        from nlp_worker.embedder import _tokens_to_text

        doc: dict[str, Any] = {}
        assert _tokens_to_text(doc) is None

    def test_single_token(self) -> None:
        from nlp_worker.embedder import _tokens_to_text

        doc = {"lemmatized_tokens": ["deforestation"]}
        assert _tokens_to_text(doc) == "deforestation"


# ---------------------------------------------------------------------------
# Tests: _get_hf_token
# ---------------------------------------------------------------------------


class TestGetHfToken:
    def test_returns_token_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nlp_worker.embedder import _get_hf_token

        monkeypatch.setenv("HF_TOKEN", "hf_mytoken")
        assert _get_hf_token() == "hf_mytoken"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nlp_worker.embedder import _get_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            _get_hf_token()


# ---------------------------------------------------------------------------
# Tests: _call_hf_api
# ---------------------------------------------------------------------------


class TestCallHfApi:
    """
    Tests for the HTTP layer. httpx.AsyncClient.post is mocked so no real
    network calls are made.
    """

    async def test_returns_vectors_on_success(self) -> None:
        from nlp_worker.embedder import _call_hf_api

        texts = ["solar energy", "renewable power"]
        mock_response = _make_hf_response(texts, seed=0.2)

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with __import__("httpx").AsyncClient() as client:
                result = await _call_hf_api(texts, client, "hf_fake")

        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == _EMBEDDING_DIM

    async def test_returns_none_on_http_error(self) -> None:
        """A 429 rate-limit response must return None, not raise."""
        import httpx
        from nlp_worker.embedder import _call_hf_api

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited",
            request=MagicMock(),
            response=MagicMock(status_code=429, text="Rate limit exceeded"),
        )

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with httpx.AsyncClient() as client:
                result = await _call_hf_api(["some text"], client, "hf_fake")

        assert result is None

    async def test_returns_none_on_network_error(self) -> None:
        """A connection error must return None, not propagate the exception."""
        import httpx
        from nlp_worker.embedder import _call_hf_api

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            async with httpx.AsyncClient() as client:
                result = await _call_hf_api(["some text"], client, "hf_fake")

        assert result is None

    async def test_pools_token_level_embeddings(self) -> None:
        """
        If HF returns token-level embeddings (shape: n_texts × n_tokens × dim),
        the function must mean-pool them to produce sentence-level vectors.

        Why this can happen: some HF pipelines return per-token vectors rather
        than pooled sentence vectors depending on model configuration. The
        defensive pooling in _call_hf_api handles this transparently.
        """
        from nlp_worker.embedder import _call_hf_api

        # Simulate token-level output: 2 texts, 3 tokens each, 4-dim vectors
        # (dim=4 to keep the test fast; real dim is 384)
        token_level = [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9], [1.0, 1.1, 1.2, 1.3]],
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = token_level

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            async with __import__("httpx").AsyncClient() as client:
                result = await _call_hf_api(["text a", "text b"], client, "hf_fake")

        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 4  # pooled to sentence-level

        # Verify mean-pooling: (0.1+0.5+0.9)/3 = 0.5 for first text, first dim
        assert abs(result[0][0] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Tests: embed_articles (end-to-end with mocked HTTP + mongomock)
# ---------------------------------------------------------------------------


class TestEmbedArticles:
    """
    Integration-style tests for the public embed_articles() function.

    MongoDB is replaced by mongomock. HTTP calls are replaced by AsyncMock.
    This verifies the full orchestration logic: cache check, batching,
    DB writes, and summary counts.
    """

    async def test_embeds_new_articles(self, mock_db: MagicMock) -> None:
        """
        Happy path: 3 articles without embeddings → 3 vectors written to MongoDB.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        docs = [_make_clean_doc(i) for i in range(3)]
        col.insert_many(docs)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # HF returns one vector per input text
        mock_response.json.side_effect = lambda: [_fake_vector(0.1)] * 3

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            summary = await embed_articles()

        assert summary["embedded"] == 3
        assert summary["skipped_no_tokens"] == 0
        assert summary["skipped_api_error"] == 0
        assert summary["already_cached"] == 0

        # Verify embeddings were written to MongoDB
        updated = list(col.find({"embedding": {"$exists": True}}))
        assert len(updated) == 3
        assert len(updated[0]["embedding"]) == _EMBEDDING_DIM

    async def test_skips_already_cached_articles(self, mock_db: MagicMock) -> None:
        """
        Articles with an existing 'embedding' field must not be re-embedded.
        The HF API must not be called at all.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        # All 3 docs already have embeddings
        docs = [_make_clean_doc(i, has_embedding=True) for i in range(3)]
        col.insert_many(docs)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            summary = await embed_articles()
            mock_post.assert_not_called()

        assert summary["embedded"] == 0
        assert summary["already_cached"] == 3

    async def test_skips_articles_with_no_tokens(self, mock_db: MagicMock) -> None:
        """
        An article with empty lemmatized_tokens is uncountable as skipped_no_tokens.
        The run must not crash and must continue processing other articles.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        docs = [
            _make_clean_doc(0, has_tokens=False),  # should be skipped
            _make_clean_doc(1, has_tokens=True),   # should be embedded
        ]
        col.insert_many(docs)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [_fake_vector(0.1)]  # only 1 vector

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            summary = await embed_articles()

        assert summary["embedded"] == 1
        assert summary["skipped_no_tokens"] == 1

    async def test_counts_api_errors_as_skipped(self, mock_db: MagicMock) -> None:
        """
        When the HF API returns an error for a batch, those articles are counted
        as skipped_api_error and the run continues (does not raise).
        """
        import httpx
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        docs = [_make_clean_doc(i) for i in range(2)]
        col.insert_many(docs)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited",
            request=MagicMock(),
            response=MagicMock(status_code=429, text="Rate limit"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            summary = await embed_articles()

        assert summary["embedded"] == 0
        assert summary["skipped_api_error"] == 2
        # MongoDB must not have any embeddings written
        assert col.count_documents({"embedding": {"$exists": True}}) == 0

    async def test_empty_collection_returns_zeros(self, mock_db: MagicMock) -> None:
        """An empty CLEAN collection produces a zero summary without calling HF."""
        from nlp_worker.embedder import embed_articles

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            summary = await embed_articles()
            mock_post.assert_not_called()

        assert summary == {
            "embedded": 0,
            "skipped_no_tokens": 0,
            "skipped_api_error": 0,
            "already_cached": 0,
        }

    async def test_mismatch_in_batch_length_skips_batch(self, mock_db: MagicMock) -> None:
        """
        If HF returns fewer vectors than inputs (malformed response), the entire
        batch must be counted as skipped_api_error and no writes must happen.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        docs = [_make_clean_doc(i) for i in range(3)]
        col.insert_many(docs)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # Only 1 vector for 3 inputs — length mismatch
        mock_response.json.return_value = [_fake_vector(0.1)]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            summary = await embed_articles()

        assert summary["embedded"] == 0
        assert summary["skipped_api_error"] == 3

    async def test_mixed_cached_and_new_articles(self, mock_db: MagicMock) -> None:
        """
        2 articles already cached + 2 new → only 2 embedded, cache count = 2.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        cached = [_make_clean_doc(i, has_embedding=True) for i in range(2)]
        new_docs = [_make_clean_doc(i + 10) for i in range(2)]
        col.insert_many(cached + new_docs)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [_fake_vector(0.1), _fake_vector(0.2)]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            summary = await embed_articles()

        assert summary["embedded"] == 2
        assert summary["already_cached"] == 2
        assert summary["skipped_no_tokens"] == 0
        assert summary["skipped_api_error"] == 0

    async def test_batch_size_respected(self, mock_db: MagicMock) -> None:
        """
        With 5 articles and batch_size=2, HF API must be called 3 times
        (batches of 2, 2, 1).
        """
        from shared.db import COL_CLEAN
        from nlp_worker.embedder import embed_articles

        col = mock_db[COL_CLEAN]
        docs = [_make_clean_doc(i) for i in range(5)]
        col.insert_many(docs)

        call_count = 0

        async def fake_post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            inputs: list[str] = kwargs.get("json", {}).get("inputs", [])
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = [_fake_vector(0.1)] * len(inputs)
            return mock_response

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_post):
            summary = await embed_articles(batch_size=2)

        assert call_count == 3  # ceil(5 / 2) = 3 batches
        assert summary["embedded"] == 5