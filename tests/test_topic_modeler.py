"""
Unit tests for nlp_worker/topic_modeler.py.

Strategy
--------
BERTopic, UMAP, and HDBSCAN are slow to instantiate and require large models.
We never run real BERTopic training in CI. Instead:

- _load_embeddings_from_db: tested with mongomock (real MongoDB logic, no ML).
- _build_bertopic_model: tested by inspecting the returned object's attributes
  (instantiation is fast; we only verify config, not training).
- train_topic_model: BERTopic.fit_transform() and joblib.dump() are mocked.
  We verify the orchestration logic: min_articles guard, DB writes, summary
  shape — not the clustering algorithm itself (that is BERTopic's responsibility).
- assign_topics: joblib.load() and BERTopic.transform() are mocked.
  We verify: FileNotFoundError when model missing, empty-collection early return,
  DB writes per topic assignment, noise vs assigned counts.

What is NOT tested here:
- The actual quality of BERTopic topics (that is BERTopic's own test suite).
- UMAP dimensionality reduction correctness.
- HDBSCAN cluster stability.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384


def _fake_embedding(seed: float = 0.1) -> list[float]:
    """Return a deterministic 384-dim vector. All values equal seed."""
    return [seed] * _EMBEDDING_DIM


def _make_clean_doc(
    n: int,
    *,
    has_embedding: bool = True,
    has_topic: bool = False,
) -> dict[str, Any]:
    """
    Build a minimal CLEAN document dict.

    Parameters
    ----------
    n:
        Index — makes url unique.
    has_embedding:
        If False, omits the embedding field (simulates un-embedded article).
    has_topic:
        If True, adds a topic_id field (simulates already-assigned article).
    """
    doc: dict[str, Any] = {
        "_id": f"mock_id_{n}",
        "url": f"https://example.com/article-{n}",
        "detected_language": "en",
        "lemmatized_tokens": ["solar", "energy", "climate"],
    }
    if has_embedding:
        doc["embedding"] = _fake_embedding(seed=float(n) * 0.01 + 0.01)
    if has_topic:
        doc["topic_id"] = 0
    return doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_db_singleton() -> Any:
    """
    Reset shared/db.py module-level singletons between tests.

    Why: get_db() caches _client and _db as module globals. Without reset,
    a mock set in one test leaks into the next.
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
    Patch get_db() in topic_modeler to return a real mongomock database.

    Why patch 'nlp_worker.topic_modeler.get_db' and not 'shared.db.get_db':
    topic_modeler.py imports get_db with 'from shared.db import get_db', which
    binds the name in the nlp_worker.topic_modeler namespace at import time.
    Patching the source module would not update the already-bound reference.
    We must patch where the name is *used*, not where it is *defined*.
    """
    try:
        import mongomock
    except ImportError:
        pytest.skip("mongomock not installed — install dev extras")

    client = mongomock.MongoClient()
    db = client["rag_climate"]

    monkeypatch.setattr("nlp_worker.topic_modeler.get_db", lambda: db)
    return db


# ---------------------------------------------------------------------------
# Tests: _load_embeddings_from_db
# ---------------------------------------------------------------------------


class TestLoadEmbeddingsFromDb:
    """
    Tests for the internal DB reader. Uses real mongomock — no ML mocking needed.
    """

    def test_returns_empty_arrays_for_empty_collection(self, mock_db: MagicMock) -> None:
        """Empty CLEAN collection → empty urls list and (0, 384) array."""
        from nlp_worker.topic_modeler import _load_embeddings_from_db

        urls, embeddings = _load_embeddings_from_db(mock_db)

        assert urls == []
        assert embeddings.shape == (0, 384)

    def test_returns_urls_and_embeddings_for_docs_with_embedding(
        self, mock_db: MagicMock
    ) -> None:
        """Documents with embeddings are returned in correct shape."""
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import _load_embeddings_from_db

        docs = [_make_clean_doc(i) for i in range(5)]
        mock_db[COL_CLEAN].insert_many(docs)

        urls, embeddings = _load_embeddings_from_db(mock_db)

        assert len(urls) == 5
        assert embeddings.shape == (5, _EMBEDDING_DIM)
        assert embeddings.dtype == np.float32

    def test_skips_docs_without_embedding(self, mock_db: MagicMock) -> None:
        """
        Documents without an 'embedding' field must not appear in results.
        The query uses {"embedding": {"$exists": True}} — docs missing the
        field are invisible to the loader.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import _load_embeddings_from_db

        with_emb = [_make_clean_doc(i, has_embedding=True) for i in range(3)]
        without_emb = [_make_clean_doc(i + 10, has_embedding=False) for i in range(2)]
        mock_db[COL_CLEAN].insert_many(with_emb + without_emb)

        urls, embeddings = _load_embeddings_from_db(mock_db)

        assert len(urls) == 3
        assert embeddings.shape == (3, _EMBEDDING_DIM)

    def test_url_and_embedding_order_is_consistent(self, mock_db: MagicMock) -> None:
        """
        urls[i] must correspond to embeddings[i]. Order consistency is critical
        because train_topic_model() zips urls and topics after fit_transform().
        A mismatch would write the wrong topic_id to MongoDB.

        We verify: the embedding stored for each url matches what we inserted.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import _load_embeddings_from_db

        docs = [_make_clean_doc(i) for i in range(4)]
        mock_db[COL_CLEAN].insert_many(docs)

        urls, embeddings = _load_embeddings_from_db(mock_db)

        for i, url in enumerate(urls):
            # Retrieve what was stored for this url
            stored = mock_db[COL_CLEAN].find_one({"url": url})
            assert stored is not None
            expected = np.array(stored["embedding"], dtype=np.float32)
            np.testing.assert_array_equal(embeddings[i], expected)


# ---------------------------------------------------------------------------
# Tests: _build_bertopic_model
# ---------------------------------------------------------------------------


class TestBuildBertopicModel:
    """
    Tests for the model factory. We instantiate the model (fast, no training)
    and inspect its configuration attributes.
    """

    def test_returns_bertopic_instance(self) -> None:
        from bertopic import BERTopic
        from nlp_worker.topic_modeler import _build_bertopic_model

        model = _build_bertopic_model()
        assert isinstance(model, BERTopic)

    def test_umap_has_correct_params(self) -> None:
        """UMAP must use n_components=5 and metric='cosine' (see module docstring)."""
        from nlp_worker.topic_modeler import _build_bertopic_model

        model = _build_bertopic_model()
        assert model.umap_model.n_components == 5
        assert model.umap_model.metric == "cosine"
        assert model.umap_model.random_state == 42

    def test_hdbscan_has_correct_params(self) -> None:
        """HDBSCAN must have prediction_data=True to support approximate_predict."""
        from nlp_worker.topic_modeler import _build_bertopic_model

        model = _build_bertopic_model()
        assert model.hdbscan_model.min_cluster_size == 10
        assert model.hdbscan_model.prediction_data is True
        assert model.hdbscan_model.metric == "euclidean"

    def test_embedding_model_is_none(self) -> None:
        """
        embedding_model must be None — we supply pre-computed embeddings.
        If this were not None, BERTopic would try to download a
        sentence-transformer on fit_transform(), hitting the network.
        """
        from nlp_worker.topic_modeler import _build_bertopic_model

        model = _build_bertopic_model()
        # BERTopic stores the raw value as self.embedding_model before
        # select_backend() is called (which happens at fit time).
        assert model.embedding_model is None

    def test_language_is_multilingual(self) -> None:
        """
        language='multilingual' disables English-only stopword filtering
        in the c-TF-IDF vectorizer. Required for ES/EN/FR corpus.
        """
        from nlp_worker.topic_modeler import _build_bertopic_model

        model = _build_bertopic_model()
        assert model.language == "multilingual"


# ---------------------------------------------------------------------------
# Tests: train_topic_model
# ---------------------------------------------------------------------------


class TestTrainTopicModel:
    """
    Tests for the training function.

    BERTopic.fit_transform() and joblib.dump() are mocked — we test the
    orchestration logic, not the clustering algorithm.
    """

    def _make_mock_bertopic(self, topics: list[int]) -> MagicMock:
        """
        Return a MagicMock that behaves like a fitted BERTopic model.

        fit_transform() returns (topics, probabilities).
        get_topic() returns [(word, score), ...] — we return two fake words.
        """
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = (topics, [0.9] * len(topics))
        # get_topic returns False for -1 (noise), word list for valid topics
        mock_model.get_topic.side_effect = lambda tid: (
            False if tid == -1 else [("solar", 0.8), ("energy", 0.7), ("climate", 0.6)]
        )
        return mock_model

    def test_raises_value_error_when_too_few_articles(
        self, mock_db: MagicMock
    ) -> None:
        """
        Training on fewer than min_articles must raise ValueError before
        calling BERTopic — unreliable clusters form below that threshold.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        # Insert only 5 articles (below default min_articles=20)
        docs = [_make_clean_doc(i) for i in range(5)]
        mock_db[COL_CLEAN].insert_many(docs)

        with pytest.raises(ValueError, match="at least 20 articles"):
            train_topic_model(min_articles=20)

    def test_raises_value_error_with_custom_min_articles(
        self, mock_db: MagicMock
    ) -> None:
        """min_articles parameter is respected when overridden."""
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        docs = [_make_clean_doc(i) for i in range(3)]
        mock_db[COL_CLEAN].insert_many(docs)

        with pytest.raises(ValueError, match="at least 5 articles"):
            train_topic_model(min_articles=5)

    def test_calls_fit_transform_with_numpy_array(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        fit_transform() must receive a numpy array, not a list.
        BERTopic's internal validation raises if documents are not strings
        but we pass embeddings — so we skip that check by patching
        _build_bertopic_model to return our mock.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        n = 25
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        topics_output = [0] * 20 + [-1] * 5
        mock_model = self._make_mock_bertopic(topics_output)
        model_path = str(tmp_path / "model.joblib")

        with (
            patch("nlp_worker.topic_modeler._build_bertopic_model", return_value=mock_model),
            patch("nlp_worker.topic_modeler.joblib.dump"),
        ):
            train_topic_model(model_path=model_path, min_articles=20)

        # Verify fit_transform was called and received a numpy array
        call_args = mock_model.fit_transform.call_args
        passed_embeddings = call_args[0][0]   # first positional argument
        assert isinstance(passed_embeddings, np.ndarray)
        assert passed_embeddings.shape == (n, _EMBEDDING_DIM)

    def test_writes_topic_ids_to_mongodb(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        After fit_transform(), each article must have its topic_id written
        to MongoDB. We verify by reading back the updated documents.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        n = 25
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        # First 20 articles get topic 0, last 5 are noise (-1)
        topics_output = [0] * 20 + [-1] * 5
        mock_model = self._make_mock_bertopic(topics_output)
        model_path = str(tmp_path / "model.joblib")

        with (
            patch("nlp_worker.topic_modeler._build_bertopic_model", return_value=mock_model),
            patch("nlp_worker.topic_modeler.joblib.dump"),
        ):
            train_topic_model(model_path=model_path, min_articles=20)

        # All 25 docs must now have a topic_id field
        updated = list(mock_db[COL_CLEAN].find({"topic_id": {"$exists": True}}))
        assert len(updated) == n

    def test_returns_correct_summary_shape(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        Summary dict must contain exactly the expected keys with correct types.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        n = 25
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        topics_output = [0] * 20 + [-1] * 5
        mock_model = self._make_mock_bertopic(topics_output)
        model_path = str(tmp_path / "model.joblib")

        with (
            patch("nlp_worker.topic_modeler._build_bertopic_model", return_value=mock_model),
            patch("nlp_worker.topic_modeler.joblib.dump"),
        ):
            summary = train_topic_model(model_path=model_path, min_articles=20)

        assert summary["n_articles"] == n
        assert summary["n_topics"] == 1          # only topic 0 (noise -1 excluded)
        assert summary["model_path"] == model_path
        assert abs(summary["noise_fraction"] - 5 / 25) < 1e-6
        assert isinstance(summary["topic_labels"], dict)
        assert 0 in summary["topic_labels"]
        assert -1 not in summary["topic_labels"]  # noise never gets a label

    def test_noise_fraction_is_zero_when_all_assigned(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """noise_fraction == 0.0 when no article is assigned to topic -1."""
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        n = 25
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        # All assigned to topic 0 — no noise
        mock_model = self._make_mock_bertopic([0] * n)
        model_path = str(tmp_path / "model.joblib")

        with (
            patch("nlp_worker.topic_modeler._build_bertopic_model", return_value=mock_model),
            patch("nlp_worker.topic_modeler.joblib.dump"),
        ):
            summary = train_topic_model(model_path=model_path, min_articles=20)

        assert summary["noise_fraction"] == 0.0

    def test_joblib_dump_is_called_with_correct_path(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """The model must be saved to the exact path requested."""
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import train_topic_model

        n = 25
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        mock_model = self._make_mock_bertopic([0] * n)
        model_path = str(tmp_path / "subdir" / "my_model.joblib")

        with (
            patch("nlp_worker.topic_modeler._build_bertopic_model", return_value=mock_model),
            patch("nlp_worker.topic_modeler.joblib.dump") as mock_dump,
        ):
            train_topic_model(model_path=model_path, min_articles=20)

        mock_dump.assert_called_once()
        saved_path = mock_dump.call_args[0][1]  # second positional arg to joblib.dump
        assert saved_path == model_path


# ---------------------------------------------------------------------------
# Tests: assign_topics
# ---------------------------------------------------------------------------


class TestAssignTopics:
    """
    Tests for the inference function.

    joblib.load() and BERTopic.transform() are mocked so no real model
    file or ML computation is needed.
    """

    def _make_mock_loaded_model(self, topics_out: list[int]) -> MagicMock:
        """
        Return a MagicMock that behaves like a loaded BERTopic model.
        transform() returns (topics, probabilities).
        """
        mock_model = MagicMock()
        mock_model.transform.return_value = (topics_out, [0.8] * len(topics_out))
        return mock_model

    def test_raises_file_not_found_when_model_missing(
        self, mock_db: MagicMock
    ) -> None:
        """
        FileNotFoundError must be raised if the model file does not exist.
        This forces the user to run train_topic_model() first.
        """
        from nlp_worker.topic_modeler import assign_topics

        with pytest.raises(FileNotFoundError, match="train_topic_model"):
            assign_topics(model_path="/nonexistent/path/model.joblib")

    def test_returns_zeros_when_no_new_articles(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        If all CLEAN articles already have a topic_id, assign_topics() must
        return early with all-zero counts. BERTopic.transform() must not be called.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import assign_topics

        # All docs already have topic_id
        docs = [_make_clean_doc(i, has_topic=True) for i in range(5)]
        mock_db[COL_CLEAN].insert_many(docs)

        model_path = str(tmp_path / "model.joblib")
        fake_model = self._make_mock_loaded_model([])

        # Create a dummy file so os.path.exists() returns True
        open(model_path, "w").close()

        with patch("nlp_worker.topic_modeler.joblib.load", return_value=fake_model):
            summary = assign_topics(model_path=model_path)

        assert summary == {"assigned": 0, "skipped_no_embedding": 0, "noise": 0}
        fake_model.transform.assert_not_called()

    def test_assigns_topics_to_new_articles(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        Happy path: 3 articles without topic_id → transform() called once →
        topic_id written to each document in MongoDB.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import assign_topics

        docs = [_make_clean_doc(i) for i in range(3)]
        mock_db[COL_CLEAN].insert_many(docs)

        # transform returns topic 0 for all 3 articles
        fake_model = self._make_mock_loaded_model([0, 0, 0])
        model_path = str(tmp_path / "model.joblib")
        open(model_path, "w").close()

        with patch("nlp_worker.topic_modeler.joblib.load", return_value=fake_model):
            summary = assign_topics(model_path=model_path)

        assert summary["assigned"] == 3
        assert summary["noise"] == 0

        # Verify MongoDB was updated
        updated = list(mock_db[COL_CLEAN].find({"topic_id": {"$exists": True}}))
        assert len(updated) == 3

    def test_counts_noise_articles_correctly(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        Articles assigned to topic -1 (HDBSCAN noise) must be counted
        as 'noise', not 'assigned'. They still get topic_id=-1 in MongoDB.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import assign_topics

        docs = [_make_clean_doc(i) for i in range(4)]
        mock_db[COL_CLEAN].insert_many(docs)

        # 2 assigned to topic 0, 2 noise
        fake_model = self._make_mock_loaded_model([0, 0, -1, -1])
        model_path = str(tmp_path / "model.joblib")
        open(model_path, "w").close()

        with patch("nlp_worker.topic_modeler.joblib.load", return_value=fake_model):
            summary = assign_topics(model_path=model_path)

        assert summary["assigned"] == 2
        assert summary["noise"] == 2

    def test_transform_receives_placeholder_docs_and_embeddings(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        model.transform() must receive:
        - placeholder_docs: list of empty strings (one per article)
        - embeddings: numpy array of shape (n, 384)

        Why verify this: if we accidentally pass the raw list of dicts from
        MongoDB instead of embeddings, BERTopic would try to re-encode strings
        and crash because embedding_model=None.
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import assign_topics

        n = 3
        docs = [_make_clean_doc(i) for i in range(n)]
        mock_db[COL_CLEAN].insert_many(docs)

        fake_model = self._make_mock_loaded_model([0] * n)
        model_path = str(tmp_path / "model.joblib")
        open(model_path, "w").close()

        with patch("nlp_worker.topic_modeler.joblib.load", return_value=fake_model):
            assign_topics(model_path=model_path)

        call_args = fake_model.transform.call_args
        passed_docs = call_args[0][0]           # first positional arg
        passed_embeddings = call_args[1]["embeddings"]  # keyword arg

        # Documents must be placeholder empty strings
        assert passed_docs == ["", "", ""]

        # Embeddings must be a numpy array of correct shape
        assert isinstance(passed_embeddings, np.ndarray)
        assert passed_embeddings.shape == (n, _EMBEDDING_DIM)

    def test_skips_articles_that_already_have_topic_id(
        self, mock_db: MagicMock, tmp_path: Any
    ) -> None:
        """
        The query filters by {"topic_id": {"$exists": False}}.
        Articles that already have a topic_id must not be passed to transform().
        """
        from shared.db import COL_CLEAN
        from nlp_worker.topic_modeler import assign_topics

        already_assigned = [_make_clean_doc(i, has_topic=True) for i in range(3)]
        new_docs = [_make_clean_doc(i + 10) for i in range(2)]
        mock_db[COL_CLEAN].insert_many(already_assigned + new_docs)

        # Only 2 new docs → transform called with 2 embeddings
        fake_model = self._make_mock_loaded_model([1, 1])
        model_path = str(tmp_path / "model.joblib")
        open(model_path, "w").close()

        with patch("nlp_worker.topic_modeler.joblib.load", return_value=fake_model):
            summary = assign_topics(model_path=model_path)

        assert summary["assigned"] == 2
        call_args = fake_model.transform.call_args
        passed_embeddings = call_args[1]["embeddings"]
        # Only 2 embeddings passed (not 5)
        assert passed_embeddings.shape == (2, _EMBEDDING_DIM)