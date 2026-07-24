"""
Tests for nlp_worker/pipeline.py.

Strategy: mock spaCy and langdetect so these tests run in CI without
downloading the 130MB language models. We test the pipeline's logic —
how it reacts to each component's output — not spaCy's NLP accuracy
(that is tested by spaCy's own test suite).

What we verify:
- _clean_text removes HTML tags and collapses whitespace correctly
- detect_language returns the right ISO code and None for unsupported langs
- process_article produces the expected output schema for ES/EN/FR
- process_article skips articles with empty text
- process_article skips articles whose language is unsupported
- process_article removes the MongoDB _id field from RAW before writing CLEAN
- run_nlp_pipeline returns the correct summary counts
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_article(
    n: int = 1,
    text: str = "The Amazon rainforest is under threat from deforestation.",
    lang_hint: str | None = None,
) -> dict[str, Any]:
    """Minimal RAW article dict matching the schema from scraper.py."""
    return {
        "_id": f"mongo_id_{n}",         
        "url": f"https://example.com/article-{n}",
        "title": f"Article {n}",
        "text": text,
        "detected_language": lang_hint,  
        "source": "carbon_brief",
        "publication_date": "2024-06-01",
        "ingestion_date": "2024-06-01T12:00:00Z",
        "sha256": f"{'a' * 60}{n:04d}",
    }


def _make_spacy_token(
    lemma: str,
    is_alpha: bool = True,
    is_stop: bool = False,
) -> MagicMock:
    """Build a minimal mock for a spaCy Token object."""
    tok = MagicMock()
    tok.lemma_ = lemma
    tok.is_alpha = is_alpha
    tok.is_stop = is_stop
    return tok


def _make_spacy_ent(text: str, label: str) -> MagicMock:
    """Build a minimal mock for a spaCy Span (named entity)."""
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    return ent


def _make_spacy_doc(
    tokens: list[MagicMock],
    ents: list[MagicMock],
) -> MagicMock:
    """Build a minimal mock for a spaCy Doc object."""
    doc = MagicMock()
    # spaCy Doc is iterable over tokens
    doc.__iter__ = MagicMock(return_value=iter(tokens))
    doc.ents = ents
    return doc


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_html_tags(self) -> None:
        from nlp_worker.pipeline import _clean_text

        raw = "Climate <em>crisis</em> is <strong>real</strong>."
        assert _clean_text(raw) == "Climate crisis is real."

    def test_collapses_whitespace(self) -> None:
        from nlp_worker.pipeline import _clean_text

        raw = "Solar   energy\n\nis\t growing."
        assert _clean_text(raw) == "Solar energy is growing."

    def test_strips_leading_trailing_whitespace(self) -> None:
        from nlp_worker.pipeline import _clean_text

        assert _clean_text("  hello  ") == "hello"

    def test_handles_empty_string(self) -> None:
        from nlp_worker.pipeline import _clean_text

        assert _clean_text("") == ""

    def test_html_with_extra_spaces_collapsed(self) -> None:
        from nlp_worker.pipeline import _clean_text

        raw = "<p>  Wind  <a href='#'>power</a>  is growing.  </p>"
        result = _clean_text(raw)
        assert "<" not in result
        assert "  " not in result


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    @patch("nlp_worker.pipeline.detect", return_value="en")
    def test_english_detected(self, _mock: MagicMock) -> None:
        from nlp_worker.pipeline import detect_language

        result = detect_language("The renewable energy sector is growing fast.")
        assert result == "en"

    @patch("nlp_worker.pipeline.detect", return_value="es")
    def test_spanish_detected(self, _mock: MagicMock) -> None:
        from nlp_worker.pipeline import detect_language

        result = detect_language("La energía renovable crece rápidamente en Europa.")
        assert result == "es"

    @patch("nlp_worker.pipeline.detect", return_value="fr")
    def test_french_detected(self, _mock: MagicMock) -> None:
        from nlp_worker.pipeline import detect_language

        result = detect_language("Les énergies renouvelables progressent en Europe.")
        assert result == "fr"

    @patch("nlp_worker.pipeline.detect", return_value="de")
    def test_unsupported_language_returns_none(self, _mock: MagicMock) -> None:
        from nlp_worker.pipeline import detect_language

        # German is not in our supported set — pipeline should skip the article
        result = detect_language("Die Solarenergie wächst in Deutschland.")
        assert result is None

    def test_langdetect_exception_returns_none(self) -> None:
        from langdetect import LangDetectException
        from nlp_worker.pipeline import detect_language

        with patch("nlp_worker.pipeline.detect", side_effect=LangDetectException(0, "")):
            result = detect_language("")
        assert result is None


# ---------------------------------------------------------------------------
# process_article
# ---------------------------------------------------------------------------

class TestProcessArticle:
    """
    We mock _get_nlp to return a fake spaCy model, and mock detect_language
    to return a fixed language. This lets us test process_article's logic
    (field mapping, filtering, _id removal) independently of actual NLP.
    """

    def _setup_spacy_mock(self, lang: str = "en") -> tuple[MagicMock, MagicMock]:
        """
        Return (mock_nlp_callable, mock_doc) configured with realistic tokens.
        """
        tokens = [
            _make_spacy_token("amazon", is_alpha=True, is_stop=False),
            _make_spacy_token("rainforest", is_alpha=True, is_stop=False),
            _make_spacy_token("the", is_alpha=True, is_stop=True),  
            _make_spacy_token("123", is_alpha=False, is_stop=False), 
        ]
        ents = [
            _make_spacy_ent("Amazon", "LOC"),
            _make_spacy_ent("Brazil", "GPE"),
        ]
        doc = _make_spacy_doc(tokens, ents)
        nlp_mock = MagicMock(return_value=doc)
        return nlp_mock, doc

    def test_returns_clean_article_with_expected_fields(self) -> None:
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock("en")
        raw = _make_raw_article(1)

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="en"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        assert result["detected_language"] == "en"
        assert isinstance(result["lemmatized_tokens"], list)
        assert isinstance(result["entities"], list)
        assert result["url"] == raw["url"]
        assert result["source"] == raw["source"]
        assert result["sha256"] == raw["sha256"]

    def test_strips_mongodb_id(self) -> None:
        """_id from RAW must not carry over to CLEAN — would cause duplicate key error."""
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock()
        raw = _make_raw_article(1)
        assert "_id" in raw  # confirm the fixture includes _id

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="en"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        assert "_id" not in result

    def test_lemmatized_tokens_exclude_stopwords_and_non_alpha(self) -> None:
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock()
        raw = _make_raw_article(1)

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="en"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        tokens = result["lemmatized_tokens"]
        assert "the" not in tokens
        assert "123" not in tokens
        assert "amazon" in tokens
        assert "rainforest" in tokens

    def test_entities_have_text_and_label(self) -> None:
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock()

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="en"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(_make_raw_article(1))

        assert result is not None
        for ent in result["entities"]:
            assert "text" in ent
            assert "label" in ent

    def test_empty_text_returns_none(self) -> None:
        from nlp_worker.pipeline import process_article

        raw = _make_raw_article(1, text="")
        result = process_article(raw)
        assert result is None

    def test_whitespace_only_text_returns_none(self) -> None:
        from nlp_worker.pipeline import process_article

        raw = _make_raw_article(1, text="   \n\t  ")
        result = process_article(raw)
        assert result is None

    def test_unsupported_language_returns_none(self) -> None:
        from nlp_worker.pipeline import process_article

        raw = _make_raw_article(1, text="Dies ist ein deutscher Text.")

        with patch("nlp_worker.pipeline.detect_language", return_value=None):
            result = process_article(raw)

        assert result is None

    def test_detected_language_overwrites_raw_value(self) -> None:
        """
        RAW has detected_language=None (set by scraper.py).
        The pipeline must overwrite it with the real detection result.
        """
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock()
        raw = _make_raw_article(1)
        assert raw["detected_language"] is None

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="fr"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        assert result["detected_language"] == "fr"

    def test_spanish_article_processed(self) -> None:
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock("es")
        raw = _make_raw_article(
            2,
            text="La energía solar y eólica crecen en América Latina.",
        )

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="es"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        assert result["detected_language"] == "es"

    def test_french_article_processed(self) -> None:
        from nlp_worker.pipeline import process_article

        nlp_mock, _ = self._setup_spacy_mock("fr")
        raw = _make_raw_article(
            3,
            text="Les énergies renouvelables progressent en Europe malgré les obstacles.",
        )

        with (
            patch("nlp_worker.pipeline.detect_language", return_value="fr"),
            patch("nlp_worker.pipeline._get_nlp", return_value=nlp_mock),
        ):
            result = process_article(raw)

        assert result is not None
        assert result["detected_language"] == "fr"


# ---------------------------------------------------------------------------
# run_nlp_pipeline
# ---------------------------------------------------------------------------

class TestRunNlpPipeline:
    """
    Tests for the async pipeline runner. We mock:
    - get_unprocessed_raw_urls: controls which URLs are returned
    - the RAW collection find_one: returns a raw article for each URL
    - process_article: controls what NLP produces
    - insert_clean_article: captures what would be written to CLEAN
    """

    async def test_processes_articles_and_returns_summary(self) -> None:
        from nlp_worker.pipeline import run_nlp_pipeline

        raw = _make_raw_article(1)
        clean = {**raw, "detected_language": "en", "lemmatized_tokens": ["amazon"], "entities": []}
        clean.pop("_id", None)

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=raw)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with (
            patch("nlp_worker.pipeline.get_db", return_value=mock_db),
            patch(
                "nlp_worker.pipeline.get_unprocessed_raw_urls",
                new=AsyncMock(return_value=["https://example.com/article-1"]),
            ),
            patch("nlp_worker.pipeline.process_article", return_value=clean),
            patch(
                "nlp_worker.pipeline.insert_clean_article",
                new=AsyncMock(),
            ),
        ):
            summary = await run_nlp_pipeline()

        assert summary["processed"] == 1
        assert summary["skipped"] == 0
        assert summary["failed"] == 0

    async def test_skips_articles_when_process_returns_none(self) -> None:
        from nlp_worker.pipeline import run_nlp_pipeline

        raw = _make_raw_article(1)

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=raw)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with (
            patch("nlp_worker.pipeline.get_db", return_value=mock_db),
            patch(
                "nlp_worker.pipeline.get_unprocessed_raw_urls",
                new=AsyncMock(return_value=["https://example.com/article-1"]),
            ),
            # process_article returns None → article should be counted as skipped
            patch("nlp_worker.pipeline.process_article", return_value=None),
        ):
            summary = await run_nlp_pipeline()

        assert summary["skipped"] == 1
        assert summary["processed"] == 0

    async def test_counts_failed_on_insert_error(self) -> None:
        from nlp_worker.pipeline import run_nlp_pipeline

        raw = _make_raw_article(1)
        clean = {**raw, "detected_language": "en", "lemmatized_tokens": [], "entities": []}

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=raw)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with (
            patch("nlp_worker.pipeline.get_db", return_value=mock_db),
            patch(
                "nlp_worker.pipeline.get_unprocessed_raw_urls",
                new=AsyncMock(return_value=["https://example.com/article-1"]),
            ),
            patch("nlp_worker.pipeline.process_article", return_value=clean),
            patch(
                "nlp_worker.pipeline.insert_clean_article",
                new=AsyncMock(side_effect=RuntimeError("DB write failed")),
            ),
        ):
            summary = await run_nlp_pipeline()

        assert summary["failed"] == 1
        assert summary["processed"] == 0

    async def test_empty_queue_returns_all_zeros(self) -> None:
        from nlp_worker.pipeline import run_nlp_pipeline

        mock_db = MagicMock()

        with (
            patch("nlp_worker.pipeline.get_db", return_value=mock_db),
            patch(
                "nlp_worker.pipeline.get_unprocessed_raw_urls",
                new=AsyncMock(return_value=[]),
            ),
        ):
            summary = await run_nlp_pipeline()

        assert summary == {"processed": 0, "skipped": 0, "failed": 0}

    async def test_missing_raw_article_counted_as_skipped(self) -> None:
        """
        Race condition: URL is in the unprocessed list but find_one returns None
        (article deleted between the URL fetch and the individual lookup).
        """
        from nlp_worker.pipeline import run_nlp_pipeline

        mock_col = MagicMock()
        mock_col.find_one = AsyncMock(return_value=None)  # disappeared
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with (
            patch("nlp_worker.pipeline.get_db", return_value=mock_db),
            patch(
                "nlp_worker.pipeline.get_unprocessed_raw_urls",
                new=AsyncMock(return_value=["https://example.com/ghost"]),
            ),
        ):
            summary = await run_nlp_pipeline()

        assert summary["skipped"] == 1
        assert summary["processed"] == 0