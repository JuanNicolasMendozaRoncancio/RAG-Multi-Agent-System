import os
import pytest
from fastapi.testclient import TestClient

os.environ["RAG_API_KEY"] = "test-key-123"
os.environ["MONGODB_URI"] = "mongodb://fake"   # evita RuntimeError en get_db
os.environ["HF_TOKEN"] = "hf_fake"
os.environ["GROQ_API_KEY"] = "gsk_fake"
os.environ["GEMINI_API_KEY"] = "AIza_fake"

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_rag_search_missing_key():
    """No header → 422 (FastAPI: campo requerido ausente)."""
    r = client.get("/rag/search", params={"query": "renewable energy"})
    assert r.status_code == 422


def test_rag_search_wrong_key():
    """Header presente pero incorrecto → 401."""
    r = client.get(
        "/rag/search",
        params={"query": "renewable energy"},
        headers={"X-RAG-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_rag_topics_active_wrong_key():
    """Mismo patrón para el segundo endpoint."""
    r = client.get(
        "/rag/topics/active",
        headers={"X-RAG-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_rag_search_query_too_short():
    """Query < 3 chars → 422 (validación FastAPI, antes de tocar DB o HF)."""
    r = client.get(
        "/rag/search",
        params={"query": "ab"},
        headers={"X-RAG-Key": "test-key-123"},
    )
    assert r.status_code == 422