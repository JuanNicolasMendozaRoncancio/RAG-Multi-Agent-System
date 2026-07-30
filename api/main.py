"""
FastAPI application — Climate & Energy Intelligence System.
 
Exposes:
- POST /pipeline/run  — triggers the full pipeline and streams progress via SSE.
- GET  /health        — liveness check (MongoDB connectivity + LLM provider status).
 
Server-Sent Events (SSE) design
---------------------------------
SSE is the correct protocol here because the pipeline is unidirectional
(server pushes progress to client) and takes ~90 seconds. REST would leave
the client waiting with no feedback. WebSockets are bidirectional and add
unnecessary complexity for a one-way progress stream.
 
Each SSE event is a JSON payload on a single `data:` line followed by two
newlines — the SSE wire format that browsers and httpx/EventSource clients
parse natively.
 
Embedding strategy
------------------
embed_articles_local() is used instead of embed_articles() because the HF
Serverless API (api-inference.huggingface.co) may be blocked by institutional
firewalls. Both functions produce identical 384-dim vectors from the same
model weights (paraphrase-multilingual-MiniLM-L12-v2). In a Docker/HF Spaces
environment where the API is reachable, swap the import to embed_articles().
 
BERTopic: train vs assign
--------------------------
The model is trained once (train_topic_model) and serialized with joblib.
On every subsequent on-demand run, assign_topics() is used — it loads the
serialized model and classifies new articles in seconds without retraining.
Retraining on every run would (a) be slow, (b) renumber topics, breaking
dashboard labels.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncGenerator
from typing import Any, Optional
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Query, HTTPException
from fastapi.responses import StreamingResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate & Energy intelligence System",
    description="RAG pipeline with multi-agent analysis. SSE on /pipeline/run.",
    version="0.1.0"
)

_local_embedder: Any = None

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse(payload:dict) -> str:
    """
    Format a dict as a single SSE event line.
 
    SSE wire format requires:
        data: <json>\n\n
 
    The double newline terminates the event. The client's EventSource parser
    fires an 'message' event for each double-newline-terminated block.
    """
    return f"data: {json.dumps(payload)}\n\n"

# ---------------------------------------------------------------------------
# Pipeline generator
# ---------------------------------------------------------------------------
async def _pipeline_generator() -> AsyncGenerator[str, None]:
    """
    Async generator that runs the full NLP pipeline step by step and yields
    SSE events for each step.
 
    Each step emits two events:
      1. {'step': <name>, 'status': 'running'} — immediately when the step starts.
      2. {'step': <name>, 'status': 'done', 'elapsed_s': <float>, ...extra} — when done.
 
    Why yield 'running' before awaiting the coroutine:
    The generator yields control back to the event loop after each yield,
    which flushes the SSE chunk to the client immediately. Without the
    'running' event, the client would see nothing until the step finishes.
 
    Why import pipeline functions inside the generator:
    These modules import spaCy, sentence-transformers, and BERTopic — all
    heavy at import time. Importing at module level would slow down the
    FastAPI startup and affect /health response time. Lazy imports keep
    startup fast and confine the loading cost to the first pipeline run.
    """
    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Ingesta
    # ------------------------------------------------------------------
    yield _sse({"step":"Ingestion","status":"running"})
    t0 = time.perf_counter()
    try:
        from nlp_worker.ingest import run_ingestion
        ingestion_summary = await run_ingestion()
        elapsed = round(time.perf_counter()-t0,2)
        yield _sse({
            "step": "Ingestion",
            "status": "done",
            "elapsed_s": elapsed,
            "new_articles": ingestion_summary.get("inserted", 0),
            "scraped": ingestion_summary.get("scraped", 0),
        })
    except Exception as exc:
        logger.error("Ingestion step failed: %s", exc, exc_info=True)
        yield _sse({"step": "Ingestion", "status": "error", "detail": str(exc)})
        return

    # ------------------------------------------------------------------
    # Step 2: NLP (language detection + lemmatization + NER)
    # ------------------------------------------------------------------
    yield _sse({"step": "NLP", "status": "running"})
    t0 = time.perf_counter()
    try:
        from nlp_worker.pipeline import run_nlp_pipeline
        nlp_summary = await run_nlp_pipeline()
        elapsed = round(time.perf_counter()- t0, 2)
        yield _sse({
            "step": "NLP",
            "status": "done",
            "elapsed_s": elapsed,
            "processed": nlp_summary.get("processed", 0),
            "skipped": nlp_summary.get("skipped", 0),
        })
    except Exception as exc:
        logger.error("NLP step failed: %s", exc, exc_info=True)
        yield _sse({"step": "NLP", "status": "error", "detail": str(exc)})
        return
    # ------------------------------------------------------------------
    # Step 3: Embeddings
    # Using embed_articles_local() because the HF Serverless API may be
    # blocked by institutional firewalls. Both produce identical 384-dim
    # vectors from the same model weights.
    # ------------------------------------------------------------------
    yield _sse({"step":"Embeddings", "status": "running"})
    t0 = time.perf_counter()
    try:
        from nlp_worker.embedder import embed_articles_local, embed_articles
        emb_summary = await embed_articles_local()
        elapsed = round(time.perf_counter() - t0, 2)
        yield _sse({
            "step": "Embeddings",
            "status": "done",
            "elapsed_s": elapsed,
            "embedded": emb_summary.get("embedded", 0),
            "already_cached": emb_summary.get("already_cached", 0),
        })
    except Exception as exc:
        logger.error("Embeddings step failed: %s", exc, exc_info=True)
        yield _sse({"step": "Embeddings", "status": "error", "detail": str(exc)})
        return

    # ------------------------------------------------------------------
    # Step 4: BERTopic — assign topics to new articles only.
    # train_topic_model() is NOT called here; the model is pre-trained and
    # serialized. assign_topics() loads the joblib file and runs inference
    # in seconds.
    # ------------------------------------------------------------------
    yield _sse({"step":"BERTopic", "status":"running"})
    t0 = time.perf_counter()
    try:
        from nlp_worker.topic_modeler import assign_topics
        topic_summary = await assign_topics()
        elapsed = round(time.perf_counter() - t0, 2)
        yield _sse({
            "step": "BERTopic",
            "status": "done",
            "elapsed_s": elapsed,
            "assigned": topic_summary.get("assigned", 0),
            "noise": topic_summary.get("noise", 0),
        })
    except FileNotFoundError:
        logger.warning("BERTopic model not found — skipping assign_topics.")
        yield _sse({
            "step": "BERTopic",
            "status": "skipped",
            "detail": "Model not trained. Run train_topic_model() first.",
        })
    except Exception as exc:
        logger.error("BERTopic step failed: %s", exc, exc_info=True)
        yield _sse({"step": "BERTopic", "status": "error", "detail": str(exc)})
        return

    # ------------------------------------------------------------------
    # Step 5: Sentiment classification (Groq → Gemini fallback)
    # ------------------------------------------------------------------
    yield _sse({"step":"Sentiment", "status":"running"})
    t0 = time.perf_counter()
    try:
        from nlp_worker.sentiment import run_sentiment_pipeline
        sent_summary = await run_sentiment_pipeline()
        elapsed = round(time.perf_counter()- t0,2)

        llm_provider = os.getenv("LLM_PROVIDER", "groq")
        yield _sse({
            "step": "Sentiment",
            "status": "done",
            "elapsed_s": elapsed,
            "clasified": sent_summary.get("classified", 0),
            "failed": sent_summary.get("failed", 0),
            "llm_provider": llm_provider,
        })
    except Exception as exc:
        logger.error("Sentimiento step failed: %s", exc, exc_info=True)
        yield _sse({"step": "Sentiment", "status": "error", "detail": str(exc)})
        return
    
    # ------------------------------------------------------------------
    # Step 6: Agents (Analytical → Contradiction → Trend → Synthesis)
    # run_agents() is synchronous — run in executor to avoid blocking
    # the event loop during the ~20s agent execution.
    # ------------------------------------------------------------------
    yield _sse({"step": "Agents", "status": "running"})
    t0 = time.perf_counter()
    try:
        import asyncio
        from agent_worker.agents import run_agents
        loop = asyncio.get_event_loop()
        agents_result = await loop.run_in_executor(None, run_agents)
        elapsed = round(time.perf_counter() - t0, 2)
        yield _sse({
            "step": "Agents",
            "status": "done",
            "elapsed_s": elapsed,
            "insights": len(agents_result.get("analytical_insights", {}).get("insights", [])),
            "contradictions": agents_result.get("contradictions", {}).get("contradiction_count", 0) if isinstance(agents_result.get("contradictions"), dict) else 0,
            "summary_chars": len(agents_result.get("summary", "")),
            "errors": agents_result.get("errors", []),
        })
    except Exception as exc:
        logger.error("Agents step failed: %s", exc, exc_info=True)
        yield _sse({"step": "Agents", "status": "error", "detail": str(exc)})
        return

    # ------------------------------------------------------------------
    # Final event
    # ------------------------------------------------------------------
    total_elapsed = round(time.perf_counter() - total_start, 2)
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    yield _sse({
        "step": "COMPLETED",
        "status": "done",
        "total_elapsed_s": total_elapsed,
        "llm_provider": llm_provider,
    })

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/pipeline/run")
async def run_pipeline() -> StreamingResponse:
    """
    Trigger the full NLP pipeline and stream progress via Server-Sent Events.
 
    Returns a StreamingResponse with media_type 'text/event-stream'.
    Each event is a JSON payload: {'step': str, 'status': str, ...extra}.
 
    Why POST and not GET:
    POST is semantically correct for an action that mutates state (inserts
    documents into MongoDB). GET must be idempotent by HTTP spec.
 
    Why x-accel-buffering: no header:
    Nginx (used by many reverse proxies) buffers SSE by default, which
    defeats the purpose of streaming. The header disables that buffering.
    """
    headers = {
        "Cache-control":"no-cache",
        "X-Accel-Buffering":"no",
        "Connection":"Keep-alive"
    }
    return StreamingResponse(
        _pipeline_generator(),
        media_type="text/event-stream",
        headers=headers 
    )

# ---------------------------------------------------------------------------
# GET /healh
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """
    Liveness check: verifies MongoDB is reachable and reports LLM provider.
 
    Why ping MongoDB here and not at startup:
    Motor creates the connection pool lazily — the first actual I/O happens
    when a query is issued. A startup check that just creates the client
    would always pass even if the URI is wrong. A ping on /health verifies
    the connection is actually live.
    """
    from shared.db import get_db
 
    db_status = "ok"
    try:
        db = get_db()
        await db.command("ping")
    except Exception as exc:
        logger.error("MongoDB ping failed: %s", exc)
        db_status = f"error: {exc}"
 
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "mongodb": db_status,
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
    }

# ---------------------------------------------------------------------------
# GET /articles
# ---------------------------------------------------------------------------
@app.get("/articles")
async def get_articles(
    source: Optional[str] = Query(None, description="Filter by source name"),
    language: Optional[str] = Query(None, description="Filter by ISO 639-1 language code"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment: positive|negative|neutral"),
    topic_id: Optional[int] = Query(None, description="Filter by BERTopic topic_id"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of articles to return"),
    skip: int = Query(0, ge=0, description="Number of articles to skip (pagination)"),
) -> dict[str, Any]:
    """
    Return articles from CURATED with optional filters.
 
    Why CURATED and not CLEAN:
    CURATED is the final enriched schema — it has sentiment, topic_id,
    principal_subject, and main_argument that the dashboard needs to display.
 
    Why exclude embedding, lemmatized_tokens, entities:
    These fields are large (embedding = 384 floats ~3KB, lemmatized_tokens
    can be 200+ items) and are consumed only by the NLP pipeline and agents
    internally. Serialising them across the API boundary wastes bandwidth
    and serialisation time with no benefit to the dashboard consumer.
 
    Why limit + skip and not cursor-based pagination:
    The corpus is small (~222 articles, growing slowly). Offset pagination
    is simpler to implement and consume. Cursor-based pagination adds
    complexity (stable sort + cursor token) only justified at >10k documents.
    """ 
    from shared.db import get_db, COL_CURATED
    db = get_db()
    col = db[COL_CURATED]

    query: dict[str, Any] = {}
    if source:
        query["source"] = source
    if language:
        query["detected_language"] = language
    if sentiment:
        if sentiment not in ("positive", "negative", "neutral"):
            raise HTTPException(status_code=400, detail="sentiment must be positive, negative, or neutral")
        query["sentiment"] = sentiment
    if topic_id is not None:
        query["topic_id"] = topic_id

    projection = {
        "embedding": 0,
        "lemmatized_tokens": 0,
        "_id": 0,
    }

    cursor = col.find(query, projection).sort("ingestion_date", -1).skip(skip).limit(limit)
    articles = await cursor.to_list(length=limit)
 
    total = await col.count_documents(query)
 
    return {
        "total": total,
        "returned": len(articles),
        "skip": skip,
        "limit": limit,
        "articles": articles,
    }

# ---------------------------------------------------------------------------
# GET /topics
# ---------------------------------------------------------------------------
@app.get("/topics")
async def get_topics(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
) -> dict[str, Any]:
    """
    Return active topics with article count and dominant sentiment for the period.
 
    Why reuse _query_by_topic() from agent_worker.agents:
    That function already implements the correct MongoDB aggregation over CURATED
    with the right date filter and grouping logic. Duplicating it would create
    two sources of truth. We call it in run_in_executor() because it uses pymongo
    (blocking I/O) and must not run directly inside an async route handler.
 
    Why run_in_executor and not asyncio.to_thread:
    Both are equivalent for CPU-bound or blocking I/O work. run_in_executor(None, fn)
    uses the default ThreadPoolExecutor and is the established FastAPI pattern for
    running synchronous blocking code without blocking the event loop.
    """
    from agent_worker.agents import _query_by_topic, _get_sync_db

    loop = asyncio.get_event_loop()
    db = await loop.run_in_executor(None, _get_sync_db)
    by_topic = await loop.run_in_executor(None, _query_by_topic, db, days)

    total_articles = sum(v["count"] for v in by_topic.values())

    topics = []
    for topic_id_str, stats in sorted(by_topic.items(), key = lambda x:x[1]["count"], reverse= True):
        topics.append({
            "topic_id": int(topic_id_str),
            "count": stats["count"],
            "relative_frequency": round(stats["count"] / total_articles, 4) if total_articles else 0.0,
            "avg_intensity": stats["avg_intensity"],
            "dominant_sentiment": stats["dominant_sentiment"],
        })
 
    return {
        "days": days,
        "total_articles_in_period": total_articles,
        "topics": topics,
    }

# ---------------------------------------------------------------------------
# GET /contradictions
# ---------------------------------------------------------------------------
@app.get("/contradictions")
async def get_contradictions(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
    threshold: float = Query(0.65, ge=0.0, le=1.0, description="Minimum cosine similarity score"),
    max_pairs: int = Query(5, ge=1, le=20, description="Maximum number of contradiction pairs to return"),
) -> dict[str, Any]:
    """
    Return pairs of articles with similar topics but opposing sentiment.
 
    Why threshold=0.65 as default and not 0.85:
    The corpus has 222 articles. At 0.85, the Vector Search returns zero pairs
    in most cases because articles must be nearly identical in semantic content.
    0.65 captures articles covering the same topic (e.g. offshore wind expansion)
    with genuinely opposite framings — which is the useful signal. The caller
    can raise the threshold via the query parameter if stricter matching is needed.
 
    Why reuse _find_contradictions() from agent_worker.agents:
    Same reasoning as /topics — avoids duplicating the Vector Search pipeline
    logic. Called via run_in_executor() because _find_contradictions() uses
    pymongo's synchronous aggregation with $vectorSearch.
    """
    from agent_worker.agents import _find_contradictions, _get_sync_db
 
    loop = asyncio.get_event_loop()
    db = await loop.run_in_executor(None, _get_sync_db)
    pairs = await loop.run_in_executor(
        None, _find_contradictions, db, days, threshold, max_pairs
    )
 
    return {
        "days": days,
        "threshold": threshold,
        "contradiction_count": len(pairs),
        "pairs": pairs,
    }

# ---------------------------------------------------------------------------
# GET /trends
# ---------------------------------------------------------------------------
@app.get("/trends")
async def get_trends(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
) -> dict[str, Any]:
    """
    Return weekly topic frequency and sentiment evolution, plus rising topics.
 
    Why combine _compute_topic_trend() and _detect_rising_topics() in one endpoint:
    The dashboard Tab 4 needs both: the time series data for the area chart and
    the rising topics list for the trend card. A single endpoint call avoids
    two round trips and two separate pymongo connections from the dashboard.
 
    Why ISO week granularity and not daily:
    With ~30 articles/week across 6 sources, daily granularity produces sparse
    data with many zero-count days. ISO week grouping yields meaningful frequency
    signals even on the small corpus. The dashboard can display this as a bar
    chart per week with no loss of interpretability.
    """
    from agent_worker.agents import _compute_topic_trend, _detect_rising_topics, _get_sync_db
 
    loop = asyncio.get_event_loop()
    db = await loop.run_in_executor(None, _get_sync_db)
    topic_trends = await loop.run_in_executor(None, _compute_topic_trend, db, days)
    rising_topics = _detect_rising_topics(topic_trends)
 
    # _detect_rising_topics is pure Python (no I/O) — no need for run_in_executor
 
    return {
        "days": days,
        "rising_topics": rising_topics,
        "topic_trends": topic_trends,
    }

# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------
@app.get("/summary")
async def get_summary() -> dict[str, Any]:
    """
    Return the most recent narrative summary produced by the Synthesis Agent.
 
    Why sort by timestamp descending and return only one document:
    The dashboard Tab 4 displays a single 'current state of discourse' card.
    Historical summaries are audit data — useful for the notebook walkthrough
    but not for the live dashboard. The caller can inspect SUMMARIES directly
    for historical access.
 
    Why not paginate SUMMARIES here:
    There will be at most one summary per pipeline run (~1/day with nightly
    Prefect). Pagination adds complexity with no practical benefit at that
    volume. A dedicated /summaries/history endpoint can be added in Step 22
    (notebook) if needed.
    """
    from shared.db import get_db, COL_SUMMARIES

    db = get_db()
    col = db[COL_SUMMARIES]
 
    doc = await col.find_one(
        {},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )
 
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="No summaries found. Run POST /pipeline/run first.",
        )
 
    return doc

# ---------------------------------------------------------------------------
# GET /embeddings
# ---------------------------------------------------------------------------
@app.get("/embeddings")
async def get_embeddings(
    limit: int = Query(500, ge=1, le=2000, description="Maximum number of documents to return"),
) -> dict[str, Any]:
    """
    Return embeddings with minimal metadata for UMAP 2D visualization (Tab 2).

    Why CURATED and not CLEAN:
    CURATED inherits the embedding from the sentiment pipeline and already
    carries topic_id, sentiment, source, and title — everything Tab 2 needs
    for coloring and hover text. Querying CLEAN would be a redundant round
    trip to a second collection for identical vectors.

    Why exclude text, lemmatized_tokens, entities, main_argument:
    Tab 2 only needs the vector (UMAP input), topic_id (point color),
    and url/title/source/sentiment/detected_language (hover labels).
    Full text fields would add hundreds of KB with no benefit.
    """
    from shared.db import get_db, COL_CURATED

    db = get_db()
    col = db[COL_CURATED]

    projection = {
        "_id": 0,
        "url": 1,
        "title": 1,
        "source": 1,
        "detected_language": 1,
        "sentiment": 1,
        "topic_id": 1,
        "embedding": 1,   
    }

    cursor = col.find(
        {"embedding": {"$exists": True}, "topic_id":{"$exists":True}}, projection
    ).limit(limit)

    docs = await cursor.to_list(length=limit)

    return {
        "total": len(docs),
        "docs": docs,
    }



# ---------------------------------------------------------------------------
# Authentication dependency — inter-system endpoints (/rag/*)
# ---------------------------------------------------------------------------
from fastapi import Depends, Header

def verify_rag_key(x_rag_key: str = Header(...)) -> None:
    """
    Validate the X-RAG-Key header against the RAG_API_KEY environment variable.
 
    Why a FastAPI dependency and not middleware:
    Middleware applies to ALL routes. Only the /rag/* endpoints require
    authentication — internal endpoints (/articles, /topics, etc.) are
    consumed by the dashboard on the same network and do not need a key.
    A dependency injected per-endpoint is surgical: it adds auth exactly
    where needed without touching the rest of the API.
 
    Why Header(...) with no default:
    The ellipsis makes the header required — FastAPI returns 422 automatically
    if it is absent. We then return 401 explicitly if the value is wrong,
    which is the semantically correct HTTP response for authentication failure
    (422 = malformed request, 401 = valid request but unauthenticated).
 
    Raises
    ------
    HTTPException 401
        If RAG_API_KEY is not configured or the provided key does not match.
    """
    expected = os.getenv("RAG_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="RAG_API_KEY is not configured on the server.",
        )
    if x_rag_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid X-RAG-Key.",
        )

# ---------------------------------------------------------------------------
# HF Serverless API helper — encode a single query string into a 384-dim vector
# ---------------------------------------------------------------------------
import httpx 

_HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_HF_ENCODE_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{_HF_MODEL}"
)
_HF_TIMEOUT = 60.0  # seconds — accounts for cold start on HF free tier

async def _enconde_query(query: str) -> list[float]:
    """
    Encode a single query string into a 384-dimensional vector using the
    HF Serverless Inference API.
 
    Why the same model as the embedder (paraphrase-multilingual-MiniLM-L12-v2):
    Vector Search requires that the query vector and the stored document vectors
    live in the same embedding space. If we encoded the query with a different
    model, cosine similarity scores would be meaningless — the spaces are not
    aligned. Using the identical model guarantees that "renewable energy" as a
    query lands near "renewable energy" in article embeddings.
 
    Why call HF API here instead of loading sentence-transformers locally:
    The API container (FastAPI on HF Spaces) has a lean requirements.txt.
    Loading sentence-transformers would add ~500MB of model weights to the
    API container at startup, slowing cold starts and consuming RAM that the
    free tier cannot spare. The HF Serverless API runs inference on HF's
    infrastructure — the API container stays lightweight.
 
    Why a dedicated function and not reusing _call_hf_api from embedder.py:
    embedder.py lives in nlp_worker, which is a separate Docker service with
    its own requirements.txt. Importing across service boundaries would couple
    two independent containers at the Python level. This function is 10 lines
    of httpx — the duplication cost is lower than the coupling cost.
 
    Parameters
    ----------
    query:
        Raw user query string (not pre-processed — the model handles tokenisation).
 
    Returns
    -------
    list[float]
        384-dimensional embedding vector.
 
    Raises
    ------
    HTTPException 503
        If the HF API is unreachable or returns an error.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured on the server.",
        )
 
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": [query], "options": {"wait_for_model": True}}

    async with httpx.AsyncClient(timeout=_HF_TIMEOUT) as client:
        try:
            response = await client.post(_HF_ENCODE_URL, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HF API returned HTTP %d for query encoding: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise HTTPException(
                status_code=503,
                detail=f"Embedding service unavailable (HTTP {exc.response.status_code}).",
            )
        except httpx.TransportError as exc:
            logger.error("HF API network error during query encoding: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Embedding service unreachable.",
            )

    result: list[Any] = response.json()

    vector = result[0]
    if isinstance(vector[0],list):
        n_tokens = len(vector)
        dim = len(vector[0])
        vector = [sum(vector[t][d] for t in range(n_tokens)) / n_tokens for d in range(dim)]

    return vector  # type: ignore[return-value]

async def _enconde_query_local(query: str) -> list[float]:
    """
    Encode a single query string into a 384-dimensional vector using the
    local sentence-transformers model.

    This function is a drop-in local replacement for the HF Serverless API version.
    It maintains the exact same input/output signature and FastAPI exception handling,
    while running inference locally on the CPU/GPU.
    """
    global _local_embedder
    if _local_embedder is None:                  
        from sentence_transformers import SentenceTransformer
        _local_embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    if _local_embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unreachable.",
        )

    try:
        # Generamos el vector delegando la carga a un hilo secundario
        vector = await asyncio.to_thread(_local_embedder.encode, query)
        return vector.tolist()
    except Exception as exc:
        logger.error("Local inference error during query encoding: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Embedding service unreachable.",
        )

# ---------------------------------------------------------------------------
# GET /rag/search  — semantic retrieval endpoint
# ---------------------------------------------------------------------------
 
@app.get("/rag/search")
async def rag_search(
    query: str = Query(..., min_length=3, description="Natural language query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    _: None = Depends(verify_rag_key),
) -> dict[str, Any]:
    """
    Encode a natural language query and retrieve the top-k most semantically
    similar articles from CURATED using MongoDB Atlas Vector Search.
 
    Why search CURATED and not CLEAN:
    CURATED has all the fields the RCA Agent needs:
    sentiment, intensity, topic_id, principal_subject, and main_argument.
    CLEAN has embeddings too, but lacks the sentiment enrichment. Since
    CURATED inherits the embedding from CLEAN (written during sentiment
    classification), there is no need to query two collections.
 
    Why $vectorSearch on CURATED and not a text index:
    Text indexes match exact keywords. Vector Search captures semantic
    similarity: a query "renewable energy transition" will retrieve articles
    about "energía renovable" and "transition énergétique" because
    paraphrase-multilingual-MiniLM-L12-v2 maps these to nearby vectors.
    This is the core value proposition of the RAG component for Proyecto Agentes.
 
    Why numCandidates = top_k * 10:
    Atlas Vector Search with HNSW uses approximate nearest neighbour (ANN).
    numCandidates controls the size of the candidate set before re-ranking.
    A ratio of 10x gives recall >95% on small corpora while staying well
    within M0 free tier limits. The MongoDB documentation recommends at least
    10x as the minimum for reliable recall.
 
    Response fields are deliberately minimal:
    - embedding, lemmatized_tokens, entities excluded (heavy, not useful to caller)
    - score included: lets the RCA Agent threshold results by confidence
    - extract (200 chars) instead of full text: keeps response payload small
 
    Authentication: X-RAG-Key header required (verified by Depends(verify_rag_key)).
    """
    from shared.db import get_db, COL_CURATED

    query_vector = await _enconde_query_local(query)

    db = get_db()
    col = db[COL_CURATED]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "url": 1,
                "title": 1,
                "source": 1,
                "detected_language": 1,
                "sentiment": 1,
                "intensity": 1,
                "topic_id": 1,
                "principal_subject": 1,
                "main_argument": 1,
                "publication_date": 1,
                "extract": {"$substr": ["$text", 0, 200]},
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = await col.aggregate(pipeline).to_list(length=top_k)

    logger.info(
        "rag_search: query='%s' top_k=%d results=%d", query, top_k, len(results)
    )
 
    return {
        "query": query,
        "top_k": top_k,
        "returned": len(results),
        "results": results,
    }

# ---------------------------------------------------------------------------
# GET /rag/topics/active — active topics endpoint
# ---------------------------------------------------------------------------
 
@app.get("/rag/topics/active")
async def rag_topics_active(
    days: int = Query(7, ge=1, le=90, description="Lookback window in days"),
    _: None = Depends(verify_rag_key),
) -> dict[str, Any]:
    """
    Return topics with growing frequency over the requested period, ordered
    by linear regression slope (steepest ascent first).
 
    Why linear regression slope and not simple delta (last week vs previous):
    The master document (section 6.3) specifies 'ordered by pendiente de
    regresión lineal'. With a 7-day window there are typically 1-2 ISO weeks
    of data per topic, making slope == delta in practice. But with days >= 14
    the regression is computed over 2+ weekly points, making it genuinely
    more informative than a simple last-vs-previous comparison. The same
    function handles both cases correctly.
 
    Why this endpoint exists separately from GET /trends:
    GET /trends is for the internal dashboard — it returns the full time series
    for chart rendering. GET /rag/topics/active is for the Narrative Agent of
    Proyecto Agentes — it needs only the ranked list of rising topics to
    enrich its narrative with "topics gaining momentum in climate discourse".
    Different consumers, different shapes.
 
    Why reuse _compute_topic_trend() from agent_worker.agents:
    The aggregation logic is identical to what the Trend Agent already does.
    Duplicating it would create two sources of truth. The difference is the
    post-processing: here we apply linear regression on the weekly counts,
    /trends returns the raw time series.
 
    Authentication: X-RAG-Key header required (verified by Depends(verify_rag_key)).
    """
    import numpy as np
    from agent_worker.agents import _compute_topic_trend, _get_sync_db
 
    loop = asyncio.get_event_loop()
    db = await loop.run_in_executor(None, _get_sync_db)
    topic_trends = await loop.run_in_executor(None, _compute_topic_trend, db, days)
 
    # Compute linear regression slope for each topic over its weekly counts.
    # Why numpy polyfit degree=1: fits a line y = slope*x + intercept over
    # the (week_number, article_count) pairs. The slope is the rate of change
    # per week. A positive slope means the topic is gaining frequency.
    # We use week index (0, 1, 2, ...) as x to avoid large ISO week numbers
    active_topics = []
    for topic_id_str, weekly_data in topic_trends.items():
        counts = [w["count"] for w in weekly_data]
 
        if len(counts) == 1:
            slope = 0.0
        else:
            x = np.arange(len(counts), dtype=np.float64)
            y = np.array(counts, dtype=np.float64)
            slope = float(np.polyfit(x, y, deg=1)[0])
 
        total_count = sum(counts)
        last_week = weekly_data[-1]
 
        active_topics.append({
            "topic_id": int(topic_id_str),
            "slope": round(slope, 4),
            "total_articles": total_count,
            "avg_intensity": last_week["avg_intensity"],
            "dominant_sentiment": last_week["dominant_sentiment"],
            "weeks_observed": len(weekly_data),
        })
 
    # Sort by slope descending — steepest growth first.
    active_topics.sort(key=lambda t: t["slope"], reverse=True)
 
    logger.info(
        "rag_topics_active: days=%d topics_found=%d", days, len(active_topics)
    )
 
    return {
        "days": days,
        "topics_count": len(active_topics),
        "topics": active_topics,
    }