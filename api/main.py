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
    yield _sse({"step":"Ingestion","Status":"running"})
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
            "procesados": nlp_summary.get("processed", 0),
            "omitidos": nlp_summary.get("skipped", 0),
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
        yield _sse({
            "step": "Embeddings",
            "status": "done",
            "elapsed_s": elapsed,
            "embebidos": emb_summary.get("embedded", 0),
            "ya_cacheados": emb_summary.get("already_cached", 0),
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
            "asignados": topic_summary.get("assigned", 0),
            "ruido": topic_summary.get("noise", 0),
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
        "entities": 0,
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