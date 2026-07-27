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

from dotenv import load_dotenv
from fastapi import FastAPI
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