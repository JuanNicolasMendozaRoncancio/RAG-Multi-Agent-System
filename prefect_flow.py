"""
Nightly maintenance flow for the Climate & Energy Intelligence System.

Runs automatically every night at 03:00 UTC via Prefect Cloud.

Why only ingestion + NLP (not embeddings, BERTopic, sentiment, agents):
- Ingestion (trafilatura + feedparser) and NLP (spaCy) are purely local —
  zero external API calls, zero token consumption.
- Embeddings require the HF Serverless API (rate-limited free tier).
- Sentiment requires Groq/Gemini (14,400 req/day combined free budget).
- Agents synthesize across the full corpus — meaningful only when triggered
  on-demand after a human reviews the new articles.

The split preserves free-tier budgets: by 06:00 UTC, new articles are
already in RAW and CLEAN, ready for the on-demand pipeline (POST /pipeline/run
from the dashboard) to run embeddings + sentiment + agents on top of them.

Execution model:
- Prefect Cloud schedules and monitors the flow.
- The worker runs on the local machine (or a GitHub Actions cron job).
- MongoDB Atlas M0 is the only external dependency at runtime.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

load_dotenv()


# ---------------------------------------------------------------------------
# Tasks — each Prefect task maps to one pipeline stage
# ---------------------------------------------------------------------------
# Why @task and not plain async functions:
# Prefect tasks give each stage its own retry policy, timeout, and log panel
# in the Prefect Cloud UI. A recruiter opening the flow run sees "Ingestion ✓
# (47 new articles)" and "NLP ✓ (47 processed)" as separate cards — much
# more informative than a single "flow succeeded" entry.
# ---------------------------------------------------------------------------

@task(name="Ingestion", retries=2, retry_delay_seconds=60)
async def ingestion_task() -> dict[str, int]:
    """
    Fetch all RSS feeds, scrape new articles, write to MongoDB RAW.

    retries=2: trafilatura occasionally times out on slow sources (Reporterre,
    Mongabay). Two automatic retries handle transient network failures without
    waking anyone up at 03:00 UTC.
    retry_delay_seconds=60: gives the source server a minute to recover before
    the retry — avoids hammering a temporarily overloaded feed.

    Returns the ingestion summary dict for logging in the flow.
    """
    logger = get_run_logger()
    from nlp_worker.ingest import run_ingestion

    logger.info("Ingestion task starting...")
    summary = await run_ingestion()
    logger.info(
        "Ingestion complete: fetched=%d scraped=%d inserted=%d skipped_dup=%d",
        summary.get("fetched", 0),
        summary.get("scraped", 0),
        summary.get("inserted", 0),
        summary.get("skipped_duplicate", 0),
    )
    return summary


@task(name="NLP", retries=1, retry_delay_seconds=30)
async def nlp_task() -> dict[str, int]:
    """
    Process all RAW articles that have no CLEAN counterpart.

    retries=1: spaCy model loading occasionally fails on the first call if
    the OS memory allocator is slow. One retry is sufficient.
    retry_delay_seconds=30: 30s is enough for the allocator to release memory.

    Returns the NLP pipeline summary dict.
    """
    logger = get_run_logger()
    from nlp_worker.pipeline import run_nlp_pipeline

    logger.info("NLP task starting...")
    summary = await run_nlp_pipeline()
    logger.info(
        "NLP complete: processed=%d skipped=%d failed=%d",
        summary.get("processed", 0),
        summary.get("skipped", 0),
        summary.get("failed", 0),
    )
    return summary


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
# Why @flow and not just calling the tasks directly:
# The @flow decorator makes Prefect track the run as a unit — start time,
# end time, state (Completed/Failed), task dependencies, and all logs are
# grouped under one flow run in the UI. Without @flow, tasks are invisible.
# ---------------------------------------------------------------------------

@flow(name="rag-climate-nightly", log_prints=True)
async def nightly_maintenance_flow() -> None:
    """
    Nightly maintenance flow: ingest new articles and run NLP processing.

    Sequential execution (not parallel):
    NLP must run after ingestion — it processes articles that ingestion just
    wrote to RAW. Running them in parallel would cause NLP to miss the
    articles inserted in the same run.

    log_prints=True: redirects Python print() calls to Prefect's log system,
    so any print() inside run_ingestion() or run_nlp_pipeline() shows up in
    the Prefect Cloud UI without requiring explicit get_run_logger() calls
    in those functions.
    """
    logger = get_run_logger()
    run_timestamp = datetime.now(timezone.utc).isoformat()
    logger.info("Nightly maintenance flow started at %s", run_timestamp)

    # Step 1: Ingestion
    ingestion_summary = await ingestion_task()

    # Step 2: NLP — only if ingestion produced new articles
    new_articles = ingestion_summary.get("inserted", 0)
    if new_articles == 0:
        logger.info("No new articles inserted — skipping NLP task.")
    else:
        await nlp_task()

    logger.info("Nightly maintenance flow complete.")


# ---------------------------------------------------------------------------
# Entrypoint — runs the flow once and exits.
# Scheduling is handled externally by GitHub Actions cron.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(nightly_maintenance_flow())