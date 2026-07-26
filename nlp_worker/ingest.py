"""
Ingestion script: reads RSS feeds, scrapes articles, and writes to MongoDB RAW.

Run from project root:
    python -m nlp_worker.ingest
"""
from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from nlp_worker.feed_reader import fetch_all_feeds
from nlp_worker.scraper import scrape_article
from shared.db import get_existing_hashes, insert_raw_articles_bulk

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

async def run_ingestion() -> dict[str, int]:
    """
    Fetch all RSS feeds, scrape each article, and write new ones to RAW.

    Returns a summary dict with keys: fetched, scraped, inserted, skipped.
    """
    logger.info("Step 1/3 — fetching RSS feeds...")
    feed_items = fetch_all_feeds()
    logger.info("Feeds returned %d URLs across all sources.", len(feed_items))

    logger.info("Step 2/3 — loading existing hashes from MongoDB...")
    existing_hashes = await get_existing_hashes()
    logger.info("Found %d articles already in RAW.", len(existing_hashes))

    logger.info("Step 3/3 — scraping and inserting new articles...")
    articles = []
    scraped = 0
    skipped_scrape = 0

    for item in feed_items:
        url = item["url"]
        source = item["source"]

        article = scrape_article(url, source)
        if article is None:
            skipped_scrape += 1
            logger.debug("Scrape failed for %s", url)
            continue

        scraped += 1
        articles.append(article)

    inserted, skipped_dup = await insert_raw_articles_bulk(articles, existing_hashes)

    summary = {
        "fetched": len(feed_items),
        "scraped": scraped,
        "skipped_scrape": skipped_scrape,
        "inserted": inserted,
        "skipped_duplicate": skipped_dup,
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    asyncio.run(run_ingestion())