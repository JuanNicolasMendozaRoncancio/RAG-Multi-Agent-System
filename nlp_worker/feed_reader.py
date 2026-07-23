from __future__ import annotations

from typing import Optional

import feedparser

SOURCES: dict[str, dict[str, str]] = {
    "yale_enviroment_360": {
        "url": "https://e360.yale.edu/feed.xml",
        "language": "en",
    },
    "carbon_brief": {
        "url": "https://www.carbonbrief.org/feed",
        "language": "en",
    },
    "bon_pot": {
        "url": "https://bonpote.com/feed",
        "language": "fr",
    },
    "reporterre": {
        "url": "https://reporterre.net/spip.php?page=backend-simple",
        "language": "fr",
    },
    "mongabay_latam": {
        "url": "https://es.mongabay.com/feed",
        "language": "es",
    },
    "climatica": {
        "url": "https://climatica.coop/feed",
        "language": "es",
    },
}

def fetch_feed(source_name: str) -> list[dict[str, Optional[str]]]:
    """
    Parses the RSS feed for a given source and returns article metadata.

    Does NOT download or extract article text — that is scraper.py's job.
    feedparser handles the network request internally; if the feed is
    unreachable it returns an empty entries list rather than raising.

    Args:
        source_name: Key from SOURCES dict (e.g. 'carbon_brief').

    Returns:
        List of dicts with article URL and feed-level metadata.
        Empty list if the feed is unreachable or has no entries.
    """
    source = SOURCES[source_name]
    feed = feedparser.parse(source["url"])

    articles = []
    for entry in feed.entries:
        url = entry.get("link")
        if url is None:
            continue

        articles.append({
            "url": url,
            "source": source_name,
            "feed_title": entry.get("title"),
            "feed_date": entry.get("published"),
            "feed_author": entry.get("author"),
            "feed_categories":[
                tag.term for tag in entry.get("tags", [])
            ]
        })

    return articles

def fetch_all_feeds() -> list[dict[str, Optional[str]]]:
    """
    Fetches all 6 sources and returns combined article list.

    Sources that fail (network error, empty feed) are skipped silently —
    the pipeline must be resilient to individual source outages.
    """
    all_articles = []
    for source_name in SOURCES:
        articles = fetch_feed(source_name)
        all_articles.extend(articles)
    return all_articles