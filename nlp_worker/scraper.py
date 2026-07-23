from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import trafilatura
from shared.dedup import compute_url_hash



def scrape_article(url: str, source_name: str) -> Optional[dict[str, object]]:
    """
    Download and scrape an article from a given URL.

    Returns None if the download fails or if trafilatura can't extract the 
    main content (category page, paywall, pure JS, etc.).
    
    Args:
        url: URL of the article to scrape.
        source_name: Legible name of the source (e.g., 'carbon_brief').

    Returns:
        Dictionary with the RAW fields of the master, or None in case of failure.
    """
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None

    raw_json = trafilatura.extract(
        downloaded,
        with_metadata=True,
        output_format="json",
    )
    if raw_json is None:
        return None

    parsed = json.loads(raw_json)

    return {
        "url": url,
        "title": parsed.get("title"),
        "text": parsed.get("text"),
        "detected_language": None,          # poblado en Paso NLP (langdetect)
        "source": source_name,
        "publication_date": parsed.get("date"),  # str ISO o None si no está en HTML
        "ingestion_date": datetime.now(timezone.utc).isoformat(),
        "sha256": compute_url_hash(url),          # dedup — se usa en Paso 3 contra MongoDB
    }
