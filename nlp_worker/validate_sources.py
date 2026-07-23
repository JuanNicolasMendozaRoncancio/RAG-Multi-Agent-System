from __future__ import annotations

import sys
from nlp_worker.feed_reader import SOURCES, fetch_feed
from nlp_worker.scraper import scrape_article
from shared.dedup import compute_url_hash

def validate_source(source_name: str) -> dict[str, object]:
    """
    Validates one source end-to-end.

    Returns a result dict with:
    - articles_in_feed: how many URLs feedparser found
    - scrape_success: whether trafilatura extracted content from the first article
    - hash_consistent: whether the same URL always produces the same hash
    - sample: the scraped article dict (or None)
    - error: error message if something failed
    """
    result: dict[str, object] = {
        "source": source_name,
        "articles_in_feed": 0,
        "scrape_success": False,
        "hash_consistent": False,
        "sample": None,
        "error": None,
    }

    try:
        articles = fetch_feed(source_name)
    except Exception as e:
        result["error"] = f"Feed fetch failed {e}"
        return result

    result["articles_in_feed"] = len(articles)
    if not articles:
        result["error"] = "Feed returned 0 articles"
        return result

    first_url = articles[0]["url"]
    try:
        sample = scrape_article(first_url, source_name)
    except Exception as e:
        result["error"] = f"Scrape raised exception: {e}"
        return result

    if sample is None:
        result["error"] = f"trafilatura returned None for {first_url}"
        return result

    result["scrape_success"] = True
    result["sample"] = sample

    hash_1 = compute_url_hash(first_url)
    hash_2 = compute_url_hash(first_url)
    hash_trailing = compute_url_hash(first_url + "/")
    result["hash_consistent"] = (hash_1 == hash_2 == hash_trailing)

    return result

def print_report(results: list[dict[str, object]]) -> None:
    """Prints a human-readable validation report."""
    print("\n" + "=" * 60)
    print("SOURCE VALIDATION REPORT")
    print("=" * 60)

    healthy = 0
    for r in results:
        status = "OK" if r["scrape_success"] else "✗ FAIL"
        print(f"\n[{status}] {r['source']}")
        print(f"  Feed articles : {r['articles_in_feed']}")
        print(f"  Scrape OK     : {r['scrape_success']}")
        print(f"  Hash consistent: {r['hash_consistent']}")

        if r["error"]:
            print(f"  Error         : {r['error']}")

        if r["sample"]:
            sample = r["sample"]
            print(f"  Title         : {sample.get('title', 'N/A')}")
            print(f"  Date          : {sample.get('publication_date', 'N/A')}")
            print(f"  Text length   : {len(sample.get('text') or '')} chars")
            print(f"  SHA-256       : {sample.get('sha256', 'N/A')[:16]}...")

        if r["scrape_success"]:
            healthy += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {healthy}/{len(results)} sources healthy")
    print("=" * 60 + "\n")

    if healthy < len(results):
        sys.exit(1)  # non-zero exit so CI can detect failures if needed


if __name__ == "__main__":
    print("Validating all sources — this will make real HTTP requests...")
    results = [validate_source(name) for name in SOURCES]
    print_report(results)