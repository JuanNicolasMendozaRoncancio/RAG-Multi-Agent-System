from __future__ import annotations

import hashlib

def compute_url_hash(url: str) -> str:
    """
    Computes SHA-256 hash of a normalized URL for deduplication.

    Normalization strips trailing slashes so that:
        https://example.com/article/
        https://example.com/article
    produce the same hash — they are the same article.

    SHA-256 is chosen over MD5 or SHA-1 because:
    - collision resistance matters when this hash is the primary dedup key in MongoDB
    - SHA-256 is available in Python's stdlib hashlib with no extra dependencies
    - the 64-char hex digest is compact enough for a MongoDB indexed field

    Args:
        url: Raw article URL, possibly with trailing slash.

    Returns:
        64-characte
    """
    normalized = url.rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()


def is_duplocate(url: str, existing_hashes: set[str]) -> bool:
    """
    Checks whether a URL has already been ingested.

    Args:
        url: Article URL to check.
        existing_hashes: Set of SHA-256 hashes already in the store.

    Returns:
        True if the article is a duplicate and should be skipped.
    """
        
    return compute_url_hash(url) in existing_hashes