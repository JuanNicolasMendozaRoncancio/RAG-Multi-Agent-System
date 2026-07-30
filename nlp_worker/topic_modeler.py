"""
Topic modeling module for the RAG pipeline.
 
Trains a BERTopic model over pre-computed embeddings stored in the CLEAN
collection, serializes the trained model with joblib, and assigns topics
to new articles on every subsequent pipeline run without retraining.
 
Design decisions
----------------
BERTopic over LDA:
    LDA is bag-of-words — "solar" (EN) and "solaire" (FR) land in different
    topics because LDA sees tokens, not semantics. BERTopic operates on the
    384-dimensional multilingual embeddings already stored in CLEAN: "solar
    energy" and "énergie solaire" are already close in that space because the
    sentence-transformer was trained on parallel multilingual data. BERTopic
    inherits that cross-lingual property for free.
 
UMAP before HDBSCAN (n_components=5, metric='cosine'):
    HDBSCAN suffers from the curse of dimensionality — in 384 dimensions,
    pairwise distances become nearly uniform and clusters become invisible.
    UMAP reduces 384 → 5 dimensions while preserving local neighbourhood
    structure. We use metric='cosine' because sentence-transformer embeddings
    are L2-normalised: cosine similarity is more meaningful than Euclidean
    distance for these vectors.
    n_components=5 (not 2): 2D is for visualisation (dashboard Tab 2).
    Clustering needs more dimensions to retain enough semantic structure.
    5 is the BERTopic default that balances reduction and structure retention.
 
HDBSCAN (min_cluster_size=10):
    Does not require specifying the number of topics a priori — a parameter
    we cannot know before seeing the data. DBSCAN classic requires epsilon
    (sensitive to local density). HDBSCAN is the hierarchical variant that
    selects epsilon adaptively. min_cluster_size=10 means a topic needs at
    least 10 articles to exist — filters noise articles into topic -1.
 
c-TF-IDF for topic representation:
    Topic keywords should be discriminative across topics, not just frequent
    globally. c-TF-IDF (class-based TF-IDF) treats each topic as a
    concatenated "document" and computes TF-IDF against all other topics.
    Applied to lemmatized tokens (already stopword-free), it produces clean,
    discriminative keywords per topic.
 
joblib over pickle for serialisation:
    BERTopic internals contain large numpy arrays (UMAP reducer state,
    HDBSCAN condensed tree). joblib.dump applies efficient numpy-aware
    compression that produces files 3-5x smaller than pickle. It is the
    scikit-learn standard for model serialisation.
 
train vs assign separation:
    Retraining on every nightly run would (a) be computationally expensive,
    and (b) renumber topics between runs, breaking the dashboard's topic
    labels. The model is trained once on the initial corpus. assign_topics()
    loads the serialised model and classifies new articles in seconds.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from shared.db import COL_CLEAN, get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Default path for the serialised model.
_DEFAULT_MODEL_PATH = "models/bertopic_model.joblib"

_UMAP_PARAMS: dict[str, Any] = {
    "n_components": 5,
    "n_neighbors": 15,
    "metric": "cosine",
    "random_state": 42,        # reproducibility
    "low_memory": False,       # False is faster when the corpus fits in RAM
}

_HDBSCAN_PARAMS: dict[str, Any] = {
    "min_cluster_size": 10,
    "metric": "euclidean",     
    "cluster_selection_method": "eom",   
    "prediction_data": True,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_bertopic_model() -> BERTopic:
    """
    Instantiate a BERTopic model with project-specific UMAP and HDBSCAN.
 
    Why pass embedding_model=None:
    We feed pre-computed embeddings directly to BERTopic.fit_transform().
    Passing embedding_model=None tells BERTopic not to load any internal
    sentence-transformer — we already ran that step in embedder.py.
    This avoids downloading a second copy of the model and makes the topic
    modeler independent of the HF API.
 
    Why verbose=False:
    UMAP and HDBSCAN emit verbose logs by default that are noisy in
    production. We control logging through Python's logging module instead.
    """
    umap_model = UMAP(**_UMAP_PARAMS)
    hdbscan_model = HDBSCAN(**_HDBSCAN_PARAMS)

    return BERTopic(
        umap_model= umap_model,
        hdbscan_model= hdbscan_model,
        embedding_model= None,
        language= "multilingual",
        verbose=False,
    )

async def _load_embeddings_from_db(db: Any) -> tuple[list[str], list[list[str]], np.ndarray]:
    col = db[COL_CLEAN]

    docs = await col.find(
        {"embedding": {"$exists": True}},
        {"url": 1, "embedding": 1, "lemmatized_tokens": 1, "_id": 0},
    ).to_list(length=None)

    if not docs:
        return [], [], np.empty((0, 384), dtype=np.float32)

    urls = [doc["url"] for doc in docs]
    token_lists = [doc.get("lemmatized_tokens", []) for doc in docs]
    embeddings = np.array(
        [doc["embedding"] for doc in docs],
        dtype=np.float32,
    )
    return urls, token_lists, embeddings

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def train_topic_model(
    model_path: str = _DEFAULT_MODEL_PATH,
    min_articles: int = 20,
) -> dict[str, Any]:
    
    db = get_db()
    urls, token_lists, embeddings = await _load_embeddings_from_db(db)

    n_articles = len(urls)
    logger.info("Loaded %d embeddings from CLEAN for BERTopic training.", n_articles)

    if n_articles < min_articles:
        raise ValueError(
            f"BERTopic requires at least {min_articles} articles with embeddings. "
            f"Found {n_articles}. Run the ingestion + NLP + embedding pipeline first."
        )

    model = _build_bertopic_model()

    logger.info("Training BERTopic (UMAP 384→5 + HDBSCAN)...")
    placeholder_docs = [" ".join(tokens) for tokens in token_lists]
    topics, _ = model.fit_transform(placeholder_docs, embeddings)

    col = db[COL_CLEAN]
    for url, topic_id in zip(urls, topics):
        await col.update_one(                              # await añadido
            {"url": url},
            {"$set": {"topic_id": int(topic_id)}},
        )
    logger.info("Topic assignments written to CLEAN collection.")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("BERTopic model serialised to '%s'.", model_path)

    topic_ids = set(topics)
    n_topics = len(topic_ids - {-1})
    noise_count = sum(1 for t in topics if t == -1)
    noise_fraction = noise_count / n_articles if n_articles > 0 else 0.0

    topic_labels: dict[int, list[str]] = {}
    for tid in topic_ids:
        if tid == -1:
            continue
        words_scores = model.get_topic(tid)
        if words_scores:
            topic_labels[tid] = [w for w, _ in words_scores[:10]]

    summary = {
        "n_articles": n_articles,
        "n_topics": n_topics,
        "topic_labels": topic_labels,
        "model_path": model_path,
        "noise_fraction": noise_fraction,
    }
    logger.info("Training complete: %s", summary)
    return summary


async def assign_topics(
    model_path: str = _DEFAULT_MODEL_PATH,
) -> dict[str, int]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained BERTopic model not found at '{model_path}'. "
            "Run train_topic_model() first."
        )

    model: BERTopic = joblib.load(model_path)
    logger.info("Loaded BERTopic model from '%s'.", model_path)

    db = get_db()
    col = db[COL_CLEAN]

    docs = await col.find(                                 # await añadido
        {"embedding": {"$exists": True}, "topic_id": {"$exists": False}},
        {"url": 1, "embedding": 1, "_id": 0},
    ).to_list(length=None)

    if not docs:
        logger.info("No new articles to assign topics to.")
        return {"assigned": 0, "skipped_no_embedding": 0, "noise": 0}

    urls = [doc["url"] for doc in docs]
    embeddings = np.array(
        [doc["embedding"] for doc in docs],
        dtype=np.float32,
    )

    logger.info("Assigning topics to %d new articles...", len(docs))

    placeholder_docs = ["" for _ in urls]
    topics, _ = model.transform(placeholder_docs, embeddings=embeddings)

    assigned = noise = 0
    for url, topic_id in zip(urls, topics):
        await col.update_one(                              # await añadido
            {"url": url},
            {"$set": {"topic_id": int(topic_id)}},
        )
        if topic_id == -1:
            noise += 1
        else:
            assigned += 1

    summary = {
        "assigned": assigned,
        "skipped_no_embedding": 0,
        "noise": noise,
    }
    logger.info("Topic assignment complete: %s", summary)
    return summary

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = asyncio.run(train_topic_model())
    print(summary)