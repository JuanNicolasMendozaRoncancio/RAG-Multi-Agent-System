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

def _load_embeddings_from_db(db: Any) -> tuple[list[str], np.ndarray]:
    """
    Fetch all CLEAN documents that have an embedding field.
 
    Returns
    -------
    urls : list[str]
        Article URLs in the same order as the embedding matrix rows.
    embeddings : np.ndarray
        Shape (n_articles, 384). Each row is the embedding of one article.
 
    Why return URLs alongside embeddings:
    After BERTopic assigns a topic id to each row, we need the URL to write
    the assignment back to the correct MongoDB document. Keeping URLs and
    embedding rows in the same order makes that mapping O(1) by index.
 
    Why np.ndarray and not list[list[float]]:
    BERTopic.fit_transform() expects a numpy array. Converting a list of
    lists requires an extra allocation; returning ndarray directly avoids it.
    The dtype float32 halves memory vs float64 with no precision loss for
    cosine-based operations.
    """
    col = db[COL_CLEAN]

    docs = list(col.find(
        {"embedding": {"$exists": True}},
        {"url": 1, "embedding": 1, "_id": 0},
    ))

    if not docs:
        return [], np.empty((0, 384), dtype= np.float32)

    urls = [doc["url"] for doc in docs]
    embeddings = np.array(
        [doc["embedding"] for doc in docs],
        dtype=np.float32,
    )
    return urls, embeddings

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_topic_model(
    model_path: str = _DEFAULT_MODEL_PATH,
    min_articles: int = 20,
) -> dict[str, Any]:
    """
    Train BERTopic on all CLEAN embeddings and serialise the model.
 
    This function is designed to be called once — on the initial corpus —
    and then not again unless explicitly triggered (e.g. a major corpus
    expansion or topic drift). Nightly runs call assign_topics() instead.
 
    Parameters
    ----------
    model_path:
        Filesystem path where the trained model is saved with joblib.
        Parent directory is created if it does not exist.
    min_articles:
        Minimum number of articles with embeddings required to train.
        Training BERTopic on fewer than ~20 articles produces unreliable
        topics because HDBSCAN cannot form stable clusters.
 
    Returns
    -------
    dict[str, Any]
        Summary with keys:
        - n_articles: int — number of articles used for training
        - n_topics: int — number of topics found (excluding noise topic -1)
        - topic_labels: dict[int, list[str]] — top-10 keywords per topic
        - model_path: str — where the model was saved
        - noise_fraction: float — fraction of articles assigned to topic -1
 
    Raises
    ------
    ValueError
        If fewer than min_articles embeddings are found in CLEAN.
    RuntimeError
        If MONGODB_URI is not set.
    """
    db = get_db()
    urls, embeddings = _load_embeddings_from_db(db)

    n_articles = len(urls)
    logger.info("Loaded %d embeddings from CLEAN for BERTopic training.", n_articles)

    if n_articles < min_articles:
        raise ValueError(
            f"BERTopic requires at least {min_articles} articles with embeddings. "
            f"Found {n_articles}. Run the ingestion + NLP + embedding pipeline first."
        )

    model = _build_bertopic_model()

    logger.info("Training BERTopic (UMAP 384→5 + HDBSCAN)...")
    topics, _ = model.fit_transform(embeddings)
    # topics: list[int], one topic id per article. -1 = noise (no cluster).
    # _     : list[float], per-document probabilities (unused here — we store
    #         only the topic id, not the soft probability).

    # Write topic assignments back to MongoDB.
    # Why update_one with $set and not bulk_write here:
    # The number of articles is bounded by the M0 corpus size (~hundreds to
    # low thousands). Individual update_one calls are acceptable. If the
    # corpus grew to tens of thousands, switching to bulk_write would be
    # the right optimisation — but that is premature here.
    col = db[COL_CLEAN]
    for url, topic_id in zip(urls, topics):
        col.update_one(
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

def assign_topics(
    model_path: str = _DEFAULT_MODEL_PATH,
):
    """
    Assign topics to CLEAN articles that do not yet have a topic_id field.
 
    Loads the serialised BERTopic model and calls model.transform()
    on the new embeddings. Does NOT retrain the model.
 
    Why model.transform() and not re-running fit_transform():
    transform() projects new embeddings through the *fitted* UMAP (preserving
    the training topology) and then calls HDBSCAN's approximate_predict
    internally via BERTopic's hdbscan_delegator. This is the correct
    BERTopic API for inference on new documents. Re-running fit_transform()
    would re-train from scratch, producing a different topic numbering each run.
 
    Parameters
    ----------
    model_path:
        Path to the joblib-serialised BERTopic model produced by
        train_topic_model().
 
    Returns
    -------
    dict[str, int]
        Keys: 'assigned', 'skipped_no_embedding', 'noise'
        'noise' counts articles assigned to topic -1 (no cluster found).
 
    Raises
    ------
    FileNotFoundError
        If the model file does not exist (train_topic_model() not yet run).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained BERTopic model not found at '{model_path}'. "
            "Run train_topic_model() first."
        )

    model: BERTopic = joblib.load(model_path)
    logger.info("Loaded BERTopic model from '%s'.", model_path)

    db = get_db()
    col = db[COL_CLEAN]

    docs = list(col.find(
        {"embedding": {"$exists": True}, "topic_id": {"$exists": False}},
        {"url": 1, "embedding": 1, "_id": 0},
    ))

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
    topics, _ = model.transform(placeholder_docs, embeddings= embeddings)

    assigned = noise = 0
    for url, topic_id in zip(urls, topics):
        col.update_one(
            {"url": url},
            {"$set": {"topic_id": int(topic_id)}},
        )
        if topic_id == -1:
            noise += 1
        else:
            assigned += 1
    summary = {
        "assigned": assigned,
        "skipped_no_embedding": 0,   # filtered by query above
        "noise": noise,
    }
    logger.info("Topic assignment complete: %s", summary)
    return summary