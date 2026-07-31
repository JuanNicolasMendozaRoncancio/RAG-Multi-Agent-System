"""
Climate & Energy Intelligence System — Streamlit Dashboard.
 
Tab 1: Pipeline runner (SSE consumption via st.status) + paginated article feed.
Tabs 2–4: implemented in Steps 15.
 
Architecture note: the dashboard is a pure API consumer.
All data access goes through the FastAPI layer — no direct MongoDB calls here.
This enforces the same separation of concerns as the inter-system /rag/* endpoints:
the dashboard cannot break if the internal DB schema changes, as long as the
API contracts hold.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

import numpy as np

load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://rag-climate-api-1024198341439.us-central1.run.app")
 
# Sentiment badge colors: these map to CSS background colors injected via
# st.markdown. Using teal/red/slate keeps the palette coherent with the
# dark theme without importing an external CSS framework.
_SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#0f766e",   # teal-700
    "negative": "#b91c1c",   # red-700
    "neutral":  "#475569",   # slate-600
}
 
_LANG_LABELS: dict[str, str] = {
    "en": "EN",
    "es": "ES",
    "fr": "FR",
}

# How many articles to show per page in the feed.
_PAGE_SIZE = 10
 
# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Climate & Energy Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark theme with teal accent, JetBrains Mono for data
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
 
    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
    .mono {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.78rem;
    }
 
    /* ── Sentiment badges ── */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .badge-positive { background: #0f766e; color: #ccfbf1; }
    .badge-negative { background: #b91c1c; color: #fee2e2; }
    .badge-neutral  { background: #475569; color: #e2e8f0; }
    .badge-lang     { background: #1e3a5f; color: #93c5fd; }
    .badge-topic    { background: #3b1f6e; color: #ddd6fe; }
    .badge-source   { background: #1a2e1a; color: #86efac; }
 
    /* ── Article card ── */
    .article-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: border-color 0.15s;
    }
    .article-card:hover { border-color: #2dd4bf; }
    .article-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .article-meta {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .article-extract {
        font-size: 0.80rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .article-subject {
        font-size: 0.75rem;
        color: #64748b;
        font-style: italic;
    }
 
    /* ── Pipeline log ── */
    .pipeline-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.80rem;
        color: #2dd4bf;
    }
    .pipeline-error {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.80rem;
        color: #f87171;
    }
 
    /* ── LLM provider indicator ── */
    .llm-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: #0c2340;
        border: 1px solid #1e40af;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #93c5fd;
    }
 
    /* ── Metric card ── */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #2dd4bf;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
 
    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 20px 0;
    }
 
    /* ── Pagination ── */
    [data-testid="stButton"] button {
        background: #1e293b;
        border: 1px solid #334155;
        color: #94a3b8;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    [data-testid="stButton"] button:hover {
        border-color: #2dd4bf;
        color: #2dd4bf;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# Streamlit re-runs the entire script on every widget interaction.
# session_state persists values across re-runs within the same browser session.
# We initialise all keys here so the rest of the code can read them safely
# without KeyError guards everywhere.
if "filter_source" not in st.session_state:
    st.session_state.filter_source = "All"
if "filter_language" not in st.session_state:
    st.session_state.filter_language = "All"
if "filter_sentiment" not in st.session_state:
    st.session_state.filter_sentiment = "All"
if "filter_topic_id" not in st.session_state:
    st.session_state.filter_topic_id = None
if "feed_page" not in st.session_state:
    st.session_state.feed_page = 0
if "selected_article" not in st.session_state:
    st.session_state.selected_article = None
if "last_pipeline_provider" not in st.session_state:
    st.session_state.last_pipeline_provider = None
if "pipeline_summary" not in st.session_state:
    st.session_state.pipeline_summary = None

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    Perform a GET request against the FastAPI backend.
 
    Returns None on connection error so callers can show a graceful
    degraded state rather than crashing the dashboard.
 
    Why requests and not httpx: requests is already a transitive dependency
    of many Streamlit packages and is sufficient for simple synchronous
    GET calls in the dashboard layer. httpx would add a dependency for
    no functional gain here — the dashboard runs synchronously.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_BASE_URL}. Is `uvicorn api.main:app` running?")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None

def _run_pipeline_sse() -> None:
    """
    Trigger POST /pipeline/run and consume the SSE stream with st.status().
 
    Why st.status() and not st.spinner():
    st.spinner() shows a single undifferentiated spinner for the whole duration.
    st.status() shows a collapsible panel where each step is a separate item
    with its own running/done/error state. The recruiter can see exactly which
    step the pipeline is on, and the elapsed time per step tells them the
    computational cost of each stage. This is exactly what the master plan
    specifies (section 7.1).
 
    Why requests + stream=True + iter_lines():
    Streamlit runs synchronously. There is no native SSE client in Streamlit.
    requests with stream=True keeps the TCP connection open and iter_lines()
    yields each line as it arrives from the server without buffering the whole
    response. This is the standard pattern for consuming SSE in Python without
    an async framework.
 
    Why iter_lines(chunk_size=1) is not used:
    iter_lines() already handles line buffering correctly for SSE — it waits
    for a newline before yielding. chunk_size affects the read buffer, not
    the yielded unit. The default is correct.
    """
    st.session_state.pipeline_summary = None
    st.session_state.last_pipeline_provider = None
 
    step_times: dict[str, float] = {}
    step_metrics: dict[str, dict[str, Any]] = {}
 
    with st.status("Running pipeline...", expanded=True) as status:
        try:
            with requests.post(
                f"{API_BASE_URL}/pipeline/run",
                stream=True,
                timeout=500,  # pipeline can take up to ~90s; 300s is safe
            ) as resp:
                resp.raise_for_status()
 
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
 
                    # SSE lines are: b"data: {...json...}"
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data:"):
                        continue
 
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
 
                    step = event.get("step", "")
                    event_status = event.get("status", "")
 
                    if step == "COMPLETED":
                        total = event.get("total_elapsed_s", "?")
                        provider = event.get("llm_provider", "groq")
                        st.session_state.last_pipeline_provider = provider
                        st.session_state.pipeline_summary = step_metrics
 
                        st.write(
                            f'<span class="pipeline-metric"> pipeline complete in {total}s</span>',
                            unsafe_allow_html=True,
                        )
                        status.update(
                            label=f"Pipeline complete ({total}s)",
                            state="complete",
                            expanded=False,
                        )
                        st.rerun()
                        return
 
                    if event_status == "running":
                        step_times[step] = time.time()
                        st.write(f"⏳ **{step}** — running...")
 
                    elif event_status == "done":
                        elapsed = event.get("elapsed_s", "?")
                        step_metrics[step] = event
 
                        # Build a compact metrics line from whatever extra
                        # fields the event has (step-specific counters).
                        extra_parts = []
                        skip_keys = {"step", "status", "elapsed_s"}
                        for k, v in event.items():
                            if k not in skip_keys:
                                extra_parts.append(f"{k}={v}")
                        extra = "  ·  ".join(extra_parts)
 
                        st.write(
                            f'<span class="pipeline-metric">'
                            f"✓ <strong>{step}</strong> — {elapsed}s"
                            f"{('  ·  ' + extra) if extra else ''}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )
 
                    elif event_status == "skipped":
                        detail = event.get("detail", "")
                        st.write(f"⏭ **{step}** — skipped  ·  {detail}")
 
                    elif event_status == "error":
                        detail = event.get("detail", "unknown error")
                        st.write(
                            f'<span class="pipeline-error">'
                            f" <strong>{step}</strong> — {detail}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )
                        status.update(
                            label=f"Pipeline failed at {step}",
                            state="error",
                            expanded=True,
                        )
                        return
 
        except requests.ConnectionError:
            status.update(label="Cannot reach API", state="error")
            st.error(f"Cannot reach API at {API_BASE_URL}. Is `uvicorn api.main:app --port 8000 --reload` running?")
        except requests.HTTPError as exc:
            status.update(label="API error", state="error")
            st.error(f"API returned {exc.response.status_code}")
        except Exception as exc:
            status.update(label="Unexpected error", state="error")
            st.error(str(exc))

# ---------------------------------------------------------------------------
# Sidebar — filters + LLM indicator + health
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    """
    Render the sidebar with filters and system health indicators.
 
    Why read filters from session_state and write back to session_state:
    Streamlit re-renders the whole script on every widget change. Reading
    from session_state lets us reset pagination (feed_page = 0) whenever
    a filter changes, which is the correct UX — jumping to page 3 of a
    filtered result would be confusing.
    """
    with st.sidebar:
        st.markdown("## 🌍 Climate Intelligence")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
        # ── LLM provider indicator ──────────────────────────────────────
        provider = st.session_state.last_pipeline_provider
        if provider:
            icon = "⚡" if provider == "groq" else "✦"
            label = "Groq (LPU)" if provider == "groq" else "Gemini Flash"
            st.markdown(
                f'<div class="llm-badge">{icon} {label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
 
        # ── Health check ────────────────────────────────────────────────
        health = _api_get("/health")
        if health:
            db_ok = health.get("mongodb") == "ok"
            db_icon = "🟢" if db_ok else "🔴"
            active_provider = health.get("llm_provider", "groq")
            st.markdown(
                f"{db_icon} MongoDB Atlas  \n"
                f"LLM: `{active_provider}`",
            )
        else:
            st.markdown("API unreachable")
 
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
        # ── Filters ─────────────────────────────────────────────────────
        st.markdown("### Filters")
 
        sources = ["All", "yale_enviroment_360", "carbon_brief", "bon_pot",
                   "reporterre", "mongabay_latam", "climatica"]
        new_source = st.selectbox(
            "Source", sources,
            index=sources.index(st.session_state.filter_source),
        )
 
        languages = ["All", "en", "es", "fr"]
        new_language = st.selectbox(
            "Language", languages,
            index=languages.index(st.session_state.filter_language),
        )
 
        sentiments = ["All", "positive", "negative", "neutral"]
        new_sentiment = st.selectbox(
            "Sentiment", sentiments,
            index=sentiments.index(st.session_state.filter_sentiment),
        )
 
        topic_input = st.text_input(
            "Topic ID (integer)",
            value="" if st.session_state.filter_topic_id is None
            else str(st.session_state.filter_topic_id),
            placeholder="e.g. 0, 1, 2",
        )
 
        # Reset pagination when any filter changes
        filter_changed = (
            new_source != st.session_state.filter_source
            or new_language != st.session_state.filter_language
            or new_sentiment != st.session_state.filter_sentiment
        )
 
        st.session_state.filter_source = new_source
        st.session_state.filter_language = new_language
        st.session_state.filter_sentiment = new_sentiment
 
        try:
            new_topic_id = int(topic_input) if topic_input.strip() else None
        except ValueError:
            new_topic_id = None
 
        if new_topic_id != st.session_state.filter_topic_id:
            filter_changed = True
        st.session_state.filter_topic_id = new_topic_id
 
        if filter_changed:
            st.session_state.feed_page = 0
            st.session_state.selected_article = None
 
        if st.button("Clear filters", use_container_width=True):
            st.session_state.filter_source = "All"
            st.session_state.filter_language = "All"
            st.session_state.filter_sentiment = "All"
            st.session_state.filter_topic_id = None
            st.session_state.feed_page = 0
            st.session_state.selected_article = None
            st.rerun()
 
        # ── NER panel for selected article ──────────────────────────────
        if st.session_state.selected_article:
            article = st.session_state.selected_article
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### Named Entities")
 
            entities: list[dict[str, str]] = article.get("entities", [])
            if entities:
                # Group by label for a compact display
                by_label: dict[str, list[str]] = {}
                for ent in entities:
                    label = ent.get("label", "?")
                    text = ent.get("text", "")
                    by_label.setdefault(label, []).append(text)
 
                for label, texts in sorted(by_label.items()):
                    unique = list(dict.fromkeys(texts))[:8]  # deduplicate, cap at 8
                    st.markdown(
                        f"**{label}**  \n"
                        + "  \n".join(f"`{t}`" for t in unique)
                    )
            else:
                st.caption("No entities extracted for this article.")

# ---------------------------------------------------------------------------
# Article card renderer
# ---------------------------------------------------------------------------
def _render_article_card(article: dict[str, Any], idx: int) -> None:
    """
    Render one article as a card with sentiment/language/topic/source badges.
 
    Why st.markdown with unsafe_allow_html and not st.expander:
    st.expander would work but looks generic. The card pattern gives us
    control over the visual hierarchy — title dominant, badges secondary,
    extract tertiary — which matches how a journalist or recruiter would
    scan a feed. The hover border effect (CSS) provides interactivity
    without JavaScript.
 
    Why a separate 'Select' button per card and not clicking the card:
    Streamlit does not support click handlers on arbitrary HTML elements.
    A small button is the pragmatic solution that preserves the card layout.
    """
    sentiment = article.get("sentiment", "neutral")
    lang = article.get("detected_language", "?")
    source = article.get("source", "?")
    topic_id = article.get("topic_id")
    title = article.get("title") or "Untitled"
    extract = article.get("text", "")[:200] if article.get("text") else ""
    subject = article.get("principal_subject", "")
    intensity = article.get("intensity")
 
    topic_badge = (
        f'<span class="badge badge-topic">topic {topic_id}</span>'
        if topic_id is not None else ""
    )
    intensity_str = f" ({intensity:.0%})" if intensity is not None else ""
 
    card_html = f"""
    <div class="article-card">
      <div class="article-title">{title}</div>
      <div class="article-meta">
        <span class="badge badge-{sentiment}">{sentiment}{intensity_str}</span>
        <span class="badge badge-lang">{_LANG_LABELS.get(lang, lang)}</span>
        <span class="badge badge-source">{source.replace('_', ' ')}</span>
        {topic_badge}
      </div>
      <div class="article-extract">{extract}{"…" if len(extract) == 200 else ""}</div>
      {"<div class='article-subject'>" + subject + "</div>" if subject else ""}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
 
    col_link, col_select = st.columns([3, 1])
    with col_link:
        url = article.get("url", "")
        if url:
            st.markdown(
                f'<a href="{url}" target="_blank" style="font-size:0.75rem;color:#2dd4bf;">↗ Read article</a>',
                unsafe_allow_html=True,
            )
    with col_select:
        # Fetch the CLEAN article for NER — but CURATED already has entities
        # inherited through the sentiment pipeline, so we use what we have.
        if st.button("Entities →", key=f"select_{idx}"):
            st.session_state.selected_article = article
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 1 main render
# ---------------------------------------------------------------------------
def _render_tab1() -> None:
    """
    Render Tab 1: pipeline runner + summary metrics + article feed.
    """
    # ── Header ──────────────────────────────────────────────────────────
    col_title, col_run = st.columns([3, 1])
    with col_title:
        st.markdown("## Pipeline & Article Feed")
        st.caption(
            "Trigger the full ingestion → NLP → embedding → topic → sentiment → agents pipeline. "
            "Results stream in real time via SSE."
        )
    with col_run:
        st.markdown("")  # vertical spacing
        run_clicked = st.button(
            "▶ Run pipeline",
            type="primary",
            use_container_width=True,
        )
 
    if run_clicked:
        _run_pipeline_sse()
 
    # ── Post-run summary metrics ─────────────────────────────────────────
    # These are populated by _run_pipeline_sse() after a successful run.
    # They persist in session_state until the next run.
    if st.session_state.pipeline_summary:
        summary = st.session_state.pipeline_summary
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Last run")
 
        # Extract the key counters across steps
        new_articles = summary.get("Ingestion", {}).get("new_articles", "—")
        nlp_processed = summary.get("NLP", {}).get("procesed", "—")
        embedded = summary.get("Embeddings", {}).get("embebidos", "—")
        classified = summary.get("Sentiment", {}).get("clasified", "—")
        insights = summary.get("Agents", {}).get("insights", "—")
 
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, value, label in [
            (c1, new_articles,  "New articles"),
            (c2, nlp_processed, "NLP processed"),
            (c3, embedded,      "Embedded"),
            (c4, classified,    "Classified"),
            (c5, insights,      "Insights"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
 
    # ── Article feed ─────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Article feed")
 
    # Build query params from session_state filters
    params: dict[str, Any] = {
        "limit": _PAGE_SIZE,
        "skip": st.session_state.feed_page * _PAGE_SIZE,
    }
    if st.session_state.filter_source != "All":
        params["source"] = st.session_state.filter_source
    if st.session_state.filter_language != "All":
        params["language"] = st.session_state.filter_language
    if st.session_state.filter_sentiment != "All":
        params["sentiment"] = st.session_state.filter_sentiment
    if st.session_state.filter_topic_id is not None:
        params["topic_id"] = st.session_state.filter_topic_id
 
    data = _api_get("/articles", params=params)
 
    if data is None:
        return  # error already shown by _api_get
 
    total = data.get("total", 0)
    articles = data.get("articles", [])
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    current_page = st.session_state.feed_page
 
    # Feed header with article count
    st.markdown(
        f'<span class="mono" style="color:#64748b;">'
        f"{total} articles  ·  page {current_page + 1} of {total_pages}"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")
 
    if not articles:
        st.info("No articles match the current filters.")
    else:
        for i, article in enumerate(articles):
            _render_article_card(article, idx=current_page * _PAGE_SIZE + i)
 
    # ── Pagination controls ───────────────────────────────────────────────
    st.markdown("")
    col_prev, col_info, col_next = st.columns([1, 2, 1])
 
    with col_prev:
        if current_page > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.feed_page -= 1
                st.session_state.selected_article = None
                st.rerun()
 
    with col_info:
        st.markdown(
            f'<div style="text-align:center" class="mono" style="color:#475569;">'
            f"Page {current_page + 1} / {total_pages}"
            f"</div>",
            unsafe_allow_html=True,
        )
 
    with col_next:
        if current_page < total_pages - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.feed_page += 1
                st.session_state.selected_article = None
                st.rerun()   


# ---------------------------------------------------------------------------
# Tab 2 — Topic Map (UMAP 2D)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def _compute_umap_2d(docs_json: str) -> np.ndarray:
    """
    Reduce 384-dim embeddings to 2D for visualization.

    Why a new UMAP with n_components=2 and not the serialized BERTopic model:
    The BERTopic model uses UMAP with n_components=5 for clustering — those
    5 dimensions preserve enough structure for HDBSCAN but are not human-readable.
    This is a separate reduction whose only job is to produce X/Y coordinates
    for a scatter plot. Two different objectives, two different UMAP instances.

    Why @st.cache_data with ttl=3600:
    UMAP over 222 x 384 arrays takes ~2-3s. Without caching it re-runs on every
    sidebar interaction. ttl=3600 is coherent with the nightly pipeline cadence.

    Why accept docs_json (str) and not a list:
    st.cache_data hashes the arguments. numpy arrays and lists are unhashable
    or change hash on every call. Serializing to JSON string gives a stable,
    hashable cache key.
    """
    import json
    from umap import UMAP

    docs = json.loads(docs_json)
    embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)

    reducer = UMAP(
        n_components=2,
        n_neighbors=10,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    return reducer.fit_transform(embeddings)


def _render_tab2() -> None:
    """
    Render Tab 2: interactive UMAP 2D scatter plot of BERTopic topics.

    Data flow:
      GET /embeddings → 222 docs with embedding + topic_id + metadata
      → UMAP 384D→2D (cached) → Plotly scatter colored by topic_id
    """
    import json
    import numpy as np
    import plotly.graph_objects as go

    st.markdown("## Topic Map")
    st.caption(
        "Each point is an article projected into 2D with UMAP. "
        "Color = BERTopic topic. Noise articles (topic -1) shown in grey."
    )

    data = _api_get("/embeddings")
    if data is None:
        return

    docs = data.get("docs", [])
    if not docs:
        st.info("No articles with embeddings found. Run the pipeline first.")
        return

    # Language filter — applied before UMAP so the map reflects the filtered corpus
    languages = sorted({d.get("detected_language", "?") for d in docs})
    lang_options = ["All"] + languages
    selected_lang = st.selectbox(
        "Filter by language", lang_options, key="tab2_lang"
    )
    if selected_lang != "All":
        docs = [d for d in docs if d.get("detected_language") == selected_lang]

    if len(docs) < 5:
        st.warning("Too few articles to compute UMAP (minimum 5). Adjust the language filter.")
        return

    # Cache key: JSON string of (url, topic_id, language) tuples — stable and hashable
    # We do NOT include the full embedding in the cache key (too large).
    # Instead we use a fingerprint: sorted urls + selected_lang.
    fingerprint = json.dumps(
        {"lang": selected_lang, "urls": sorted(d["url"] for d in docs)}
    )
    coords_2d = _compute_umap_2d(json.dumps(docs))

    # Build arrays for Plotly
    x = coords_2d[:, 0].tolist()
    y = coords_2d[:, 1].tolist()

    topic_ids = [d.get("topic_id", -1) for d in docs]
    unique_topics = sorted(set(topic_ids))

    # Color palette: teal family for real topics, grey for noise (-1)
    # Using a discrete palette so topic IDs map consistently across filter changes
    _TOPIC_COLORS = [
        "#2dd4bf", "#0d9488", "#f59e0b", "#8b5cf6",
        "#ec4899", "#3b82f6", "#10b981", "#f97316",
        "#a78bfa", "#34d399",
    ]

    fig = go.Figure()

    for topic_id in unique_topics:
        mask = [i for i, t in enumerate(topic_ids) if t == topic_id]
        color = (
            "#475569"  # slate for noise
            if topic_id == -1
            else _TOPIC_COLORS[topic_id % len(_TOPIC_COLORS)]
        )
        label = "Noise" if topic_id == -1 else f"Topic {topic_id}"

        fig.add_trace(go.Scatter(
            x=[x[i] for i in mask],
            y=[y[i] for i in mask],
            mode="markers",
            name=label,
            marker=dict(
                color=color,
                size=7,
                opacity=0.85,
                line=dict(width=0.5, color="#0f172a"),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Source: %{customdata[1]}<br>"
                "Sentiment: %{customdata[2]}<br>"
                "Lang: %{customdata[3]}<br>"
                "<extra></extra>"
            ),
            customdata=[
                [
                    (docs[i].get("title") or "Untitled")[:60],
                    docs[i].get("source", "?"),
                    docs[i].get("sentiment", "?"),
                    docs[i].get("detected_language", "?"),
                ]
                for i in mask
            ],
        ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
        legend=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Topic keyword reference below the chart
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Topic keywords (from BERTopic model)")
    st.caption("Keywords extracted by c-TF-IDF from the trained BERTopic model.")

    try:
        import joblib
        from pathlib import Path
        model = joblib.load(Path(__file__).parent.parent / "models" / "bertopic_model.joblib")
        real_topics = [t for t in unique_topics if t != -1]
        if real_topics:
            cols = st.columns(len(real_topics))
            for col, tid in zip(cols, real_topics):
                words_scores = model.get_topic(tid)
                if words_scores:
                    keywords = ", ".join(w for w, _ in words_scores[:6])
                    color = _TOPIC_COLORS[tid % len(_TOPIC_COLORS)]
                    col.markdown(
                        f'<div style="border-left: 3px solid {color}; padding-left: 8px;">'
                        f'<span style="color:{color}; font-family: JetBrains Mono, monospace; '
                        f'font-size:0.75rem; font-weight:600;">TOPIC {tid}</span><br>'
                        f'<span style="font-size:0.78rem; color:#94a3b8;">{keywords}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    except FileNotFoundError:
        st.caption("BERTopic model not found at models/bertopic_model.joblib — keywords unavailable.")

# ---------------------------------------------------------------------------
# Tab 3 — Contradiction Explorer (PyVis network)
# ---------------------------------------------------------------------------
def _render_tab3() -> None:
    """
    Render Tab 3: network graph of semantically similar articles with
    opposing sentiment, plus side-by-side article comparison.

    Data flow:
      GET /contradictions?days={d}&threshold={t}&max_pairs={n}
      → PyVis network rendered via st.components.v1.html()
      → click on edge label → side-by-side article detail
    """
    import json
    from pyvis.network import Network

    st.markdown("## Contradiction Explorer")
    st.caption(
        "Articles covering the same topic but with opposing sentiment. "
        "Edge thickness = cosine similarity. Green = positive, Red = negative."
    )

    col_days, col_thresh, col_pairs = st.columns(3)
    with col_days:
        days = st.slider("Lookback (days)", 1, 90, 30, key="tab3_days")
    with col_thresh:
        threshold = st.slider("Min similarity", 0.0, 1.0, 0.65, step=0.05, key="tab3_thresh")
    with col_pairs:
        max_pairs = st.slider("Max pairs", 1, 20, 5, key="tab3_pairs")

    data = _api_get(
        "/contradictions",
        params={"days": days, "threshold": threshold, "max_pairs": max_pairs},
    )
    if data is None:
        return

    pairs = data.get("pairs", [])

    if not pairs:
        st.info(
            f"No contradictions found above similarity {threshold} in the last {days} days. "
            "Try lowering the threshold or increasing the lookback window."
        )
        return

    st.markdown(
        f'<span class="mono" style="color:#64748b;">'
        f"{len(pairs)} contradiction pair(s) found"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Build PyVis network
    # Why bgcolor and font_color matching the dashboard dark theme:
    # PyVis renders inside an iframe — it has its own CSS context, so we
    # must pass the palette explicitly rather than inheriting from the parent.
    net = Network(
        height="450px",
        width="100%",
        bgcolor="#1e293b",
        font_color="#1e293b",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    seen_nodes: set[str] = set()

    for pair in pairs:
        neg = pair["negative_article"]
        pos = pair["positive_article"]
        similarity = pair["cosine_similarity"]

        neg_id = neg["url"]
        pos_id = pos["url"]

        neg_label = (neg.get("title") or neg["url"])[:35]
        pos_label = (pos.get("title") or pos["url"])[:35]

        if neg_id not in seen_nodes:
            net.add_node(
                neg_id,
                label=neg_label,
                color="#b91c1c",   # red-700 — negative sentiment
                title=f"[NEGATIVE] {neg.get('source','?')}\n{neg.get('subject','')}",
                size=18,
            )
            seen_nodes.add(neg_id)

        if pos_id not in seen_nodes:
            net.add_node(
                pos_id,
                label=pos_label,
                color="#0f766e",   # teal-700 — positive sentiment
                title=f"[POSITIVE] {pos.get('source','?')}\n{pos.get('subject','')}",
                size=18,
            )
            seen_nodes.add(pos_id)

        # Edge width proportional to cosine similarity (range 1–8px)
        edge_width = round(1 + similarity * 5, 1)
        net.add_edge(
            neg_id,
            pos_id,
            value=edge_width,
            title=f"Similarity: {similarity:.2f}",
            color="#2dd4bf",
        )

    # Disable the PyVis control panel — it adds visual noise with no benefit
    # at this corpus size
    net.set_options(json.dumps({
        "interaction": {"hover": True, "tooltipDelay": 100},
        "physics": {"enabled": True},
        "edges": {"smooth": {"type": "dynamic"}},
    }))

    html_str = net.generate_html()
    st.components.v1.html(html_str, height=470, scrolling=False)

    # Side-by-side article comparison
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Contradiction pairs — detail")

    for i, pair in enumerate(pairs):
        neg = pair["negative_article"]
        pos = pair["positive_article"]
        similarity = pair["cosine_similarity"]

        st.markdown(
            f'<span class="mono" style="color:#64748b;">pair {i+1} · similarity {similarity:.2f}</span>',
            unsafe_allow_html=True,
        )

        col_neg, col_pos = st.columns(2)

        with col_neg:
            st.markdown(
                f'<div class="article-card">'
                f'<div class="article-meta">'
                f'<span class="badge badge-negative">negative</span>'
                f'<span class="badge badge-source">{neg.get("source","?").replace("_"," ")}</span>'
                f'</div>'
                f'<div class="article-title">{neg.get("title","Untitled")}</div>'
                f'<div class="article-subject">{neg.get("subject","")}</div>'
                f'<a href="{neg["url"]}" target="_blank" '
                f'style="font-size:0.75rem;color:#2dd4bf;">↗ Read article</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_pos:
            st.markdown(
                f'<div class="article-card">'
                f'<div class="article-meta">'
                f'<span class="badge badge-positive">positive</span>'
                f'<span class="badge badge-source">{pos.get("source","?").replace("_"," ")}</span>'
                f'</div>'
                f'<div class="article-title">{pos.get("title","Untitled")}</div>'
                f'<div class="article-subject">{pos.get("subject","")}</div>'
                f'<div class="article-extract">{pos.get("main_argument","")}</div>'
                f'<a href="{pos["url"]}" target="_blank" '
                f'style="font-size:0.75rem;color:#2dd4bf;">↗ Read article</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")


# ---------------------------------------------------------------------------
# Tab 4 — Trends & Narrative Summary
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def _fetch_all_articles_for_trends() -> list[dict]:
    """
    Fetch all articles from CURATED for client-side sentiment aggregation.

    Why cache with ttl=1800 (30 min) and not ttl=3600:
    The sentiment chart aggregates by source and week — if the pipeline runs
    mid-session the chart should refresh within 30 minutes, not an hour.

    Why a standalone cached function and not calling _api_get() directly:
    _api_get() is not cacheable by st.cache_data because it references the
    st module internally (st.error). This wrapper isolates the HTTP call so
    the cache key is just the function identity + no args.
    """
    import requests
    try:
        resp = requests.get(
            f"{API_BASE_URL}/articles",
            params={"limit": 100, "skip": 0},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception:
        return []


def _render_tab4() -> None:
    """
    Render Tab 4: topic frequency trends (bar chart) + sentiment evolution
    (line chart) + latest narrative summary card.

    Data flow:
      GET /trends?days={d}     → grouped bar chart (topic frequency by week)
      GET /articles?limit=500  → line chart (sentiment ratio by source over time)
      GET /summary             → narrative summary card
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    st.markdown("## Trends & Narrative Summary")
    st.caption(
        "Topic frequency evolution, sentiment distribution by source, "
        "and the latest narrative generated by the Synthesis Agent."
    )

    # Temporal window selector — shared across all charts in this tab
    days = st.select_slider(
        "Analysis window",
        options=[7, 14, 30, 60, 90],
        value=30,
        key="tab4_days",
        format_func=lambda d: f"{d} days",
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Chart 1: Topic frequency by ISO week ────────────────────────────
    st.markdown("#### Topic frequency by week")

    trends_data = _api_get("/trends", params={"days": days})

    if trends_data:
        topic_trends = trends_data.get("topic_trends", {})
        rising = trends_data.get("rising_topics", [])

        if rising:
            rising_str = ", ".join(f"Topic {t}" for t in rising)
            st.markdown(
                f'<span class="mono" style="color:#2dd4bf;">↑ Rising: {rising_str}</span>',
                unsafe_allow_html=True,
            )

        if topic_trends:
            # Collect all unique ISO weeks across all topics
            all_weeks: list[int] = sorted({
                w["week"]
                for weekly_data in topic_trends.values()
                for w in weekly_data
            })

            _TOPIC_COLORS = [
                "#2dd4bf", "#0d9488", "#f59e0b", "#8b5cf6",
                "#ec4899", "#3b82f6", "#10b981", "#f97316",
            ]

            fig_freq = go.Figure()

            for topic_id_str, weekly_data in sorted(topic_trends.items()):
                tid = int(topic_id_str)
                week_to_count = {w["week"]: w["count"] for w in weekly_data}
                counts = [week_to_count.get(wk, 0) for wk in all_weeks]
                color = _TOPIC_COLORS[tid % len(_TOPIC_COLORS)]

                fig_freq.add_trace(go.Bar(
                    name=f"Topic {tid}",
                    x=[f"W{wk}" for wk in all_weeks],
                    y=counts,
                    marker_color=color,
                    opacity=0.85,
                ))

            fig_freq.update_layout(
                barmode="group",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
                margin=dict(l=20, r=20, t=10, b=20),
                height=320,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155", title="Articles"),
            )

            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.info("No topic trend data for this period.")
    else:
        st.warning("Could not load trend data.")

    # ── Chart 2: Sentiment ratio by source over time ─────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Sentiment ratio by source")
    st.caption("Fraction of negative articles per source over the selected period.")

    articles = _fetch_all_articles_for_trends()

    if articles:
        from datetime import datetime, timezone, timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Filter by period and group by (source, ISO week, sentiment)
        # Why ISO week: consistent with /trends endpoint granularity
        weekly_neg: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        weekly_total: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        all_sources: set[str] = set()

        for art in articles:
            ingestion_str = art.get("ingestion_date", "")
            try:
                ingestion_dt = datetime.fromisoformat(
                    ingestion_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                continue

            if ingestion_dt < cutoff:
                continue

            source = art.get("source", "unknown")
            week = ingestion_dt.isocalendar()[1]
            sentiment = art.get("sentiment", "neutral")

            all_sources.add(source)
            weekly_total[source][week] += 1
            if sentiment == "negative":
                weekly_neg[source][week] += 1

        all_weeks_sent: list[int] = sorted({
            wk
            for src_data in weekly_total.values()
            for wk in src_data.keys()
        })

        _SOURCE_COLORS = [
            "#2dd4bf", "#f59e0b", "#8b5cf6",
            "#ec4899", "#3b82f6", "#f97316", "#10b981",
        ]

        fig_sent = go.Figure()

        for i, source in enumerate(sorted(all_sources)):
            ratios = []
            for wk in all_weeks_sent:
                total = weekly_total[source].get(wk, 0)
                neg = weekly_neg[source].get(wk, 0)
                ratios.append(round(neg / total, 2) if total > 0 else None)

            fig_sent.add_trace(go.Scatter(
                name=source.replace("_", " "),
                x=[f"W{wk}" for wk in all_weeks_sent],
                y=ratios,
                mode="lines+markers",
                line=dict(color=_SOURCE_COLORS[i % len(_SOURCE_COLORS)], width=2),
                marker=dict(size=6),
                connectgaps=False,  # gaps where source had no articles that week
            ))

        fig_sent.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="JetBrains Mono, monospace"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
            margin=dict(l=20, r=20, t=10, b=20),
            height=320,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(
                gridcolor="#334155",
                title="Negative ratio",
                range=[0, 1],
                tickformat=".0%",
            ),
        )

        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No article data available for sentiment chart.")

    # ── Narrative summary card ────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Latest narrative summary")
    st.caption("Generated by the Synthesis Agent. Run the pipeline to refresh.")

    summary_data = _api_get("/summary")

    if summary_data is None:
        st.info("No summary yet. Run POST /pipeline/run first.")
        return

    provider = summary_data.get("llm_provider", "groq")
    timestamp = summary_data.get("timestamp", "")
    period = summary_data.get("period_days", "?")
    text = summary_data.get("text", "")
    insights = summary_data.get("analytical_insights", [])
    contradiction_count = summary_data.get("contradiction_count", 0)
    rising = summary_data.get("rising_topics", [])

    provider_icon = "⚡" if provider == "groq" else "✦"
    provider_label = "Groq LPU" if provider == "groq" else "Gemini Flash"

    # Metadata row
    col_ts, col_prov, col_period = st.columns(3)
    with col_ts:
        ts_display = timestamp[:10] if timestamp else "—"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="font-size:1.1rem;">{ts_display}</div>'
            f'<div class="metric-label">Generated</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_prov:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="font-size:1.1rem;">{provider_icon} {provider_label}</div>'
            f'<div class="metric-label">LLM Provider</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_period:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="font-size:1.1rem;">{period}d</div>'
            f'<div class="metric-label">Period analyzed</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Summary text
    st.markdown(
        f'<div class="article-card" style="border-color:#2dd4bf33;">'
        f'<div style="font-size:0.88rem; line-height:1.7; color:#cbd5e1;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Supporting stats below the text
    if insights or contradiction_count or rising:
        st.markdown("")
        col_ins, col_con, col_ris = st.columns(3)
        with col_ins:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(insights)}</div>'
                f'<div class="metric-label">Insights generated</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_con:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{contradiction_count}</div>'
                f'<div class="metric-label">Contradictions found</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_ris:
            rising_display = ", ".join(str(t) for t in rising) if rising else "—"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="font-size:1rem;">{rising_display}</div>'
                f'<div class="metric-label">Rising topics</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main — tab layout
# ---------------------------------------------------------------------------
def main() -> None:
    _render_sidebar()
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pipeline & Feed",
        "Topic Map",
        "Contradictions",
        "Trends & Summary",
    ])
 
    with tab1:
        _render_tab1()
 
    with tab2:
        _render_tab2()
 
    with tab3:
        _render_tab3()
 
    with tab4:
        _render_tab4()
 


if __name__ == "__main__":
    main()