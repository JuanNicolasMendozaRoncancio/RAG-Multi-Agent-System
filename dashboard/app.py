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

load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
 
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
# Main — tab layout
# ---------------------------------------------------------------------------
def main() -> None:
    _render_sidebar()
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Pipeline & Feed",
        "🗺 Topic Map",
        "⚡ Contradictions",
        "📈 Trends & Summary",
    ])
 
    with tab1:
        _render_tab1()
 
    with tab2:
        st.info("Tab 2 — UMAP 2D topic map. Implemented in Step 15.")
 
    with tab3:
        st.info("Tab 3 — Contradiction network (PyVis). Implemented in Step 15.")
 
    with tab4:
        st.info("Tab 4 — Trends & narrative summary. Implemented in Step 15.")
 
 
if __name__ == "__main__":
    main()