"""
Multi-agent system for the Climate & Energy Intelligence System.
 
Four LangGraph agents run in sequence, sharing a single typed state dict:
  1. Analytical Agent  — insights by source, language, and topic
  2. Contradiction Agent — pairs of articles with similar embeddings but opposite sentiment
  3. Trend Agent       — temporal evolution of topics and sentiment
  4. Synthesis Agent   — narrative summary combining all three outputs → SUMMARIES
 
Why LangGraph over plain LangChain:
    The Synthesis Agent needs the outputs of all three preceding agents simultaneously.
    In a LangChain sequential chain, that state would have to be passed manually between
    steps. LangGraph models the flow as a directed graph with a shared typed state dict
    that every node can read and write — the Synthesis Agent sees the full accumulated
    state without any manual wiring.
 
Why a single shared state and not separate function calls:
    Shared state makes the pipeline observable end-to-end: LangSmith traces show each
    node's input and output as part of a single run, not as four unrelated calls.
    It also means that if any node fails, the partial state is preserved and inspectable.
 
Why all agents use chat_complete() from shared/llm_client.py and not LangChain's
ChatGroq or ChatGoogleGenerativeAI:
    chat_complete() already implements the Groq → Gemini fallback with logging and
    json_mode support. Replacing it with LangChain's provider-specific classes would
    mean re-implementing the fallback logic inside LangGraph, duplicating code.
    The tradeoff: LangSmith won't see the individual LLM calls inside chat_complete()
    as "LLM" spans — they appear as Python function calls. The node-level tracing
    (which agent ran, inputs, outputs, timing) is fully preserved.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from shared.db import get_db, COL_CURATED, COL_SUMMARIES, insert_summary
from shared.llm_client import chat_complete

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing — activated by env vars, no code changes needed
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls__...
# LANGCHAIN_PROJECT=rag-climate
# ---------------------------------------------------------------------------
_ANALYSIS_DAYS = 7  # default lookback window for all agents

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    Shared state dict passed through every node in the LangGraph.
 
    Why TypedDict and not a dataclass or Pydantic model:
    LangGraph requires the state to be a plain dict-compatible structure so
    it can serialize it for checkpointing and LangSmith tracing. TypedDict
    provides type hints without adding a runtime serialization layer.
 
    Fields populated by each agent:
    - analytical_insights: dict produced by the Analytical Agent
    - contradictions: list produced by the Contradiction Agent
    - trends: dict produced by the Trend Agent
    - summary: str produced by the Synthesis Agent
    - errors: list of error strings — agents append here on failure instead
      of raising, so the pipeline continues even if one agent fails
    - days: lookback window in days, set at graph invocation time
    """
    days: int
    analytical_insights: dict[str, Any]
    contradictions: list[dict[str, Any]]
    trends: dict[str, Any]
    summary: str
    errors: list[str]

# ---------------------------------------------------------------------------
# Database tools — synchronous wrappers over motor (used inside sync agents)
# ---------------------------------------------------------------------------
# Why synchronous wrappers here and not async agents:
# LangGraph's StateGraph.invoke() is synchronous. Running async motor calls
# inside a sync context requires asyncio.run(), which conflicts with an
# already-running event loop (e.g. when called from FastAPI's async pipeline
# generator). The solution: use pymongo (synchronous) for the agent tools,
# reserving motor for the FastAPI/pipeline layer. Both point to the same
# MongoDB Atlas cluster via MONGODB_URI.
import pymongo

def _get_sync_db() -> pymongo.database.Database:  # type: ignore[type-arg]
    """
    Return a synchronous pymongo database handle.
 
    Why a separate sync client and not reusing shared/db.py's motor client:
    motor's AsyncIOMotorClient cannot be used from synchronous code without
    asyncio.run(), which fails inside an already-running event loop (FastAPI).
    pymongo's MongoClient is purely synchronous and safe to call from any context.
    Both clients connect to the same MONGODB_URI and see the same data.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI environment variable is not set.")
    client: pymongo.MongoClient = pymongo.MongoClient(uri)  # type: ignore[type-arg]
    return client.get_default_database()
 
 
def _cutoff_date(days: int) -> datetime:
    """Return UTC datetime for 'days' ago — used to filter recent articles."""
    return datetime.now(timezone.utc) - timedelta(days=days)

# ---------------------------------------------------------------------------
# Agent 1: Analytical Agent
# ---------------------------------------------------------------------------

def _query_by_source(db: Any, days: int) -> dict[str, Any]:
    """
    Count articles and compute average sentiment intensity per source.
 
    Why average intensity and not just count:
    Count alone tells us which source is most prolific. Average intensity
    tells us which source is most alarmist or most optimistic — a more
    interesting signal for the narrative summary.
    """
    cutoff = _cutoff_date(days)
    pipeline = [
        {"$match": {"ingestion_date": {"$gte": cutoff.isoformat()}}},
        {"$group": {
            "_id": "$source",
            "count": {"$sum": 1},
            "avg_intensity": {"$avg": "$intensity"},
            "sentiments": {"$push": "$sentiment"},
        }},
        {"$sort": {"count": -1}},
    ]
    results = list(db[COL_CURATED].aggregate(pipeline))
    return {
        r["_id"]: {
            "count": r["count"],
            "avg_intensity": round(r.get("avg_intensity") or 0, 3),
            "dominant_sentiment": max(
                ["positive", "negative", "neutral"],
                key=lambda s: r["sentiments"].count(s),
            ),
        }
        for r in results
    }

def _query_by_language(db: Any, days: int) -> dict[str, Any]:
    """Count articles and sentiment distribution per detected language."""
    cutoff = _cutoff_date(days)
    pipeline = [
        {"$match": {"ingestion_date": {"$gte": cutoff.isoformat()}}},
        {"$group": {
            "_id": "$detected_language",
            "count": {"$sum": 1},
            "positive": {"$sum": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}},
            "negative": {"$sum": {"$cond": [{"$eq": ["$sentiment", "negative"]}, 1, 0]}},
            "neutral":  {"$sum": {"$cond": [{"$eq": ["$sentiment", "neutral"]},  1, 0]}},
        }},
        {"$sort": {"count": -1}},
    ]
    results = list(db[COL_CURATED].aggregate(pipeline))
    return {
        r["_id"]: {
            "count": r["count"],
            "positive": r["positive"],
            "negative": r["negative"],
            "neutral": r["neutral"],
        }
        for r in results
    }

def _query_by_topic(db: Any, days: int) -> dict[str, Any]:
    """Count articles and average sentiment intensity per BERTopic topic_id."""
    cutoff = _cutoff_date(days)
    pipeline = [
        {"$match": {
            "ingestion_date": {"$gte": cutoff.isoformat()},
            "topic_id": {"$exists": True, "$ne": -1},  # exclude noise
        }},
        {"$group": {
            "_id": "$topic_id",
            "count": {"$sum": 1},
            "avg_intensity": {"$avg": "$intensity"},
            "dominant_sentiment": {"$push": "$sentiment"},
        }},
        {"$sort": {"count": -1}},
    ]
    results = list(db[COL_CURATED].aggregate(pipeline))
    return {
        str(r["_id"]): {
            "count": r["count"],
            "avg_intensity": round(r.get("avg_intensity") or 0, 3),
            "dominant_sentiment": max(
                ["positive", "negative", "neutral"],
                key=lambda s: r["dominant_sentiment"].count(s),
            ),
        }
        for r in results
    }

def analytical_agent(state: AgentState) -> AgentState:
    """
    Node 1: gather structured insights from CURATED by source, language, topic.
 
    The LLM is used here to interpret the aggregated stats and produce
    human-readable insights — not to query the database. The database
    queries run first (pure Python/MongoDB), then the LLM synthesizes.
 
    Why generate insights with the LLM instead of just returning the raw stats:
    The raw stats (counts, averages) are already computed and stored. The LLM
    adds the interpretive layer: "Carbon Brief is the most negative source
    this week, driven by wildfire coverage" is more useful than
    {"carbon_brief": {"count": 12, "avg_intensity": 0.78, ...}}.
    """
    logger.info("Analytical Agent: starting.")
    days = state["days"]
 
    try:
        db = _get_sync_db()
        by_source = _query_by_source(db, days)
        by_language = _query_by_language(db, days)
        by_topic = _query_by_topic(db, days)

        total_articles = sum(v["count"] for v in by_source.values())

        prompt = f"""You are a climate and energy journalism analyst.
        Based on the following aggregated statistics from the last {days} days, produce 3-5 concise insights.
        Each insight must be a single sentence that identifies a notable pattern, trend, or anomaly.
 
        Statistics by source: {by_source}
        Statistics by language: {by_language}
        Statistics by topic: {by_topic}
 
        Respond ONLY with a JSON object with this structure:
        {{
        "insights": ["insight 1", "insight 2", "insight 3"],
        "most_active_source": "source_name",
        "dominant_sentiment_overall": "positive|negative|neutral",
        "total_articles_analyzed": {total_articles}
        }}"""
 
        response = chat_complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )
 
        import json
        parsed = json.loads(response["content"])
        parsed["raw_stats"] = {
            "by_source": by_source,
            "by_language": by_language,
            "by_topic": by_topic,
        }
 
        state["analytical_insights"] = parsed
        logger.info(
            "Analytical Agent: done. %d insights generated via %s.",
            len(parsed.get("insights", [])),
            response["provider"],
        )
 
    except Exception as exc:
        logger.error("Analytical Agent failed: %s", exc, exc_info=True)
        state["errors"].append(f"analytical_agent: {exc}")
        state["analytical_insights"] = {}
 
    return state

# ---------------------------------------------------------------------------
# Agent 2: Contradiction Agent
# ---------------------------------------------------------------------------
def _find_contradictions(
    db: Any,
    days: int,
    threshold: float = 0.65,
    max_pairs: int = 5,
) -> list[dict[str, Any]]:
    """
    Find pairs of articles with similar embeddings but opposite sentiment
    using MongoDB Atlas Vector Search on the CURATED collection.
 
    Why threshold=0.65 and not 0.85 as stated in the master plan:
    0.85 is very strict — two articles must be almost identical in semantic
    content. In practice, with 222 articles across 3 languages, 0.85 often
    returns zero pairs. 0.65 captures articles covering the same topic
    (e.g. offshore wind) with genuinely opposite framings — which is the
    signal we want. This is configurable via the threshold parameter.
 
    Why iterate over negative articles and search for positive neighbors
    (not the reverse):
    Negative articles dominate the corpus (66.7% per the benchmark). Starting
    from negatives and searching for positive neighbors maximizes the chance
    of finding contradictions without scanning the full corpus.
 
    Why max_pairs=5:
    The Synthesis Agent's prompt would become too long with more pairs.
    5 pairs is enough to illustrate the contradiction pattern narratively.
    """
    cutoff = _cutoff_date(days)
 
    # Fetch negative articles with embeddings from the period
    negative_articles = list(db[COL_CURATED].find(
        {
            "sentiment": "negative",
            "embedding": {"$exists": True},
            "ingestion_date": {"$gte": cutoff.isoformat()},
        },
        {"url": 1, "title": 1, "embedding": 1, "source": 1, "principal_subject": 1},
        limit=20,  # check up to 20 negative articles
    ))
 
    pairs: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
 
    for neg_article in negative_articles:
        if len(pairs) >= max_pairs:
            break
 
        embedding = neg_article.get("embedding")
        if not embedding:
            continue
 
        # Vector Search for semantically similar articles
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 50,
                    "limit": 10,
                }
            },
            {
                "$match": {
                    "sentiment": "positive",
                    "url": {"$ne": neg_article["url"]},
                    "ingestion_date": {"$gte": cutoff.isoformat()},
                }
            },
            {
                "$project": {
                    "url": 1,
                    "title": 1,
                    "source": 1,
                    "principal_subject": 1,
                    "main_argument": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
 
        candidates = list(db[COL_CURATED].aggregate(pipeline))
 
        for candidate in candidates:
            score = candidate.get("score", 0)
            if score < threshold:
                continue
 
            pair_key = tuple(sorted([neg_article["url"], candidate["url"]]))
            if pair_key in seen_urls:
                continue
            seen_urls.add(str(pair_key))
 
            pairs.append({
                "negative_article": {
                    "url": neg_article["url"],
                    "title": neg_article.get("title", ""),
                    "source": neg_article.get("source", ""),
                    "subject": neg_article.get("principal_subject", ""),
                },
                "positive_article": {
                    "url": candidate["url"],
                    "title": candidate.get("title", ""),
                    "source": candidate.get("source", ""),
                    "subject": candidate.get("principal_subject", ""),
                    "main_argument": candidate.get("main_argument", ""),
                },
                "cosine_similarity": round(score, 4),
            })
            break  # one pair per negative article
 
    return pairs
 
 
def contradiction_agent(state: AgentState) -> AgentState:
    """
    Node 2: detect pairs of articles with similar topic but opposite sentiment.
 
    Uses MongoDB Atlas Vector Search on CURATED (which already has embeddings
    inherited from CLEAN) — no need to query CLEAN separately.
    """
    logger.info("Contradiction Agent: starting.")
    days = state["days"]
 
    try:
        db = _get_sync_db()
        pairs = _find_contradictions(db, days)
 
        if not pairs:
            logger.info("Contradiction Agent: no contradictions found above threshold.")
            state["contradictions"] = []
            return state
 
        # Ask the LLM to interpret the contradiction pairs
        prompt = f"""You are a media analyst specializing in climate and energy journalism.
        The following pairs of articles cover similar topics but express opposite sentiments.
        Briefly explain what makes each pair contradictory and what it reveals about media framing.
        
        Pairs found:
        {pairs}
        
        Respond ONLY with a JSON object:
        {{
        "contradiction_count": <integer>,
        "interpretation": "1-2 sentence overall interpretation of the contradictions found",
        "pairs": [
            {{
            "topic": "brief topic description",
            "framing_conflict": "one sentence explaining the conflict"
            }}
        ]
        }}"""
        
        response = chat_complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )
 
        import json
        parsed = json.loads(response["content"])
        parsed["raw_pairs"] = pairs
 
        state["contradictions"] = parsed
        logger.info(
            "Contradiction Agent: done. %d pairs found via %s.",
            len(pairs),
            response["provider"],
        )
 
    except Exception as exc:
        logger.error("Contradiction Agent failed: %s", exc, exc_info=True)
        state["errors"].append(f"contradiction_agent: {exc}")
        state["contradictions"] = []
 
    return state

# ---------------------------------------------------------------------------
# Agent 3: Trend Agent
# ---------------------------------------------------------------------------
def _compute_topic_trend(db: Any, days: int) -> dict[str, Any]:
    """
    Compute weekly frequency and average sentiment intensity per topic.
 
    Why $dateFromString and not $toDate:
    ingestion_date is stored as an ISO string (set by scraper.py with
    datetime.now(timezone.utc).isoformat()). $dateFromString parses it
    correctly. $toDate would fail on string input.
 
    Why week-level granularity and not day-level:
    With ~30 articles/week across 6 sources, day-level granularity produces
    sparse data (many zeros). Week-level aggregation produces meaningful
    frequency signals even with a small corpus.
    """
    cutoff = _cutoff_date(days)
    pipeline = [
        {"$match": {
            "ingestion_date": {"$gte": cutoff.isoformat()},
            "topic_id": {"$exists": True, "$ne": -1},
        }},
        {"$addFields": {
            "parsed_date": {"$dateFromString": {"dateString": "$ingestion_date"}},
        }},
        {"$group": {
            "_id": {
                "topic_id": "$topic_id",
                "week": {"$isoWeek": "$parsed_date"},
            },
            "count": {"$sum": 1},
            "avg_intensity": {"$avg": "$intensity"},
            "sentiments": {"$push": "$sentiment"},
        }},
        {"$sort": {"_id.week": 1}},
    ]
    results = list(db[COL_CURATED].aggregate(pipeline))
 
    # Restructure: {topic_id: [{week, count, avg_intensity, dominant_sentiment}]}
    trends: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        tid = str(r["_id"]["topic_id"])
        sentiments = r["sentiments"]
        dominant = max(
            ["positive", "negative", "neutral"],
            key=lambda s: sentiments.count(s),
        )
        trends.setdefault(tid, []).append({
            "week": r["_id"]["week"],
            "count": r["count"],
            "avg_intensity": round(r.get("avg_intensity") or 0, 3),
            "dominant_sentiment": dominant,
        })
 
    return trends
 
 
def _detect_rising_topics(trends: dict[str, list[dict[str, Any]]]) -> list[str]:
    """
    Identify topics whose weekly article count is increasing.
 
    Why simple delta (last week vs previous week) and not linear regression:
    With a 7-day window there are at most 1-2 data points per topic — linear
    regression over 2 points is mathematically equivalent to a delta comparison
    but adds unnecessary complexity. Linear regression becomes meaningful at
    30+ day windows (the master plan mentions it for longer periods).
    """
    rising = []
    for topic_id, weekly_data in trends.items():
        if len(weekly_data) < 2:
            continue
        last = weekly_data[-1]["count"]
        prev = weekly_data[-2]["count"]
        if last > prev:
            rising.append(topic_id)
    return rising
 
 
def trend_agent(state: AgentState) -> AgentState:
    """
    Node 3: track topic frequency and sentiment evolution over the period.
    """
    logger.info("Trend Agent: starting.")
    days = state["days"]
 
    try:
        db = _get_sync_db()
        topic_trends = _compute_topic_trend(db, days)
        rising_topics = _detect_rising_topics(topic_trends)
 
        prompt = f"""You are a climate journalism trend analyst.
        Based on the following weekly topic data from the last {days} days, identify 2-3 notable trends.
        
        Topic trends (topic_id → weekly data): {topic_trends}
        Rising topics (increasing frequency): {rising_topics}
        
        Respond ONLY with a JSON object:
        {{
        "trend_summary": "1-2 sentence overall trend description",
        "rising_topics": {rising_topics},
        "notable_trends": ["trend 1", "trend 2"],
        "sentiment_shift": "one sentence about overall sentiment direction"
        }}"""
        
        response = chat_complete(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )
 
        import json
        parsed = json.loads(response["content"])
        parsed["raw_trends"] = topic_trends
 
        state["trends"] = parsed
        logger.info(
            "Trend Agent: done. %d rising topics via %s.",
            len(rising_topics),
            response["provider"],
        )
 
    except Exception as exc:
        logger.error("Trend Agent failed: %s", exc, exc_info=True)
        state["errors"].append(f"trend_agent: {exc}")
        state["trends"] = {}
 
    return state

# ---------------------------------------------------------------------------
# Agent 4: Synthesis Agent
# ---------------------------------------------------------------------------
 
_SYNTHESIS_SYSTEM_PROMPT = """You are an expert science communicator specializing in climate and energy journalism.
Your task is to produce a clear, structured narrative summary of the current climate and energy discourse
based on analysis from three specialized agents: an analytical agent, a contradiction detector, and a trend tracker.
Write for an informed general audience. Be precise, avoid alarmism, and highlight genuine tensions in the discourse.
Respond in the same language as the majority of the articles analyzed (Spanish, English, or French)."""
 
 
def synthesis_agent(state: AgentState) -> AgentState:
    """
    Node 4: generate a narrative summary from the three preceding agents' outputs.
 
    Why the Synthesis Agent runs last and not in parallel:
    It needs the outputs of all three agents as context. Running it in parallel
    would mean it has no inputs to synthesize. LangGraph's sequential graph
    ensures it only runs after the other three nodes have completed.
 
    Why write to MongoDB SUMMARIES here and not in the FastAPI endpoint:
    The agent is responsible for its own side effects. The FastAPI endpoint
    only triggers the graph and streams progress — it should not have
    database write logic for agent outputs.
    """
    logger.info("Synthesis Agent: starting.")
    days = state["days"]
 
    analytical = state.get("analytical_insights", {})
    contradictions = state.get("contradictions", {})
    trends = state.get("trends", {})
 
    try:
        insights = analytical.get("insights", [])
        contradiction_interp = (
            contradictions.get("interpretation", "No contradictions detected.")
            if isinstance(contradictions, dict) else "No contradictions detected."
        )
        trend_summary = trends.get("trend_summary", "No trend data available.")
        notable_trends = trends.get("notable_trends", [])
 
        user_prompt = f"""Period analyzed: last {days} days
 
        ANALYTICAL INSIGHTS:
        {chr(10).join(f"- {i}" for i in insights) if insights else "No insights available."}
        
        CONTRADICTION ANALYSIS:
        {contradiction_interp}
        
        TREND ANALYSIS:
        {trend_summary}
        Notable trends: {notable_trends}
        
        Please produce a narrative summary (200-350 words) that integrates all three analyses into a coherent
        picture of the current climate and energy discourse. Structure it with:
        1. Overall picture (2-3 sentences)
        2. Key tensions and contradictions (2-3 sentences)
        3. Emerging trends (2-3 sentences)
        4. Closing observation (1-2 sentences)"""
 
        response = chat_complete(
            [
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=False,  # narrative text, not JSON
        )
 
        summary_text = response["content"]
        llm_provider = response["provider"]
 
        # Write to SUMMARIES collection via motor (called from async context
        # in FastAPI) or via direct pymongo insert here when run standalone.
        # For standalone runs we use pymongo directly.
        db = _get_sync_db()
        db[COL_SUMMARIES].insert_one({
            "text": summary_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_provider": llm_provider,
            "period_days": days,
            "analytical_insights": analytical.get("insights", []),
            "contradiction_count": (
                contradictions.get("contradiction_count", 0)
                if isinstance(contradictions, dict) else 0
            ),
            "rising_topics": trends.get("rising_topics", []),
        })
 
        state["summary"] = summary_text
        logger.info(
            "Synthesis Agent: done. Summary written to SUMMARIES via %s.",
            llm_provider,
        )
 
    except Exception as exc:
        logger.error("Synthesis Agent failed: %s", exc, exc_info=True)
        state["errors"].append(f"synthesis_agent: {exc}")
        state["summary"] = ""
 
    return state

 # ---------------------------------------------------------------------------
# LangGraph graph definition
# ---------------------------------------------------------------------------
 
def build_agent_graph() -> Any:
    """
    Build and compile the LangGraph directed graph.
 
    Graph structure:
        analytical_agent → contradiction_agent → trend_agent → synthesis_agent → END
 
    Why sequential and not parallel for agents 1-3:
    The corpus is small (~222 articles) and the database queries are fast (<1s each).
    Parallelism would add complexity (async nodes, thread safety for pymongo client)
    with negligible time savings. Sequential execution also makes LangSmith traces
    easier to read — each node's timing is clearly separated.
 
    Why StateGraph(AgentState) and not MessageGraph:
    MessageGraph is designed for chatbot-style conversations where the state
    is a list of messages. Our state is a structured dict of agent outputs —
    StateGraph with a TypedDict is the correct abstraction.
    """
    graph = StateGraph(AgentState)
 
    graph.add_node("analytical_agent", analytical_agent)
    graph.add_node("contradiction_agent", contradiction_agent)
    graph.add_node("trend_agent", trend_agent)
    graph.add_node("synthesis_agent", synthesis_agent)
 
    graph.set_entry_point("analytical_agent")
    graph.add_edge("analytical_agent", "contradiction_agent")
    graph.add_edge("contradiction_agent", "trend_agent")
    graph.add_edge("trend_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)
 
    return graph.compile()

# ---------------------------------------------------------------------------
# Public entry point — called by FastAPI pipeline generator
# ---------------------------------------------------------------------------
 
def run_agents(days: int = _ANALYSIS_DAYS) -> dict[str, Any]:
    """
    Run all four agents and return the final state.
 
    This is a synchronous function intentionally — it is called from
    FastAPI's async generator via asyncio.get_event_loop().run_in_executor()
    to avoid blocking the event loop during the ~20s agent run.
 
    Parameters
    ----------
    days:
        Lookback window in days. Default 7.
 
    Returns
    -------
    dict with keys: analytical_insights, contradictions, trends, summary, errors
    """
    graph = build_agent_graph()
 
    initial_state: AgentState = {
        "days": days,
        "analytical_insights": {},
        "contradictions": [],
        "trends": {},
        "summary": "",
        "errors": [],
    }
 
    logger.info("Running agent graph for last %d days...", days)
    final_state = graph.invoke(initial_state)
    logger.info(
        "Agent graph complete. Errors: %s",
        final_state.get("errors") or "none",
    )
    return final_state

# ---------------------------------------------------------------------------
# Standalone entrypoint (called by Docker CMD: python -m agent_worker.agents)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    result = run_agents(days=_ANALYSIS_DAYS)
    print("\n=== AGENT RUN COMPLETE ===")
    print(f"Insights: {result['analytical_insights'].get('insights', [])}")
    print(f"Contradictions found: {result['contradictions'].get('contradiction_count', 0) if isinstance(result['contradictions'], dict) else 0}")
    print(f"Trends: {result['trends'].get('trend_summary', 'N/A')}")
    print(f"Summary length: {len(result['summary'])} chars")
    print(f"Errors: {result['errors'] or 'none'}")