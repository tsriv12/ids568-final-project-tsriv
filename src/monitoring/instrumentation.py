"""
Prometheus metrics instrumentation for the Mistral 7B RAG + Agent pipeline.
Wraps the three tools from agent_controller.py (Retriever, Summarizer, KeywordExtractor)
plus end-to-end pipeline metrics.

Real system values from Milestone 6:
  - Avg retrieval latency:   ~31ms
  - Avg generation latency:  ~3039ms (RAG), ~22434ms (Agent)
  - Tool match accuracy:     ~70-80%
"""

from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server
)
import time
import functools
import logging

logger = logging.getLogger(__name__)

# ── Request counters ──────────────────────────────────────────────────────────
REQUEST_TOTAL = Counter(
    "agent_requests_total",
    "Total pipeline requests (RAG or Agent)",
    ["pipeline_type", "status"]
)

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total tool invocations by type",
    ["tool_name", "status"]
)

# ── Latency histograms ────────────────────────────────────────────────────────
RETRIEVAL_LATENCY = Histogram(
    "agent_retrieval_duration_seconds",
    "ChromaDB retrieval latency (real M6 baseline ~31ms)",
    buckets=[0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0]
)

GENERATION_LATENCY = Histogram(
    "agent_generation_duration_seconds",
    "Mistral 7B generation latency (real M6 baseline ~3039ms)",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
)

E2E_LATENCY = Histogram(
    "agent_e2e_duration_seconds",
    "End-to-end pipeline latency (RAG ~3s, Agent ~22s)",
    buckets=[1.0, 3.0, 5.0, 10.0, 15.0, 22.0, 30.0, 60.0, 90.0]
)

TOOL_LATENCY = Histogram(
    "agent_tool_duration_seconds",
    "Per-tool execution latency",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.5, 1.0, 3.0, 5.0, 10.0]
)

# ── Gauges ────────────────────────────────────────────────────────────────────
ACTIVE_REQUESTS = Gauge(
    "agent_active_requests",
    "Requests currently in flight"
)

RETRIEVAL_SIMILARITY = Gauge(
    "agent_retrieval_top_similarity",
    "Top-1 cosine similarity of most recent retrieval (0-1). Real M6 baseline ~0.891"
)

TOOL_MATCH_ACCURACY = Gauge(
    "agent_tool_match_accuracy",
    "Rolling tool selection accuracy vs expected tools (0-1). Real M6 baseline ~0.75"
)

AGENT_STEPS = Gauge(
    "agent_steps_taken",
    "Number of tool steps taken in the most recent agent task"
)

INPUT_ANOMALY = Gauge(
    "agent_input_anomaly_flag",
    "1 if last query was anomalous (too short/long), else 0"
)

OLLAMA_MODEL_LOADED = Gauge(
    "agent_ollama_model_loaded",
    "1 if mistral:7b-instruct is loaded in Ollama, else 0"
)


def check_ollama_health():
    """Poll Ollama health and update the gauge."""
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        loaded = any("mistral" in n for n in names)
        OLLAMA_MODEL_LOADED.set(1 if loaded else 0)
    except Exception:
        OLLAMA_MODEL_LOADED.set(0)


def track_e2e(pipeline_type: str = "rag"):
    """Wraps a full pipeline call (RAG or Agent)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_TOTAL.labels(pipeline_type=pipeline_type, status="success").inc()
                return result
            except Exception as e:
                REQUEST_TOTAL.labels(pipeline_type=pipeline_type, status="error").inc()
                raise
            finally:
                E2E_LATENCY.observe(time.time() - t0)
                ACTIVE_REQUESTS.dec()
        return wrapper
    return decorator


def track_tool(tool_name: str):
    """Wraps an individual tool call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                TOOL_CALLS.labels(tool_name=tool_name, status="success").inc()
                return result
            except Exception as e:
                TOOL_CALLS.labels(tool_name=tool_name, status="error").inc()
                raise
            finally:
                TOOL_LATENCY.labels(tool_name=tool_name).observe(time.time() - t0)
        return wrapper
    return decorator


def record_retrieval_quality(top_similarity: float):
    RETRIEVAL_SIMILARITY.set(top_similarity)


def record_generation_latency(latency_ms: float):
    GENERATION_LATENCY.observe(latency_ms / 1000.0)


def record_agent_outcome(steps: int, tool_match: float):
    AGENT_STEPS.set(steps)
    TOOL_MATCH_ACCURACY.set(tool_match)


def record_input_features(query: str):
    tokens = len(query.split())
    INPUT_ANOMALY.set(1 if (tokens < 3 or tokens > 300) else 0)


def start_metrics_server(port: int = 8001):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")
