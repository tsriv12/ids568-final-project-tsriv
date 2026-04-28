"""
Traffic simulator for the M6 RAG + Agent pipeline.
Calls the REAL agent_controller tools via Ollama + ChromaDB.

Run from repo root:
  cd /home/tanyasrivastava/ids568-final-project-tsriv
  python src/monitoring/traffic_simulator.py
"""

import sys
import os
import time
import random
import logging
from pathlib import Path

# Make agent_controller and instrumentation importable regardless of where repo is cloned
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "monitoring"))

from instrumentation import (
    track_e2e, record_retrieval_quality,
    record_generation_latency, record_input_features,
    start_metrics_server, check_ollama_health
)
import agent_controller as ac

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUERY_POOL = [
    "What is retrieval-augmented generation and what are the key metrics used to evaluate it?",
    "Compare FAISS and ChromaDB as vector databases.",
    "How does data drift affect model performance in production?",
    "What chunking strategies exist for RAG systems?",
    "How does vLLM improve LLM serving throughput?",
    "What is the role of a feature store in MLOps?",
    "Explain the ReAct framework for agent tool selection.",
    "What embedding models are used for semantic search?",
    "Walk through the complete RAG pipeline steps.",
    "What MLOps tools are used for experiment tracking?",
    "What is RLHF and how does it improve LLMs?",
    "How do you monitor for concept drift in a deployed model?",
    "Explain continuous batching in LLM inference servers.",
    "hi",
    " ".join(["word"] * 400),
]


@track_e2e(pipeline_type="rag")
def run_rag_query(query: str):
    """Calls real tool_retriever + Mistral generation."""
    record_input_features(query)

    # Retrieve
    t0 = time.time()
    retrieval_result = ac.tool_retriever(query, k=3)
    retrieval_ms = (time.time() - t0) * 1000
    record_retrieval_quality(retrieval_result["top_similarity"])

    # Generate
    import ollama
    context = " ".join([c["content"] for c in retrieval_result["chunks"][:2]])
    prompt = f"Using this context: {context}\n\nAnswer concisely: {query}"
    t1 = time.time()
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 150}
    )
    gen_ms = (time.time() - t1) * 1000
    record_generation_latency(gen_ms)
    return response["message"]["content"]


def run_simulation(duration_seconds: int = 180, qps: float = 0.2):
    """
    Run RAG traffic simulation.
    Args:
        duration_seconds: How long to run (default 3 minutes)
        qps: Queries per second (0.2 = one query every 5 seconds)
    """
    start_metrics_server(port=8001)
    check_ollama_health()

    logger.info(f"Starting simulation for {duration_seconds}s at {qps} QPS")
    logger.info("Metrics: http://localhost:8001/metrics")

    end_time = time.time() + duration_seconds
    count = 0

    while time.time() < end_time:
        query = random.choice(QUERY_POOL)
        try:
            result = run_rag_query(query)
            count += 1
            logger.info(f"[{count}] OK: {query[:60]}...")
        except Exception as e:
            logger.error(f"[{count}] Error: {e}")

        check_ollama_health()
        time.sleep(1.0 / qps)

    logger.info(f"Simulation complete. {count} queries processed.")


if __name__ == "__main__":
    run_simulation(duration_seconds=600, qps=0.2)
