"""
Incident Simulation Script — C1 Challenge Extension
Simulates a production incident where Ollama goes down, causing:
  1. agent_ollama_model_loaded drops to 0 (fires OllamaModelDown alert)
  2. Requests start failing (fires ZeroThroughput alert)
  3. Recovery is detected when Ollama comes back up

Run from repo root:
  python src/monitoring/simulate_incident.py
"""

import sys
import time
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "monitoring"))

from instrumentation import (
    start_metrics_server,
    OLLAMA_MODEL_LOADED,
    REQUEST_TOTAL,
    RETRIEVAL_SIMILARITY,
    E2E_LATENCY,
    record_input_features
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def simulate_normal_traffic(duration_seconds: int = 30):
    """Simulate healthy traffic for baseline."""
    import random
    logger.info(f"[NORMAL] Simulating {duration_seconds}s of healthy traffic...")
    end = time.time() + duration_seconds
    count = 0
    while time.time() < end:
        OLLAMA_MODEL_LOADED.set(1)
        REQUEST_TOTAL.labels(pipeline_type="rag", status="success").inc()
        RETRIEVAL_SIMILARITY.set(round(random.uniform(0.75, 0.92), 3))
        E2E_LATENCY.observe(random.uniform(2.5, 4.0))
        count += 1
        logger.info(f"[NORMAL] Request {count} OK — similarity={RETRIEVAL_SIMILARITY._value.get():.3f}")
        time.sleep(3)


def simulate_incident(duration_seconds: int = 60):
    """
    Simulate Ollama crash incident.
    - Sets ollama_model_loaded to 0
    - Requests start failing
    - This would fire OllamaModelDown + ZeroThroughput alerts in production
    """
    logger.info("=" * 60)
    logger.info("[INCIDENT] Simulating Ollama crash!")
    logger.info("[INCIDENT] agent_ollama_model_loaded -> 0")
    logger.info("[INCIDENT] This would fire: OllamaModelDown (critical)")
    logger.info("=" * 60)

    end = time.time() + duration_seconds
    count = 0
    while time.time() < end:
        # Ollama is down
        OLLAMA_MODEL_LOADED.set(0)
        # Requests are now failing
        REQUEST_TOTAL.labels(pipeline_type="rag", status="error").inc()
        count += 1
        logger.warning(f"[INCIDENT] Request {count} FAILED — Ollama down, error count rising")
        time.sleep(3)

    logger.info(f"[INCIDENT] {count} requests failed during incident window")


def simulate_recovery():
    """Simulate Ollama coming back up — alert resolves."""
    logger.info("=" * 60)
    logger.info("[RECOVERY] Ollama restarted via systemd watchdog")
    logger.info("[RECOVERY] agent_ollama_model_loaded -> 1")
    logger.info("[RECOVERY] OllamaModelDown alert resolves")
    logger.info("=" * 60)

    OLLAMA_MODEL_LOADED.set(1)
    for i in range(5):
        REQUEST_TOTAL.labels(pipeline_type="rag", status="success").inc()
        RETRIEVAL_SIMILARITY.set(0.88)
        logger.info(f"[RECOVERY] Request {i+1} OK — system restored")
        time.sleep(2)


def main():
    logger.info("Starting incident simulation...")
    logger.info("Metrics: http://localhost:8001/metrics")
    logger.info("Watch: agent_ollama_model_loaded and agent_requests_total{status=error}")

    start_metrics_server(port=8001)
    time.sleep(2)

    # Phase 1: Normal traffic (30s)
    simulate_normal_traffic(duration_seconds=30)

    # Phase 2: Incident — Ollama crash (60s)
    simulate_incident(duration_seconds=60)

    # Phase 3: Recovery
    simulate_recovery()

    logger.info("Incident simulation complete.")
    logger.info("In production: OllamaModelDown alert would have fired within 1 minute of crash.")
    logger.info("Mitigation: systemd watchdog auto-restarts Ollama within 30 seconds.")


if __name__ == "__main__":
    main()
