# ids568-final-project-tsriv
**MLOps Final Project — Module 8**
**Author:** Tanya Srivastava | tsriv
**Base System:** Milestone 6 — Mistral 7B RAG + Agentic Controller
**Stack:** Ollama · mistral:7b-instruct · ChromaDB · all-MiniLM-L6-v2 · Prometheus · Grafana
**Hardware:** GCP VM · NVIDIA Tesla T4 · CUDA 12.2 · 100% local inference (no API keys)

## System Overview

This project builds a production operations framework around the Milestone 6 RAG pipeline.
The base system answers MLOps questions by retrieving from a 10-document ChromaDB knowledge
base and generating responses with Mistral 7B via Ollama. A multi-tool agent handles
complex multi-step queries using three tools: Retriever, Summarizer, KeywordExtractor.

Pipeline:
User Query -> all-MiniLM-L6-v2 -> ChromaDB (mlops_rag) -> Mistral 7B (Ollama) -> Response
                                                                 |
                                         Agent: tool selection loop (max 4 steps)

## Component Links

| Component | Description | Key Files |
|---|---|---|
| C1: Monitoring | Prometheus + Grafana wrapping real Ollama calls | [Instrumentation](src/monitoring/instrumentation.py) · [Simulator](src/monitoring/traffic_simulator.py) · [Interpretation](docs/dashboard-interpretation.md) |
| C2: A/B Test | chunk_size=256 vs 512 on mlops_rag corpus | [Spec](docs/experiment-specification.md) · [Script](src/ab_test/simulation.py) · [Memo](docs/recommendation-memo.md) |
| C3: Governance | mistral:7b-instruct model card + audit trail | [Model Card](docs/model-card.md) · [Risk Register](docs/risk-register.md) · [Audit Trail](logs/audit-trail.json) · [Lineage](docs/lineage-diagram.png) |
| C4: Drift Detection | 4-week PSI analysis from M6 baselines | [Script](src/drift/drift_detection.py) · [Report](docs/drift-diagnostic-report.md) |
| C5: Risk Assessment | System boundary + governance review + CTO memo | [Governance](docs/governance-review.md) · [Risk Matrix](docs/risk-matrix.md) · [CTO Memo](docs/cto-memo.md) · [Diagram](docs/system-boundary-diagram.png) |

## Setup and Reproduction

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- GCP VM with NVIDIA T4 GPU
- Ollama installed with mistral:7b-instruct pulled

### 1. Clone and install
    git clone https://github.com/tsriv12/ids568-final-project-tsriv.git
    cd ids568-final-project-tsriv
    source /path/to/milestone6/venv/bin/activate
    pip install -r requirements.txt

### 2. Build ChromaDB index
    jupyter nbconvert --to notebook --execute rag_pipeline.ipynb
      --output rag_pipeline.ipynb
      --ExecutePreprocessor.timeout=600
      --ExecutePreprocessor.kernel_name=milestone6

### 3. Run monitoring dashboard
    cd dashboards && docker compose up -d && cd ..
    python src/monitoring/traffic_simulator.py

### 4. Run A/B simulation
    python src/ab_test/simulation.py --dry-run
    python src/ab_test/simulation.py

### 5. Run drift detection
    python src/drift/drift_detection.py

### 6. Generate diagrams
    python src/monitoring/generate_diagrams.py

## Lessons Learned

1. Instrumentation reveals what batch evaluation hides. The Prometheus dashboard showed
   RAG and agent latencies differ by an order of magnitude (3s vs 22s). Separate
   histograms per pipeline type are essential for meaningful P99 alerting.

2. Chunk size mattered more than expected. M6 Precision@3=0.37 is partly explained by
   chunks being cut mid-sentence at 256 tokens. The A/B test confirmed chunk_size=512
   significantly improves groundedness (p=0.0006) on the 10-document MLOps corpus.

3. The 30% JSON failure rate is a real production risk. M6 evaluation showed Mistral 7B
   failed to produce clean JSON for tool selection 3 out of 10 times. The risk register
   forced a concrete mitigation: enable format=json in the Ollama API call.

4. Drift analysis made the corpus limitation quantitative. M6 noted qualitatively that
   OOD queries cause hallucination. Drift detection made this precise: PSI above 4.0
   by Week 4, retrieval similarity drops to 0.68. This is now a triggerable alert.

5. Components connect. The A/B test result (ship chunk_size=512) appears in the audit
   trail, updates the model card, and the monitoring dashboard detects the groundedness
   improvement as a rising retrieval similarity score. All 5 components tell one story.
