# Dashboard Interpretation — Mistral RAG Agent Production Monitor

## System Context
This dashboard monitors a two-mode pipeline built on Milestone 6:
- **RAG mode:** Query → all-MiniLM-L6-v2 → ChromaDB (mlops_rag) → Mistral 7B → Answer (~3s)
- **Agent mode:** Task → Mistral tool selection → [Retriever | Summarizer | KeywordExtractor] → Answer (~22s)

Hardware: GCP VM · NVIDIA Tesla T4 · Ollama (mistral:7b-instruct, Q4_K_M quantized)

Baseline metrics from Milestone 6 evaluation:
- RAG E2E latency: ~3,069ms (P50)
- Retrieval latency: ~31ms
- Generation latency: ~3,039ms
- Top-1 retrieval similarity: ~0.891
- Precision@3: 0.37

---

## 1. What the Dashboard Reveals About System Health

### RAG Request Rate (QPS)
Tracks queries per second flowing through the RAG pipeline. During simulation,
the system processed 71 requests at ~0.12 QPS — well within the single T4 GPU's
capacity. A sustained rate above 0.5 QPS would cause queuing since Ollama handles
one inference at a time. A drop to 0 QPS for >2 minutes indicates a service crash.

### Total RAG Requests
Cumulative success counter. In production, a flat counter combined with active
user sessions indicates a silent failure — requests arriving but not completing.

### RAG P99 Latency
P99 fluctuated between 5s and 9s during simulation, breaching the 8s SLA line
twice. This is expected behavior: the 8s breaches correspond to longer queries
(word-heavy anomalous inputs) causing larger prompts for Mistral to process.
In production, sustained P99 > 8s for 3+ minutes would trigger a Warning alert.

### Generation Latency P50
Median generation time of ~3.0–4.0s matches the M6 baseline of 3,039ms exactly,
confirming the instrumentation is correctly wrapping the Ollama chat call. The
variance (2.0s–4.0s) reflects Mistral's natural token generation variability
depending on answer length.

### Retrieval Similarity Score
Current value: 0.535 — below the 0.70 healthy threshold (shown in red).
This is lower than the M6 baseline of 0.891 because the simulation includes
anomalous queries ("hi", 400-word repetitive strings) that retrieve irrelevant
documents. In production, a sustained score below 0.70 means:
- Users are querying outside the 10-document MLOps corpus scope, OR
- The knowledge base needs re-indexing with updated documents.
This is the single most important leading indicator for hallucination risk.

### Ollama Model Status
Shows "LOADED" (value=1) confirming mistral:7b-instruct is available via Ollama.
If this drops to 0, all pipeline requests fail immediately. This panel should
trigger a Critical PagerDuty alert within 1 minute of going to 0.

### Input Anomaly Flag
Spikes to 1 when a query is <3 tokens or >300 tokens. The regular spikes in
the simulation correspond to the intentional anomalous queries in the query pool
("hi" = too short, 400-word string = too long). In production, a sustained
anomaly rate >10% could indicate bot traffic or prompt injection attempts.

---

## 2. Identified Bottlenecks and Risks

| Bottleneck | Signal | Severity |
|---|---|---|
| Single T4 GPU | RAG and Agent requests queue; P99 spikes to 9s | Medium |
| Mistral generation variance | P50 ranges 2s–4s; hard to SLA guarantee | Medium |
| Retrieval quality on OOD queries | Similarity drops to 0.535 on anomalous inputs | High |
| Ollama single process | No redundancy; crash = full outage | Critical |

**Primary bottleneck:** Ollama runs a single model instance on one GPU. Concurrent
requests queue silently. At >0.3 QPS with mixed RAG + Agent traffic, P99 will
consistently breach 8s due to the 22s agent tasks blocking RAG queries.

---

## 3. Alert Trigger Conditions for Production

| Alert Name | Condition | Severity | Action |
|---|---|---|---|
| Ollama Down | `agent_ollama_model_loaded == 0` for 1 min | Critical | Page on-call; auto-restart Ollama |
| RAG SLA Breach | RAG P99 > 8s for 3 min | Warning | Check for agent queue blocking |
| Retrieval Degradation | `agent_retrieval_top_similarity < 0.70` for 10 min | Warning | Check corpus scope; re-index |
| Zero Throughput | `rate(agent_requests_total[2m]) == 0` for 2 min | Critical | Check service health |
| Anomaly Spike | Anomaly flag = 1 for >10% of requests in 5 min | Warning | Check for bot/injection traffic |

---

## 4. Design Justification

**Why Prometheus + Grafana?**
The entire M6 stack (Ollama, ChromaDB) runs locally on the GCP VM with no external
API dependencies. Prometheus's pull-based scrape model works perfectly with a local
Python metrics server (port 8001). No cloud monitoring credentials required —
consistent with the project's no-proprietary-API constraint.

**Why separate histograms for retrieval vs generation?**
RAG latency has two distinct components: retrieval (~31ms) and generation (~3,039ms).
Mixing them in a single histogram would make percentiles meaningless. Separate
histograms allow pinpointing whether a latency spike is a ChromaDB issue or a
Mistral issue — critical for fast root cause analysis.

**Why retrieval similarity as a gauge vs histogram?**
The similarity score is a point-in-time quality signal, not a latency distribution.
A gauge is the correct Prometheus type — it shows the current health state and
enables threshold-based alerting, which a histogram cannot do directly.
