# Risk Matrix — Mistral 7B RAG + Agent Pipeline

## Scoring Scale
Likelihood: 1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain
Severity:   1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Critical
Risk Score = Likelihood x Severity

| Risk ID | Risk | Likelihood | Severity | Score | Priority | Mitigation |
|---|---|---|---|---|---|---|
| R03 | Corpus scope triggers hallucination on OOD queries | 5 | 4 | 20 | CRITICAL | Add retrieval confidence threshold: return I dont know if top-1 similarity < 0.70 |
| R01 | Mistral JSON parsing fails 30% of agent tasks | 5 | 3 | 15 | CRITICAL | Enable Ollama format=json parameter in all tool-selection calls |
| R04 | Ollama crash causes full pipeline outage | 3 | 5 | 15 | CRITICAL | Systemd watchdog + Prometheus alert at agent_ollama_model_loaded == 0 |
| R06 | Prompt injection via retrieved document content | 3 | 4 | 12 | HIGH | Add RETRIEVED_CONTENT delimiters in system prompt; sanitize chunks |
| R09 | No AI disclaimer leads to over-reliance on outputs | 4 | 3 | 12 | HIGH | Append disclaimer to all responses before returning to user |
| R05 | Query logs retained indefinitely violating GDPR | 3 | 4 | 12 | HIGH | 30-day rolling deletion of agent_traces/ and query logs |
| R02 | T4 GPU contention causes silent request queuing | 5 | 2 | 10 | MEDIUM | Request queue with 10s timeout; Prometheus P99 alert |
| R08 | No PII scanner on document ingestion | 2 | 5 | 10 | MEDIUM | Add Microsoft Presidio scan before any new document indexing |
| R07 | 10-doc corpus bias toward ChromaDB and MLflow | 4 | 2 | 8 | MEDIUM | Expand corpus to 50+ docs with diverse tooling coverage |
| R10 | Context window overflow with 5 chunks at 512 tokens | 2 | 2 | 4 | LOW | Enforce max 4 chunks; add token budget check before generation |

## Top Risks by Score
R03=20, R01=15, R04=15, R06=12, R09=12, R05=12

## Pre-launch Blockers (score >= 15)
R03 corpus hallucination, R01 JSON parsing failure, R04 Ollama crash
