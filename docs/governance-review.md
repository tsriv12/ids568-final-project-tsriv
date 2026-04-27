# Governance Review — Mistral 7B RAG + Agent Pipeline

## 1. Data Security

All inference runs locally on the GCP VM. No data leaves the machine.
No external API keys are required (Ollama serves Mistral locally).

Current gaps:
- ChromaDB vector store has no encryption at rest
- Prometheus metrics endpoint (port 8001) is unauthenticated
- Ollama API (port 11434) has no authentication
- In production: restrict ports via GCP firewall rules and add API key auth

## 2. Retrieval Risks

Exposure risk: The current 10-document corpus contains no PII. However, no PII
scanner exists on the ingestion pipeline. If the corpus is ever expanded with
enterprise documents, sensitive information could be surfaced in retrieval results
without warning.

Contamination risk: No adversarial documents were tested in M6. Mistral 7B is
susceptible to indirect prompt injection — if a retrieved document contains
embedded instructions (e.g., "Ignore previous instructions and..."), Mistral
may follow them. No guardrails exist against this attack vector.

Stale knowledge risk: All 10 documents were indexed once at project start.
The system provides no freshness indicator to users. Queries about rapidly
evolving MLOps tools (e.g., new LLM releases, updated APIs) will receive
outdated answers. M6 correctly abstained on some OOD queries but this relies
on Mistral judgment, not an explicit confidence threshold in code.

## 3. Hallucination Risk Points

| Stage | Risk Level | Mechanism |
|---|---|---|
| OOD queries (similarity < 0.70) | Critical | Mistral generates from parametric memory when retrieved docs are irrelevant |
| Multi-hop agent reasoning | High | Each step compounds errors from previous steps |
| Summarizer tool output | Medium | Mistral summarization can lose specific numbers and model names |
| Long query context overflow | Medium | Silent truncation loses key facts from long retrieved chunks |

Monitoring mitigation in place: Prometheus tracks retrieval_similarity in real
time. Alert fires when similarity drops below 0.70, providing early warning
before hallucination rate becomes visible to users.

## 4. Tool-Misuse Pathways

The agent controller (agent_controller.py) has three tools:
- tool_retriever: ChromaDB semantic search (read-only)
- tool_summarizer: Mistral-based summarization of retrieved content
- tool_keyword_extractor: Keyword extraction from retrieved content

Risk pathways observed in M6:
- JSON parsing failure (30% of tasks): Mistral produces preamble text before
  JSON, triggering rule-based fallback. Fallback uses keyword matching which
  is less accurate than LLM tool selection.
- Unbounded summarization: Summarizer faithfully summarizes wrong content if
  retriever returns irrelevant documents. No relevance check before summarizing.
- Max steps enforcement: run_agent() enforces max 4 steps. However the DONE
  signal relies on Mistral deciding to stop. If JSON fails every step, agent
  runs all 4 steps regardless of whether sufficient context was gathered.

No write tools exist in the current implementation (confirmed in M6 code).
This eliminates data exfiltration via tool output as a risk vector.

## 5. Compliance Concerns

GDPR / CCPA:
- Query content is logged to agent_traces/ directory in plain JSON
- In a real deployment, user queries may constitute personal data under GDPR
- No deletion mechanism exists for query logs
- Action required: implement 30-day rolling deletion before any user-facing deployment

Model transparency:
- No AI output disclaimer is appended to responses
- Users cannot distinguish between grounded and hallucinated answers
- EU AI Act transparency requirements would mandate disclosure of AI-generated content

Data minimization:
- Ollama model weights (4.2GB) stored on VM disk without encryption
- No access controls on the ChromaDB directory
- Any user with VM access can read all indexed documents and query logs
