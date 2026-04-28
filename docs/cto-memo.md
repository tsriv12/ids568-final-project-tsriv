# Memorandum

**To:** Chief Technology Officer
**From:** Tanya Srivastava, MLOps Engineer
**Date:** April 27, 2026
**Re:** AI Risk Assessment — Mistral 7B RAG Question-Answering System
**Classification:** Internal Confidential

## What This System Does

This system answers technical questions about MLOps by searching an internal
knowledge base of 10 documents and generating responses using Mistral 7B, a
7-billion-parameter AI model running entirely on our own GCP hardware with no
external cloud AI services. A more capable agent mode breaks complex tasks into
steps using three specialized tools: a document retriever, a summarizer, and a
keyword extractor.

## Why We Built It

To demonstrate production-grade RAG and agentic AI capabilities for the MLOps
course capstone, and to establish an operational framework including monitoring,
governance, and drift detection suitable as a template for future internal AI tools.

## Key Findings — 3 Actions Required Before Any Production Use

### 1. The system fabricates answers when users ask outside its knowledge scope (Risk Score: 20/25)

When users ask questions not covered by the 10-document knowledge base, the AI
generates confident but fabricated answers rather than saying it does not know.
This was confirmed during evaluation: retrieval similarity drops to 0.535 on
out-of-scope queries, yet the system still generates an answer.

Action required: A 2-hour code fix — if the database search returns a similarity
score below 0.70, return a standard message: I don't have enough information to
answer this reliably. Please consult primary sources.

### 2. The AI fails to select the right tool 30% of the time in agent mode (Risk Score: 15/25)

In 3 out of 10 test tasks, Mistral 7B produced malformed output during tool
selection, causing the system to fall back to simpler rule-based behavior. This
degrades the quality of multi-step reasoning tasks significantly.

Action required: Enable structured JSON output enforcement in the Ollama API call
(30-minute code change: add format=json parameter). This forces Mistral to output
valid JSON every time.

### 3. A server crash causes complete system failure with no recovery (Risk Score: 15/25)

If the Ollama model server crashes, all requests fail immediately with no automatic
restart. There is currently no watchdog process.

Action required: Configure automatic restart via Linux systemd service manager
(1 hour of work). Add a Prometheus alert that pages within 1 minute of failure.
The monitoring dashboard for this alert is already built and deployed.

## What Is Already Working Well

Full observability is in place. The Prometheus and Grafana monitoring dashboard
tracks pipeline latency, retrieval quality, error rates, and model health in real
time with configured alert thresholds. This gives us immediate visibility into
system behavior that most early-stage AI deployments lack.

No external dependencies. All inference runs on our own T4 GPU via Ollama.
There are no third-party AI API costs, no data leaving our infrastructure, and
no vendor reliability risk.

Drift detection is operational. Automated weekly PSI analysis identifies when
user query patterns are drifting outside the system knowledge scope before the
degradation becomes visible to users.

## Recommendation

Safe for internal educational and research use only with existing controls.
Not ready for any user-facing production deployment until the 3 items above
are resolved. Estimated remediation time: 1 engineer, 1 working day.

For questions contact: tsriv@uic.edu
