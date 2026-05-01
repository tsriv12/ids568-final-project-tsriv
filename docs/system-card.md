# System Card — Mistral 7B RAG + Agentic Controller Pipeline

**Version:** 1.1.0 (chunk_size=512, post A/B test)
**Card Type:** System Card (RAG + Agentic Pipeline)
**Date:** April 27, 2026
**Owner:** Tanya Srivastava | tsriv | ids568
**Repository:** ids568-milestone6-tsriv -> ids568-final-project-tsriv

## Model Overview

**Architecture:** Retrieval-Augmented Generation with Multi-Tool Agent Controller
**LLM:** mistral:7b-instruct (7B parameters, Q4_K_M quantized via Ollama)
**Embedding Model:** all-MiniLM-L6-v2 (sentence-transformers, 22M params)
**Vector Store:** ChromaDB (persistent, collection: mlops_rag)
**Chunk Size:** 512 tokens (validated via A/B test, upgraded from 256 in v1.0.0)
**Hardware:** GCP VM · NVIDIA Tesla T4 · 15GB VRAM · CUDA 12.2

Two operational modes:
1. RAG Mode: Query -> Embedding -> ChromaDB -> Mistral generation -> Answer (~3s)
2. Agent Mode: Task -> Mistral tool selection -> [Retriever | Summarizer | KeywordExtractor] -> up to 4 steps -> Answer (~22s)

## Performance Metrics

All metrics from Milestone 6 evaluation (10 evaluation queries, 10 agent tasks).

### RAG Pipeline
| Metric | Value |
|---|---|
| Avg retrieval latency | ~31ms |
| Avg generation latency | ~3,039ms |
| Avg E2E latency | ~3,069ms |
| Precision@3 | 0.37 |
| Top-1 retrieval similarity | ~0.891 |

### Agent Controller
| Metric | Value |
|---|---|
| Avg E2E latency | ~22,434ms |
| Tool selection match | ~70-80% |
| Avg steps per task | 2.3 |
| JSON parse success rate | 7/10 tasks (70%) |

## Training Data Description

This system does not train any new model weights. It uses:
- LLM: mistral:7b-instruct (pre-trained, pulled via Ollama, weights unchanged)
- Embedding model: all-MiniLM-L6-v2 (pre-trained, not fine-tuned)
- Knowledge base: 10 synthetic MLOps documents generated for Milestone 6
  Topics: RAG, vector databases, drift detection, LLM serving, feature stores,
  embeddings, experiment tracking, chunking strategies, vLLM, agent frameworks
  Total chunks: ~120 (at chunk_size=512)
  No real user data, no PII

## Limitations and Failure Modes

1. 10-document corpus boundary: Queries about topics outside these 10 MLOps documents
   produce low-similarity retrievals (< 0.70) and Mistral hallucinates rather than
   abstaining. Observed in M6: model correctly abstains on fine-tuning and RL queries.

2. Mistral JSON parsing reliability (30% failure rate): On 3/10 agent tasks, Mistral 7B
   produced tool selection output with preamble text before the JSON block, triggering
   the rule-based fallback. A larger model (14B) or structured output enforcement fixes this.

3. Single T4 GPU contention: Concurrent RAG + Agent requests queue behind each other
   since Ollama uses a single model instance. Not suitable for multi-user production
   without vLLM or dedicated GPU per path.

4. English-only: The embedding model and Mistral perform poorly on non-English queries.
   No language detection is implemented.

5. Static knowledge base: Documents are indexed once. Queries about events after
   the indexing date will receive outdated answers without any staleness warning.

6. Context window cap: Mistral 7B instruction context is 8,192 tokens. With 5 chunks
   at 512 tokens each (~2,560 tokens) plus prompt overhead, we have ~5,500 tokens of
   generation budget — adequate but leaves little room for very long answers.

## Ethical Risks and Considerations

- Hallucination confidence: Mistral 7B presents hallucinated content with the same
  confident tone as grounded content. Users cannot distinguish without source checking.
- Corpus bias: The 10 MLOps documents may over-represent certain tools (ChromaDB,
  MLflow) at the expense of alternatives, biasing agent recommendations.
- Prompt injection: Retrieved document content could contain adversarial instructions
  that manipulate Mistral generation (indirect prompt injection).

## Intended Use

In-scope:
- MLOps concept Q&A for students and practitioners
- Research summarization over the 10-document corpus
- Demonstrating RAG + agentic patterns for educational purposes

Out-of-scope:
- Production user-facing deployments without additional safety measures
- Queries requiring real-time information (no live web access)
- Medical, legal, or financial decisions
- High-throughput serving (more than 2 concurrent users without GPU upgrade)

## Retrieval System Details

The retrieval component is a first-class system citizen, not just a model input stage.

| Property | Value |
|---|---|
| Vector store | ChromaDB (persistent, local disk) |
| Collection | mlops_rag |
| Embedding model | all-MiniLM-L6-v2 (22M params, not fine-tuned) |
| Chunk size | 512 tokens (validated via A/B test) |
| Chunk overlap | 0 tokens |
| Top-k retrieval | k=3 documents per query |
| Similarity metric | Cosine similarity |
| Corpus size | 10 synthetic MLOps documents, ~120 chunks |
| Freshness | Static — indexed once at project start, no automated updates |

Retrieval quality directly determines generation quality. At top-1 similarity below 0.70,
Mistral 7B generates from parametric memory rather than retrieved context, significantly
increasing hallucination risk. This threshold is monitored in real time via Prometheus.

## Lineage Summary
See docs/lineage-diagram.png

10 MLOps Docs -> Chunker (512 tok) -> all-MiniLM-L6-v2 -> ChromaDB (mlops_rag)
-> Retriever (top-k=3) -> Mistral 7B (Ollama) -> [Summarizer | KeywordExtractor]
-> Final Answer -> Prometheus Monitoring
