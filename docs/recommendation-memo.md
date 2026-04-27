# Recommendation Memo — A/B Test: RAG Chunk Size

**To:** MLOps Course Instructor
**From:** Tanya Srivastava
**System:** Mistral 7B (Ollama) + ChromaDB mlops_rag + all-MiniLM-L6-v2
**Decision: Ship Variant B (chunk_size=512)**

## Results Summary

| Metric | A (chunk=256) | B (chunk=512) | Diff | p-value | Significant? |
|---|---|---|---|---|---|
| Groundedness | 0.673 | 0.725 | +0.052 | 0.0006 | Yes (alpha_adj=0.025) |
| E2E Latency (s) | 3.172 | 3.312 | +0.140s | 0.1922 | No |

## Reasoning

Variant B achieves a statistically significant groundedness improvement of +0.052
(p=0.0006, well below adjusted alpha=0.025), meeting our MDE of 0.05. This is
meaningful for our Mistral 7B system: larger 512-token chunks give the model more
surrounding context when answering MLOps questions from our 10-document corpus.

The +0.14s latency increase is not statistically significant (p=0.19) and both
variants stay well below the 8s SLA guardrail. On a T4 GPU, the extra context
tokens add negligible generation time.

The real-world implication: Mistral 7B benefits from 512-token chunks on technical
MLOps content, where concepts span multiple sentences. The 256-token split
frequently cuts explanations mid-sentence, degrading answer coherence.

## Next Steps
1. Re-index ChromaDB mlops_rag with chunk_size=512 in rag_pipeline.ipynb
2. Update model card to reflect chunk_size=512 as the validated configuration
3. Log configuration change in audit trail with timestamp
4. Monitor retrieval similarity in Grafana for first 48 hours post-change
