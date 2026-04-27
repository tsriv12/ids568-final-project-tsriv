# Drift Diagnostic Report — Mistral 7B RAG Pipeline

**Period:** Weeks 2-4 post-deployment
**Reference:** Week 1 (M6 evaluation baselines: similarity=0.891, P@3=0.37, latency=3,069ms)
**Method:** Population Stability Index (PSI) + Kolmogorov-Smirnov test

## 1. Features That Drifted Most

| Feature | W2 PSI | W3 PSI | W4 PSI | Status |
|---|---|---|---|---|
| query_length_tokens | 1.838 | 4.865 | 4.211 | CRITICAL all weeks |
| retrieval_similarity | 0.543 | 1.423 | 2.120 | CRITICAL all weeks |
| precision_at_3 | 0.108 | 0.317 | 0.986 | MONITOR W2, CRITICAL W3-4 |
| latency_ms | 0.142 | 0.462 | 0.578 | MONITOR W2, CRITICAL W3-4 |
| groundedness | 0.065 | 0.254 | 0.553 | OK W2, CRITICAL W3-4 |

PSI interpretation: below 0.10 = stable, 0.10-0.25 = monitor, above 0.25 = critical.
All features show critical drift by Week 4. All KS tests confirm drift (p=0.0000).

Most drifted feature: query_length_tokens (PSI=4.211 at Week 4).
Second most drifted: retrieval_similarity (PSI=2.120 at Week 4).

Root cause hypothesis: Users graduate from short, well-scoped MLOps questions
(M6 eval queries average ~17 words) to longer, multi-part queries spanning topics
outside the 10-document corpus. This is consistent with M6 known limitation:
queries about topics outside the corpus retrieve irrelevant documents.
The extreme PSI values (well above 0.25) indicate the production distribution
has shifted fundamentally, not just marginally.

## 2. Impact on Model Performance

Retrieval similarity dropped from 0.891 (baseline) to ~0.68 by Week 4, a 24%
relative decline. Based on M6 evaluation data:

- M6 Precision@3 was 0.37 at similarity=0.891
- At similarity=0.68, Precision@3 is estimated at ~0.22 (40% relative drop)
- Groundedness drops from 0.68 to ~0.55 by Week 4 (a 19% relative decline)

This means by Week 4, the majority of answers are more hallucinated than grounded.
Users asking longer, complex queries are receiving the least reliable answers,
which is the opposite of what a production system should deliver.

Latency also increased significantly (P50: 3,069ms to ~4,100ms by Week 4, a 34%
increase) due to longer queries creating larger prompts. This approaches but does
not yet exceed the 8s P99 SLA.

## 3. Retraining and Intervention Recommendations

| Priority | Action | Trigger Condition | Est. Effort |
|---|---|---|---|
| IMMEDIATE | Add retrieval confidence threshold: if top-1 similarity < 0.70 return I don't have enough information instead of hallucinating | retrieval_similarity week mean < 0.75 | 2 hours |
| IMMEDIATE | Implement query length cap: split queries over 100 tokens into sub-questions | query_length PSI > 0.25 | 4 hours |
| SHORT-TERM | Expand corpus from 10 to 50+ MLOps documents to cover more topic space | Sustained PSI > 0.15 on retrieval_similarity | 1-2 days |
| SHORT-TERM | Schedule weekly PSI monitoring job with automated alert at PSI > 0.15 | Ongoing | 2 hours |
| LONG-TERM | Fine-tune all-MiniLM-L6-v2 on in-domain MLOps query-document pairs | Precision@3 below 0.25 sustained | 3-5 days |
| LONG-TERM | Add query rewriting module to normalize input length before embedding | Sustained PSI > 0.20 on query_length | 2-3 days |

No full model retraining required. Mistral 7B weights are unchanged.
The issues are corpus scope and retrieval threshold logic, not model quality.
The recommended interventions are retrieval-layer changes achievable in days,
not weeks-long model training runs.
