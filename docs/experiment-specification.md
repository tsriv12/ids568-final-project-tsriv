# A/B Experiment Specification — RAG Chunk Size

**System:** Mistral 7B (Ollama) + ChromaDB mlops_rag + all-MiniLM-L6-v2
**Baseline from M6:** Precision@3 = 0.37, top-1 similarity ~0.891, E2E latency ~3,069ms

## Hypothesis

H0: No significant difference in answer groundedness between chunk_size=256 (A)
and chunk_size=512 (B) on the 10-document MLOps corpus.

H1: chunk_size=512 (B) improves answer groundedness by >= 0.05 absolute points
without increasing P99 latency beyond 8 seconds.

Groundedness is measured as keyword overlap between the generated answer and the
top retrieved chunk content (proxy metric suitable for offline simulation).

## Metrics

| Metric | Role | MDE |
|---|---|---|
| Groundedness (keyword overlap, 0-1) | Primary | +0.05 absolute |
| Retrieval top-1 similarity | Guardrail | Must not drop below 0.85 |
| E2E latency (seconds) | Guardrail | P99 must stay <= 8s |

## Randomization
- Unit: Query request (each of the 10 eval queries run N times)
- Split: 50/50 — even-indexed runs go to Variant A, odd-indexed to Variant B
- No carryover: each run uses a fresh ChromaDB query; stateless pipeline

## Sample Size and Power Calculation

alpha = 0.05 (two-tailed), power = 0.80
Baseline groundedness mu_A = 0.68 (estimated from M6 Precision@3=0.37)
MDE = delta = 0.05, sigma = 0.14 (estimated SD)

n = 2 x ((z_alpha/2 + z_beta) x sigma / delta)^2
  = 2 x ((1.96 + 0.84) x 0.14 / 0.05)^2
  = 2 x (7.84)^2
  = 2 x 61.5 = 123 per variant, 246 total

Actual sample used: 130 per variant (260 total) — exceeds minimum.

## Statistical Tests
- Welch's t-test (unequal variance) on groundedness and latency
- Bootstrap 95% CI on mean difference (B minus A), 5,000 iterations
- Bonferroni correction: alpha_adj = 0.05 / 2 = 0.025 (2 primary metrics)
- No early stopping — full sample run to avoid peeking bias
- Seeded RNG (seed=42) for full reproducibility
