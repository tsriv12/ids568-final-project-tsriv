"""
A/B Test Simulation: RAG Chunk Size 256 vs 512
System: Mistral 7B (Ollama) + ChromaDB mlops_rag + all-MiniLM-L6-v2

Calibrated to real M6 metrics:
  - M6 Precision@3 = 0.37
  - M6 top-1 similarity ~0.891
  - M6 E2E latency ~3,069ms
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
N_PER_VARIANT = 130  # exceeds minimum of 123 from power calculation

EVAL_QUERIES = [
    "What is retrieval-augmented generation and what are the key metrics used to evaluate it?",
    "Compare FAISS and ChromaDB as vector databases.",
    "How does data drift affect production ML systems?",
    "What chunking strategies exist for RAG systems?",
    "How does vLLM improve LLM serving performance?",
    "What is the role of a feature store in MLOps?",
    "Explain the ReAct framework for agents.",
    "What embedding models are used for semantic search?",
    "Walk through the complete RAG pipeline steps.",
    "What MLOps tools are used for experiment tracking?",
]

# Variant A: chunk_size=256
# Smaller chunks -> precise but incomplete context -> lower groundedness
A_GROUND_MU, A_GROUND_SD = 0.68, 0.14
A_SIM_MU,    A_SIM_SD    = 0.88, 0.06
A_LAT_MU,    A_LAT_SD    = 3.0,  0.8

# Variant B: chunk_size=512
# Larger chunks -> more context for Mistral -> better groundedness
B_GROUND_MU, B_GROUND_SD = 0.735, 0.13
B_SIM_MU,    B_SIM_SD    = 0.87, 0.07
B_LAT_MU,    B_LAT_SD    = 3.3,  0.9


def simulate_variant(n, ground_mu, ground_sd, sim_mu, sim_sd,
                     lat_mu, lat_sd, error_rate=0.02):
    return pd.DataFrame({
        "groundedness":     np.clip(rng.normal(ground_mu, ground_sd, n), 0, 1),
        "top1_similarity":  np.clip(rng.normal(sim_mu, sim_sd, n), 0, 1),
        "latency_s":        np.abs(rng.lognormal(np.log(lat_mu), lat_sd / lat_mu, n)),
        "error":            rng.binomial(1, error_rate, n),
        "query":            [EVAL_QUERIES[i % len(EVAL_QUERIES)] for i in range(n)],
    })


def run_tests(df_a, df_b):
    BONFERRONI_K = 2
    alpha_adj = 0.05 / BONFERRONI_K
    results = {}

    for metric in ["groundedness", "latency_s"]:
        a, b = df_a[metric].values, df_b[metric].values
        t, p = stats.ttest_ind(a, b, equal_var=False)

        diffs = [rng.choice(b, len(b), replace=True).mean()
                 - rng.choice(a, len(a), replace=True).mean()
                 for _ in range(5000)]
        ci = np.percentile(diffs, [2.5, 97.5])

        results[metric] = {
            "mean_a": round(float(a.mean()), 4),
            "mean_b": round(float(b.mean()), 4),
            "diff_b_minus_a": round(float(b.mean() - a.mean()), 4),
            "p_value": round(float(p), 4),
            "significant_at_bonferroni": bool(p < alpha_adj),
            "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        }
    return results


def plot_results(df_a, df_b, results, out_dir="visualizations"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    colors = {"A (chunk=256)": "#4C72B0", "B (chunk=512)": "#DD8452"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "A/B Test: RAG Chunk Size 256 vs 512\n"
        "Mistral 7B (Ollama) · ChromaDB mlops_rag · all-MiniLM-L6-v2",
        fontsize=11, fontweight="bold"
    )

    # 1. Groundedness distribution
    combined = pd.concat([
        df_a.assign(Variant="A (chunk=256)"),
        df_b.assign(Variant="B (chunk=512)")
    ])
    sns.histplot(data=combined, x="groundedness", hue="Variant",
                 kde=True, ax=axes[0], palette=colors, alpha=0.6)
    axes[0].set_title(
        f"Groundedness Score\n"
        f"B−A = {results['groundedness']['diff_b_minus_a']:+.3f}, "
        f"p = {results['groundedness']['p_value']:.4f}"
    )
    axes[0].axvline(df_a["groundedness"].mean(), color="#4C72B0", linestyle="--", alpha=0.8)
    axes[0].axvline(df_b["groundedness"].mean(), color="#DD8452", linestyle="--", alpha=0.8)

    # 2. Latency boxplot
    sns.boxplot(data=combined, x="Variant", y="latency_s",
                ax=axes[1], palette=colors)
    axes[1].axhline(8.0, color="red", linestyle="--", alpha=0.6, label="8s SLA")
    axes[1].set_title(
        f"E2E Latency (s)\n"
        f"B−A = {results['latency_s']['diff_b_minus_a']:+.3f}s, "
        f"p = {results['latency_s']['p_value']:.4f}"
    )
    axes[1].legend(fontsize=8)

    # 3. CI forest plot
    metrics = ["groundedness", "latency_s"]
    labels  = ["Groundedness", "Latency (s)"]
    diffs   = [results[m]["diff_b_minus_a"] for m in metrics]
    ci_lo   = [results[m]["diff_b_minus_a"] - results[m]["ci_95"][0] for m in metrics]
    ci_hi   = [results[m]["ci_95"][1] - results[m]["diff_b_minus_a"] for m in metrics]
    axes[2].errorbar(diffs, labels, xerr=[ci_lo, ci_hi],
                     fmt="o", capsize=8, color="#2ecc71", ecolor="gray", markersize=9)
    axes[2].axvline(0, color="red", linestyle="--", alpha=0.7, label="No difference")
    axes[2].set_title("Mean Difference (B − A)\n95% Bootstrap CI")
    axes[2].set_xlabel("Difference")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = f"{out_dir}/ab_test_chunk_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main(dry_run=False):
    print("=" * 60)
    print("A/B Test: chunk_size=256 (A) vs chunk_size=512 (B)")
    print(f"System: Mistral 7B · ChromaDB mlops_rag · {N_PER_VARIANT} samples/variant")
    print("=" * 60)

    df_a = simulate_variant(N_PER_VARIANT, A_GROUND_MU, A_GROUND_SD,
                             A_SIM_MU, A_SIM_SD, A_LAT_MU, A_LAT_SD)
    df_b = simulate_variant(N_PER_VARIANT, B_GROUND_MU, B_GROUND_SD,
                             B_SIM_MU, B_SIM_SD, B_LAT_MU, B_LAT_SD)

    if dry_run:
        print("Dry run: simulation data generated successfully.")
        return

    results = run_tests(df_a, df_b)

    print("\n── RESULTS ──")
    for metric, r in results.items():
        sig = "✅ Significant" if r["significant_at_bonferroni"] else "❌ Not significant"
        print(f"{metric}: A={r['mean_a']:.3f} → B={r['mean_b']:.3f}  "
              f"(diff={r['diff_b_minus_a']:+.3f}, p={r['p_value']:.4f}) {sig}")

    with open("ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: ab_test_results.json")

    plot_results(df_a, df_b, results)
    return results


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
