"""
Drift Detection for Mistral 7B RAG Pipeline.
Reference distribution: Week 1 calibrated to real M6 eval baselines:
  - top-1 similarity: 0.891
  - Precision@3: 0.37
  - E2E latency: ~3,069ms
  - Query length: ~17 words avg (from 10 M6 eval queries)

Production weeks 2-4 simulate real-world query evolution
as users ask longer, more complex OOD questions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

M6_QUERY_LENGTHS = [21, 15, 18, 16, 13, 16, 22, 15, 17, 15]


def generate_reference(n=500):
    return pd.DataFrame({
        "query_length_tokens":  rng.normal(np.mean(M6_QUERY_LENGTHS), 4.0, n).clip(8, 60).astype(int),
        "retrieval_similarity": rng.normal(0.891, 0.05, n).clip(0.5, 1.0),
        "precision_at_3":       rng.normal(0.37, 0.12, n).clip(0.0, 1.0),
        "latency_ms":           rng.lognormal(np.log(3069), 0.25, n).clip(500, 15000),
        "groundedness":         rng.normal(0.68, 0.14, n).clip(0, 1),
        "week": 1
    })


def generate_production(n_per_week=500):
    frames = []
    frames.append(pd.DataFrame({
        "query_length_tokens":  rng.normal(28, 7, n_per_week).clip(8, 100).astype(int),
        "retrieval_similarity": rng.normal(0.845, 0.07, n_per_week).clip(0.4, 1.0),
        "precision_at_3":       rng.normal(0.34, 0.13, n_per_week).clip(0, 1),
        "latency_ms":           rng.lognormal(np.log(3200), 0.28, n_per_week).clip(500, 20000),
        "groundedness":         rng.normal(0.65, 0.15, n_per_week).clip(0, 1),
        "week": 2
    }))
    frames.append(pd.DataFrame({
        "query_length_tokens":  rng.normal(42, 12, n_per_week).clip(8, 150).astype(int),
        "retrieval_similarity": rng.normal(0.775, 0.09, n_per_week).clip(0.3, 1.0),
        "precision_at_3":       rng.normal(0.29, 0.14, n_per_week).clip(0, 1),
        "latency_ms":           rng.lognormal(np.log(3600), 0.32, n_per_week).clip(500, 20000),
        "groundedness":         rng.normal(0.61, 0.16, n_per_week).clip(0, 1),
        "week": 3
    }))
    frames.append(pd.DataFrame({
        "query_length_tokens":  rng.normal(60, 18, n_per_week).clip(8, 250).astype(int),
        "retrieval_similarity": rng.normal(0.68, 0.12, n_per_week).clip(0.2, 1.0),
        "precision_at_3":       rng.normal(0.22, 0.14, n_per_week).clip(0, 1),
        "latency_ms":           rng.lognormal(np.log(4100), 0.38, n_per_week).clip(500, 25000),
        "groundedness":         rng.normal(0.55, 0.17, n_per_week).clip(0, 1),
        "week": 4
    }))
    return pd.concat(frames, ignore_index=True)


def psi(ref, prod, n_bins=10):
    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    r = np.histogram(ref, bins=bins)[0] / len(ref)
    p = np.histogram(prod, bins=bins)[0] / len(prod)
    r = np.where(r == 0, 1e-6, r)
    p = np.where(p == 0, 1e-6, p)
    return float(np.sum((p - r) * np.log(p / r)))


def run_drift_analysis(out_dir="visualizations"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("Generating reference (Week 1) and production (Weeks 2-4) data...")
    ref  = generate_reference(500)
    prod = generate_production(500)

    features = ["query_length_tokens", "retrieval_similarity",
                "precision_at_3", "latency_ms", "groundedness"]
    weeks = [2, 3, 4]

    print(f"\n{'Feature':<28} {'W2 PSI':>8} {'W3 PSI':>8} {'W4 PSI':>8}  Status")
    print("-" * 70)
    psi_table = {}
    for feat in features:
        row = {}
        flags = []
        for w in weeks:
            v = psi(ref[feat].values, prod[prod["week"] == w][feat].values)
            row[w] = round(v, 3)
            flags.append("CRITICAL" if v > 0.25 else ("MONITOR" if v > 0.10 else "OK"))
        psi_table[feat] = row
        print(f"{feat:<28} {row[2]:>8.3f} {row[3]:>8.3f} {row[4]:>8.3f}  {flags}")

    print("\nKS tests (Week 4 vs Week 1):")
    for feat in features:
        ks, p = stats.ks_2samp(ref[feat].values,
                               prod[prod["week"] == 4][feat].values)
        print(f"  {feat:<28} KS={ks:.3f}  p={p:.4f}  {'DRIFT' if p < 0.05 else 'OK'}")

    # Time-series plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Feature Drift Over Time — Mistral 7B RAG Pipeline\n"
        "Reference: Week 1 M6 baselines (similarity=0.891, P@3=0.37, latency~3069ms)",
        fontsize=11, fontweight="bold"
    )
    for ax, feat in zip(axes.flatten(), features):
        ref_mu = ref[feat].mean()
        ax.axhline(ref_mu, color="blue", linestyle="--", alpha=0.8,
                   label=f"Week 1 baseline={ref_mu:.3f}")
        means = [prod[prod["week"] == w][feat].mean() for w in weeks]
        stds  = [prod[prod["week"] == w][feat].std()  for w in weeks]
        ax.plot(weeks, means, "ro-", label="Production")
        ax.fill_between(weeks,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color="red")
        ax.set_title(feat.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Week")
        ax.legend(fontsize=7)
    axes.flatten()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/drift_time_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir}/drift_time_series.png")

    # PSI bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(weeks))
    width = 0.15
    for i, feat in enumerate(features):
        bars = [psi_table[feat][w] for w in weeks]
        ax.bar(x + i * width, bars, width, label=feat.replace("_", " "))
    ax.axhline(0.10, color="orange", linestyle="--", label="PSI=0.10 Monitor", lw=1.5)
    ax.axhline(0.25, color="red",    linestyle="--", label="PSI=0.25 Retrain", lw=1.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Week {w}" for w in weeks])
    ax.set_ylabel("PSI Score")
    ax.set_title(
        "Population Stability Index by Feature and Week\n"
        "Mistral 7B RAG Pipeline — based on M6 baselines"
    )
    ax.legend(fontsize=7.5, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/drift_psi_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir}/drift_psi_summary.png")

    return psi_table


if __name__ == "__main__":
    run_drift_analysis()
