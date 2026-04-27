"""
Generates two required diagrams:
1. docs/lineage-diagram.png
2. docs/system-boundary-diagram.png
Run from repo root: python src/monitoring/generate_diagrams.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("docs").mkdir(exist_ok=True)


def lineage_diagram():
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    stages = [
        ("10 MLOps\nDocuments", 1.0, "#AED6F1"),
        ("Chunker\n(512 tok)", 3.2, "#A9DFBF"),
        ("all-MiniLM\nL6-v2", 5.4, "#A9DFBF"),
        ("ChromaDB\nmlops_rag", 7.6, "#F9E79F"),
        ("Retriever\ntop-k=3", 9.8, "#F9E79F"),
        ("Mistral 7B\n(Ollama)", 12.0, "#F1948A"),
        ("Prometheus\nMonitor", 14.2, "#D7BDE2"),
    ]

    for label, x, color in stages:
        patch = mpatches.FancyBboxPatch(
            (x - 0.9, 1.3), 1.8, 1.4,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#555", lw=1.2
        )
        ax.add_patch(patch)
        ax.text(x, 2.0, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    for i in range(len(stages) - 1):
        ax.annotate("",
            xy=(stages[i+1][1] - 0.9, 2.0),
            xytext=(stages[i][1] + 0.9, 2.0),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.5)
        )

    ax.text(8.0, 3.6,
            "Model Lineage: Data -> Chunking -> Embedding -> Vector Store -> Retrieval -> Generation -> Monitoring",
            ha="center", fontsize=9, fontweight="bold", color="#333")

    plt.savefig("docs/lineage-diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: docs/lineage-diagram.png")


def system_boundary_diagram():
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    def box(x, y, w, h, label, sub="", color="#DAE8FC"):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="#2C3E50", lw=1.4
        )
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2 + (0.12 if sub else 0), label,
                ha="center", va="center", fontsize=8.5, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.22, sub,
                    ha="center", va="center", fontsize=7, color="#555")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.4))

    boundary = mpatches.FancyBboxPatch(
        (0.2, 0.3), 15.6, 6.8,
        boxstyle="round,pad=0.1",
        facecolor="none", edgecolor="#E74C3C", lw=2, linestyle="--"
    )
    ax.add_patch(boundary)
    ax.text(8.0, 7.25,
            "System Boundary — Mistral 7B RAG + Agentic Controller (ids568-milestone6)",
            ha="center", fontsize=10.5, fontweight="bold", color="#E74C3C")

    box(0.4,  2.8, 1.8, 1.2, "User\nQuery", color="#D5E8D4")
    box(2.6,  2.8, 2.0, 1.2, "all-MiniLM\nEmbedder", "sentence-transformers", color="#DAE8FC")
    box(5.0,  2.8, 2.0, 1.2, "ChromaDB\nRetriever", "mlops_rag top-k=3", color="#F9E79F")
    box(7.4,  2.8, 2.0, 1.2, "Mistral 7B\n(Ollama)", "mistral:7b-instruct", color="#FFE6CC")
    box(9.8,  2.8, 2.0, 1.2, "Agent\nController", "Tool selection loop", color="#FFE6CC")
    box(12.2, 2.8, 1.8, 1.2, "Output\n+ Citation", color="#DAE8FC")
    box(14.3, 2.8, 1.4, 1.2, "User\nResponse", color="#D5E8D4")

    box(5.0,  0.8, 2.0, 1.0, "ChromaDB\nVector Store", "10 docs 120 chunks", color="#F9E79F")
    box(7.4,  0.8, 2.0, 1.0, "Tools", "Ret | Sum | KW", color="#FDEBD0")
    box(12.2, 0.8, 1.8, 1.0, "Prometheus\nMetrics", color="#D7BDE2")

    arrow(2.2,  3.4, 2.6,  3.4)
    arrow(4.6,  3.4, 5.0,  3.4)
    arrow(7.0,  3.4, 7.4,  3.4)
    arrow(9.4,  3.4, 9.8,  3.4)
    arrow(11.8, 3.4, 12.2, 3.4)
    arrow(14.0, 3.4, 14.3, 3.4)
    arrow(6.0,  2.8, 6.0,  1.8)
    arrow(8.4,  2.8, 8.4,  1.8)
    arrow(13.1, 2.8, 13.1, 1.8)

    risks = [
        (3.6,  4.35, "Risk: OOD query\nlow similarity"),
        (6.0,  4.35, "Risk: Stale corpus\noutdated answer"),
        (8.4,  4.35, "Risk: Prompt injection\nanswer hijack"),
        (10.8, 4.35, "Risk: JSON fail 30%\nrule fallback"),
        (13.1, 4.35, "Risk: Log retention\nGDPR gap"),
    ]
    for x, y, label in risks:
        ax.text(x, y, label, ha="center", fontsize=7, color="#C0392B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FADBD8",
                          edgecolor="#E74C3C", alpha=0.85))

    plt.savefig("docs/system-boundary-diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: docs/system-boundary-diagram.png")


if __name__ == "__main__":
    lineage_diagram()
    system_boundary_diagram()
