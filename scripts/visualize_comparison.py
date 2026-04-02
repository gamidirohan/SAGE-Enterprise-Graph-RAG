"""
Visualize comparison results: generate one image per metric type,
with all model combinations (LLM × Embedding) side-by-side.

LLM models:       gemma2-9b-it, llama3-8b-8192
Embedding models:  all-mpnet-base-v2, multi-qa-mpnet-base-dot-v1

Usage:
    uv run python scripts/visualize_comparison.py [--results RESULTS_FILE]
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

# ── Consistent styling ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {
    "SAGE Graph RAG": "#2ecc71",
    "Traditional RAG": "#e74c3c",
}
COMBO_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load and return raw results list from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


def filter_valid(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop entries where either system returned an error answer."""
    valid = []
    for r in results:
        sage_ans = r.get("sage_response", {}).get("answer", "")
        trad_ans = r.get("traditional_response", {}).get("answer", "")
        if "Error" in sage_ans or "Error" in trad_ans:
            continue
        valid.append(r)
    logger.info(f"Kept {len(valid)}/{len(results)} valid entries (filtered out errors)")
    return valid


def combo_label(llm_model: str, embedding_model: str) -> str:
    """Create a short human-readable label for an LLM + Embedding combination."""
    # Shorten for readability on axes
    llm_short = llm_model.replace("-8192", "").replace("-9b-it", "")
    emb_short = embedding_model.replace("all-mpnet-base-v2", "mpnet-v2").replace(
        "multi-qa-mpnet-base-dot-v1", "multi-qa-mpnet"
    )
    return f"{llm_short}\n({emb_short})"


def build_combo_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with one row per model combination, aggregating metrics."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        llm = r["llm_model"]
        emb = r["embedding_model"]
        key = f"{llm}||{emb}"
        groups.setdefault(key, []).append(r)

    rows = []
    for key, entries in groups.items():
        llm, emb = key.split("||")
        sage_scores = [e["llm_evaluation"].get("system1_score", 0) for e in entries]
        trad_scores = [e["llm_evaluation"].get("system2_score", 0) for e in entries]
        sage_latencies = [e["sage_response"]["latency"] for e in entries]
        trad_latencies = [e["traditional_response"]["latency"] for e in entries]
        sage_wins = sum(1 for e in entries if e["llm_evaluation"].get("better_system") == "system1")
        trad_wins = sum(1 for e in entries if e["llm_evaluation"].get("better_system") == "system2")
        ties = sum(1 for e in entries if e["llm_evaluation"].get("better_system") == "tie")

        rows.append({
            "llm_model": llm,
            "embedding_model": emb,
            "label": combo_label(llm, emb),
            "sage_avg_score": np.mean(sage_scores),
            "trad_avg_score": np.mean(trad_scores),
            "sage_avg_latency": np.mean(sage_latencies),
            "trad_avg_latency": np.mean(trad_latencies),
            "sage_wins": sage_wins,
            "trad_wins": trad_wins,
            "ties": ties,
            "n_queries": len(entries),
        })

    return pd.DataFrame(rows)


# ── Chart 1: Average LLM-Evaluation Scores ──────────────────────────────────

def plot_avg_scores(df: pd.DataFrame, out: Path):
    """Grouped bar chart: SAGE vs Traditional avg score per model combo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w / 2, df["sage_avg_score"], w, label="SAGE Graph RAG",
                   color=PALETTE["SAGE Graph RAG"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w / 2, df["trad_avg_score"], w, label="Traditional RAG",
                   color=PALETTE["Traditional RAG"], edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Score (out of 10)")
    ax.set_title("Average LLM-Evaluation Scores — SAGE vs Traditional RAG", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylim(0, 11)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Chart 2: Win / Tie / Loss Counts ────────────────────────────────────────

def plot_win_counts(df: pd.DataFrame, out: Path):
    """Stacked bar chart: SAGE wins / Traditional wins / Ties per combo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))

    ax.bar(x, df["sage_wins"], label="SAGE Wins", color=PALETTE["SAGE Graph RAG"],
           edgecolor="white", linewidth=0.8)
    ax.bar(x, df["ties"], bottom=df["sage_wins"], label="Tie", color="#95a5a6",
           edgecolor="white", linewidth=0.8)
    ax.bar(x, df["trad_wins"], bottom=df["sage_wins"] + df["ties"],
           label="Traditional Wins", color=PALETTE["Traditional RAG"],
           edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Number of Queries")
    ax.set_title("Win / Tie / Loss Counts per Model Combination", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Chart 3: Score Heatmaps (SAGE & Traditional side-by-side) ────────────────

def plot_score_heatmap(df: pd.DataFrame, out: Path):
    """Two heatmaps: LLM rows × Embedding columns for SAGE and Traditional."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pivot for SAGE
    sage_pivot = df.pivot_table(values="sage_avg_score", index="llm_model", columns="embedding_model")
    sns.heatmap(sage_pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=10,
                ax=ax1, cbar_kws={"label": "Score"}, linewidths=1, linecolor="white")
    ax1.set_title("SAGE Graph RAG\nAvg Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("LLM Model")
    ax1.set_xlabel("Embedding Model")

    # Pivot for Traditional
    trad_pivot = df.pivot_table(values="trad_avg_score", index="llm_model", columns="embedding_model")
    sns.heatmap(trad_pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=10,
                ax=ax2, cbar_kws={"label": "Score"}, linewidths=1, linecolor="white")
    ax2.set_title("Traditional RAG\nAvg Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")
    ax2.set_xlabel("Embedding Model")

    fig.suptitle("Score Heatmap — LLM × Embedding Model", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Chart 4: Average Latency ────────────────────────────────────────────────

def plot_latency(df: pd.DataFrame, out: Path):
    """Grouped bar chart: avg latency per system per model combo."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w / 2, df["sage_avg_latency"], w, label="SAGE Graph RAG",
                   color=PALETTE["SAGE Graph RAG"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w / 2, df["trad_avg_latency"], w, label="Traditional RAG",
                   color=PALETTE["Traditional RAG"], edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}s", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Average Response Latency — SAGE vs Traditional RAG", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Chart 5: Score Difference Distribution ───────────────────────────────────

def plot_score_distribution(results: List[Dict[str, Any]], out: Path):
    """Overlaid histograms of (SAGE − Traditional) score deltas per combo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups: Dict[str, List[float]] = {}
    for r in results:
        llm = r["llm_model"]
        emb = r["embedding_model"]
        label = combo_label(llm, emb)
        sage_score = r["llm_evaluation"].get("system1_score", 0)
        trad_score = r["llm_evaluation"].get("system2_score", 0)
        groups.setdefault(label, []).append(sage_score - trad_score)

    for i, (label, diffs) in enumerate(groups.items()):
        ax.hist(diffs, bins=np.arange(-10.5, 11.5, 1), alpha=0.5,
                label=label, color=COMBO_COLORS[i % len(COMBO_COLORS)], edgecolor="white")

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Score Difference (SAGE − Traditional)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Score Differences", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Chart 6: Per-Query Scores ────────────────────────────────────────────────

def plot_per_query_scores(results: List[Dict[str, Any]], out: Path):
    """Per-query SAGE vs Traditional scores across all model combos."""
    # Collect unique queries (in order)
    seen = set()
    queries = []
    for r in results:
        q = r["query"]
        if q not in seen:
            seen.add(q)
            queries.append(q)

    # Short query labels
    q_labels = [f"Q{i+1}" for i in range(len(queries))]

    # Collect groups
    groups: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        llm = r["llm_model"]
        emb = r["embedding_model"]
        label = combo_label(llm, emb)
        if label not in groups:
            groups[label] = {"sage": [], "trad": []}
        groups[label]["sage"].append(r["llm_evaluation"].get("system1_score", 0))
        groups[label]["trad"].append(r["llm_evaluation"].get("system2_score", 0))

    n_combos = len(groups)
    n_queries = len(queries)

    fig, axes = plt.subplots(n_combos, 1, figsize=(12, 4 * n_combos), sharex=True)
    if n_combos == 1:
        axes = [axes]

    for idx, (label, scores) in enumerate(groups.items()):
        ax = axes[idx]
        x = np.arange(n_queries)
        w = 0.35
        ax.bar(x - w / 2, scores["sage"][:n_queries], w, label="SAGE",
               color=PALETTE["SAGE Graph RAG"], edgecolor="white")
        ax.bar(x + w / 2, scores["trad"][:n_queries], w, label="Traditional",
               color=PALETTE["Traditional RAG"], edgecolor="white")
        ax.set_ylabel("Score")
        ax.set_title(label.replace("\n", " — "), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(np.arange(n_queries))
    axes[-1].set_xticklabels(q_labels, fontsize=9)
    axes[-1].set_xlabel("Query")

    fig.suptitle("Per-Query Scores Across Model Combinations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate comparison visualizations")
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR / "comparison_results.json"),
                        help="Path to the comparison results JSON file")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT_DIR / results_path

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    raw = load_results(results_path)
    results = filter_valid(raw)

    if not results:
        logger.error("No valid results to visualize after filtering errors!")
        return

    combo_df = build_combo_df(results)
    logger.info(f"Model combinations found: {len(combo_df)}")
    logger.info(f"\n{combo_df[['llm_model', 'embedding_model', 'sage_avg_score', 'trad_avg_score']].to_string()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all 6 images
    plot_avg_scores(combo_df, RESULTS_DIR / "avg_scores_comparison.png")
    plot_win_counts(combo_df, RESULTS_DIR / "win_counts_comparison.png")
    plot_score_heatmap(combo_df, RESULTS_DIR / "score_heatmap.png")
    plot_latency(combo_df, RESULTS_DIR / "latency_comparison.png")
    plot_score_distribution(results, RESULTS_DIR / "score_distribution.png")
    plot_per_query_scores(results, RESULTS_DIR / "per_query_scores.png")

    logger.info("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
