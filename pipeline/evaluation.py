import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CLASS_NAMES = ["No Stroke", "Stroke Type 1", "Stroke Type 2", "Stroke Type 3"]
SHORT_NAMES = ["No\nStroke", "S-Type1\nFH Topspin", "S-Type2\nBH Drive", "S-Type3\nFH Smash"]

# ######################### Load results #########################

def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


# ######################### 1. Confusion Matrix #########################

def plot_confusion_matrix(results: dict, out_dir: pathlib.Path):
    cm = np.array(results["global_confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # Normalised (row = recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        ax=axes[1], vmin=0, vmax=1, linewidths=0.5,
    )
    axes[1].set_title("Confusion Matrix — Normalised (Recall)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.suptitle(
        f"LOSO Evaluation — TTSwing Dataset  "
        f"(Acc={results['overall_accuracy']:.2%}  Macro-F1={results['overall_macro_f1']:.2%})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ######################### 2. Per-class metrics table #########################

def compute_per_class_metrics(results: dict) -> pd.DataFrame:
    cm = np.array(results["global_confusion_matrix"])
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = int(cm[i, :].sum())

        rows.append({
            "Class"    : name,
            "Precision": round(precision, 4),
            "Recall"   : round(recall, 4),
            "F1-Score" : round(f1, 4),
            "Support"  : support,
        })

    df = pd.DataFrame(rows)
    return df


def plot_per_class_metrics(metrics_df: pd.DataFrame, out_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(CLASS_NAMES))
    width = 0.25

    bars_p = ax.bar(x - width, metrics_df["Precision"], width, label="Precision", color="#4C72B0")
    bars_r = ax.bar(x,         metrics_df["Recall"],    width, label="Recall",    color="#55A868")
    bars_f = ax.bar(x + width, metrics_df["F1-Score"],  width, label="F1-Score",  color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Class Metrics — LOSO Evaluation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    path = out_dir / "per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ######################### 3. Per-fold F1 distribution #########################

def plot_fold_distribution(results: dict, out_dir: pathlib.Path):
    fold_df = pd.DataFrame(results["per_fold"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Accuracy per fold
    axes[0].bar(fold_df["subject_id"], fold_df["accuracy"], color="#4C72B0", alpha=0.8)
    axes[0].axhline(results["overall_accuracy"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean={results['overall_accuracy']:.2%}")
    axes[0].set_xlabel("Subject ID", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].set_title("Per-Subject Accuracy (LOSO)", fontsize=12, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Macro-F1 per fold
    axes[1].bar(fold_df["subject_id"], fold_df["macro_f1"], color="#55A868", alpha=0.8)
    axes[1].axhline(results["overall_macro_f1"], color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean={results['overall_macro_f1']:.2%}")
    axes[1].set_xlabel("Subject ID", fontsize=10)
    axes[1].set_ylabel("Macro F1", fontsize=10)
    axes[1].set_title("Per-Subject Macro-F1 (LOSO)", fontsize=12, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = out_dir / "fold_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ######################### 4. Print thesis-ready summary table #########################

def print_thesis_summary(results: dict, metrics_df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  THESIS RESULTS SUMMARY — TTSwing LOSO Evaluation")
    print("═" * 60)
    print(f"\n  Dataset       : TTSwing (97,350 samples, 93 subjects)")
    print(f"  Model         : MLP  [34 → 256 → 128 → 64 → 4]")
    print(f"  Evaluation    : Leave-One-Subject-Out (LOSO)")
    print(f"  Optimizer     : Adam  lr=0.001  weight_decay=0.0001")
    print(f"  Loss          : CrossEntropy (class-weighted)")
    print(f"\n  ── Overall ──────────────────────────────────────────")
    print(f"  Accuracy      : {results['overall_accuracy']:.4f}  "
          f"({results['overall_accuracy']:.2%})")
    print(f"  Macro-F1      : {results['overall_macro_f1']:.4f}  "
          f"({results['overall_macro_f1']:.2%})")
    print(f"\n  ── Per-Class ────────────────────────────────────────")
    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Support':>10}")
    print(f"  {'-'*58}")
    for _, row in metrics_df.iterrows():
        print(f"  {row['Class']:<18} {row['Precision']:>10.4f} "
              f"{row['Recall']:>10.4f} {row['F1-Score']:>10.4f} "
              f"{row['Support']:>10,}")
    print(f"\n  ── LLM Coaching ─────────────────────────────────────")
    print(f"  Model         : Mistral-7B-Instruct-v0.3 (local)")
    print(f"  Deployment    : LM Studio  http://localhost:1234")
    print(f"  Integration   : OpenAI-compatible REST API")
    print("═" * 60)


# ######################### Main #########################

if __name__ == "__main__":
    root      = pathlib.Path(__file__).resolve().parent.parent
    ckpt_dir  = root / "checkpoints"
    out_dir   = root / "results"
    out_dir.mkdir(exist_ok=True)

    results    = load_results(ckpt_dir / "loso_results.json")
    metrics_df = compute_per_class_metrics(results)

    plot_confusion_matrix(results, out_dir)
    plot_per_class_metrics(metrics_df, out_dir)
    plot_fold_distribution(results, out_dir)
    print_thesis_summary(results, metrics_df)

    # Save metrics table to CSV for thesis appendix
    csv_path = out_dir / "per_class_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics table saved → {csv_path}")