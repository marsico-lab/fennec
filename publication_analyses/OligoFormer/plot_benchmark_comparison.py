# Third-party benchmark code for OligoFormer comparison.
# OligoFormer: https://github.com/jerryji1993/OligoFormer
#
# Generates SFig panels: cross-validation performance bar plots
# and feature importance box plots for simple ML baselines trained
# on the Huesken dataset with OligoFormer / internal features.
"""
Plot cross-validation performance and feature importance for simple ML
models (LogisticRegression, RandomForest, XGBoost, SVM) on the Huesken
benchmark dataset.

Expects two CSVs:

1. --metrics CSV with columns:
       model, performance_metric, value
   where performance_metric is one of {accuracy, f1_score}.

2. --features CSV (optional) with columns:
       feature, importance, model, iteration
   for feature-importance box plots.

Example
-------
    python plot_benchmark_comparison.py \
        --metrics cv_metrics.csv \
        --features feature_importance.csv \
        --output_dir figures/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style():
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.5})
    sns.set_style(
        "ticks",
        {
            "axes.grid": False,
            "grid.linestyle": "",
            "font.family": "sans-serif",
            "text.color": "0",
            "xtick.color": "0",
            "ytick.color": "0",
        },
    )


def plot_cv_performance(met_df: pd.DataFrame, out: Path):
    """Horizontal grouped bar plot of accuracy and F1 by model."""
    fig, ax = plt.subplots(dpi=200)
    sns.barplot(
        data=met_df, y="model", x="value",
        hue="performance_metric", palette="Blues", alpha=0.5, ax=ax,
    )
    ax.set_xlim(0.6, 0.8)
    ax.set_ylabel("Model")
    ax.set_xlabel("Score")
    ax.legend(bbox_to_anchor=(1, 1))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="center", padding=0)
    ax.set_title("Cross-Validation Performance (Huesken Dataset)")
    fig.tight_layout()
    fig.savefig(out / "cv_performance.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feat_df: pd.DataFrame, out: Path):
    """Box plots of feature importance for each model in a 2x2 grid."""
    rf_df = feat_df[feat_df["model"] == "RandomForest"]
    if rf_df.empty:
        rf_df = feat_df
    sort_order = (
        rf_df.groupby("feature")["importance"]
        .median()
        .sort_values()
        .index
    )

    model_names = feat_df["model"].unique()
    n_models = len(model_names)
    ncols = min(n_models, 2)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 7 * nrows), dpi=200,
    )
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, mod in zip(axes, model_names):
        temp_df = feat_df[feat_df["model"] == mod].copy()
        temp_df["feature"] = pd.Categorical(
            temp_df["feature"], categories=sort_order, ordered=True,
        )
        sns.boxplot(data=temp_df, y="feature", x="importance", palette="rainbow", ax=ax)
        ax.set_title(mod)
        ax.set_xlabel("Importance")
        ax.set_ylabel("")

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_single_model_importance(
    feat_df: pd.DataFrame, out: Path, model_name: str = "RandomForest",
):
    """Single-model feature importance box plot."""
    sub = feat_df[feat_df["model"] == model_name]
    if sub.empty:
        return
    sort_order = sub.groupby("feature")["importance"].median().sort_values().index
    sub = sub.copy()
    sub["feature"] = pd.Categorical(sub["feature"], categories=sort_order, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 9), dpi=200)
    sns.boxplot(data=sub, y="feature", x="importance", palette="rainbow", ax=ax)
    ax.set_title(f"{model_name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / f"feature_importance_{model_name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metrics", required=True, help="CSV: model, performance_metric, value")
    parser.add_argument("--features", default=None, help="CSV: feature, importance, model, iteration")
    parser.add_argument("--output_dir", default=".", help="Directory for output figures")
    args = parser.parse_args()

    _setup_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    met_df = pd.read_csv(args.metrics)
    plot_cv_performance(met_df, out)

    if args.features:
        feat_df = pd.read_csv(args.features)
        feat_df["importance"] = feat_df["importance"].astype(float)
        plot_feature_importance(feat_df, out)
        plot_single_model_importance(feat_df, out, model_name="RandomForest")

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
