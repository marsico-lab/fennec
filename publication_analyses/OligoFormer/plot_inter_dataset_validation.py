# Third-party benchmark code for OligoFormer comparison.
# OligoFormer: https://github.com/jerryji1993/OligoFormer
#
# Generates SFig panels: inter-dataset validation bar plots and
# dataset PCA/UMAP projections for OligoFormer feature-based models.
"""
Plot inter-dataset validation performance and dataset projections.

Expects a CSV (--metrics) with columns:
    model, dataset, accuracy, f1_score, auc

Optionally accepts a features CSV (--features_csv) with columns:
    <feature columns...>, data_source, y
for PCA / UMAP projection plots.

Example
-------
    python plot_inter_dataset_validation.py \
        --metrics inter_dataset_metrics.csv \
        --features_csv all_datasets_features.csv \
        --output_dir figures/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_inter_dataset_f1(met_df: pd.DataFrame, out: Path):
    """Bar plot of F1 score by model and dataset."""
    fig, ax = plt.subplots(dpi=200)
    ax.set_title("OligoFormer + Internal Features Inter-dataset Validation")
    sns.barplot(data=met_df, x="model", y="f1_score", hue="dataset", palette="crest", ax=ax)
    ax.legend(bbox_to_anchor=(1, 1))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", padding=0)
    fig.tight_layout()
    fig.savefig(out / "inter_dataset_f1.png", bbox_inches="tight")
    plt.close(fig)


def plot_inter_dataset_auc(met_df: pd.DataFrame, out: Path):
    """Bar plot of AUC by model and dataset."""
    fig, ax = plt.subplots(dpi=200)
    ax.set_title("OligoFormer + Internal Features Inter-dataset Validation")
    sns.barplot(data=met_df, x="model", y="auc", hue="dataset", palette="crest", ax=ax)
    ax.legend(bbox_to_anchor=(1, 1))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", padding=0)
    fig.tight_layout()
    fig.savefig(out / "inter_dataset_auc.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca_projection(features_df: pd.DataFrame, out: Path):
    """PCA scatter plot coloured by data source."""
    from sklearn.decomposition import PCA

    meta_cols = ["data_source", "y"]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    X = features_df[feat_cols].astype(float)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
    evr1, evr2 = pca.explained_variance_ratio_

    proj_df = pd.DataFrame(embedding, columns=["PC1", "PC2"])
    proj_df["data_source"] = features_df["data_source"].values
    proj_df["y"] = features_df["y"].values

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))
    sns.scatterplot(
        data=proj_df, x="PC1", y="PC2",
        hue="data_source", style="y", alpha=0.3, marker=".", ax=ax,
    )
    ax.set_xlabel(f"PC1 ({round(evr1 * 100, 1)}%)")
    ax.set_ylabel(f"PC2 ({round(evr2 * 100, 1)}%)")
    ax.set_title("PCA Projection")
    fig.tight_layout()
    fig.savefig(out / "pca_projection.png", bbox_inches="tight")
    plt.close(fig)


def plot_umap_projection(features_df: pd.DataFrame, out: Path):
    """UMAP scatter plot coloured by data source (requires umap-learn)."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed, skipping UMAP plot")
        return

    meta_cols = ["data_source", "y"]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    X = features_df[feat_cols].astype(float)

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    proj_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    proj_df["data_source"] = features_df["data_source"].values
    proj_df["y"] = features_df["y"].values

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))
    sns.scatterplot(
        data=proj_df, x="UMAP1", y="UMAP2",
        hue="data_source", style="y", alpha=0.3, marker=".", ax=ax,
    )
    ax.set_title("UMAP Projection")
    fig.tight_layout()
    fig.savefig(out / "umap_projection.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics", required=True,
        help="CSV with model, dataset, accuracy, f1_score, auc columns",
    )
    parser.add_argument(
        "--features_csv", default=None,
        help="CSV with feature columns + data_source + y for projections",
    )
    parser.add_argument("--output_dir", default=".", help="Directory for output figures")
    args = parser.parse_args()

    _setup_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    met_df = pd.read_csv(args.metrics)
    plot_inter_dataset_f1(met_df, out)
    if "auc" in met_df.columns:
        plot_inter_dataset_auc(met_df, out)

    if args.features_csv:
        feat_df = pd.read_csv(args.features_csv)
        plot_pca_projection(feat_df, out)
        plot_umap_projection(feat_df, out)

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
