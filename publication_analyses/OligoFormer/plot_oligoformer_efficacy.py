# Third-party benchmark code for OligoFormer comparison.
# OligoFormer: https://github.com/jerryji1993/OligoFormer
#
# Generates SFig panels: OligoFormer predicted efficacy vs true efficacy
# (scatter plots, confusion matrices, prediction histograms).
"""
Plot OligoFormer prediction quality on the Huesken (Hu) dataset.

Expects a CSV with columns:
    label   -- true efficacy (float, 0-1)
    y       -- binary class label (0 or 1)
    efficacy -- OligoFormer predicted efficacy (float)

Example
-------
    python plot_oligoformer_efficacy.py --input merged_hu_oligo.csv --output_dir figures/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


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


def plot_scatter(df, out, correction=1.0, tag=""):
    """Scatter plot of predicted vs true efficacy."""
    plot_df = df.copy()
    plot_df["efficacy"] = plot_df["efficacy"] / correction

    fig, ax = plt.subplots(dpi=200)
    sns.scatterplot(
        data=plot_df, x="efficacy", y="label", hue="y",
        marker=".", alpha=0.5, ax=ax,
    )
    ax.set_ylabel("True Efficacy")
    ax.set_xlabel("Predicted Efficacy")
    title = "Predicted vs True Efficacy"
    if correction != 1.0:
        title += f" (corrected / {correction})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / f"scatter_efficacy{tag}.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(df, out, correction=1.0, tag=""):
    """Confusion-matrix heatmap at threshold=0.5."""
    y_true = df["y"]
    y_pred = (df["efficacy"] / correction >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{labels[i][j]}\n{cm[i, j]}"

    f1 = f1_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"], ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    title = f"Confusion Matrix (F1={f1:.3f})"
    if correction != 1.0:
        title += f" [corrected / {correction}]"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix{tag}.png", bbox_inches="tight")
    plt.close(fig)


def plot_histogram(df, out, correction=1.0, tag=""):
    """Histogram of predicted efficacy."""
    fig, ax = plt.subplots(dpi=200)
    (df["efficacy"] / correction).plot.hist(color="black", bins=10, ax=ax)
    ax.set_xlabel("Predicted Efficacy")
    title = "OligoFormer Predicted Efficacy Distribution"
    if correction != 1.0:
        title += f" (/ {correction})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / f"hist_efficacy{tag}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="CSV with label, y, efficacy columns")
    parser.add_argument("--output_dir", default=".", help="Directory for output figures")
    parser.add_argument(
        "--correction", type=float, default=1.341,
        help="OligoFormer output correction factor (default: 1.341)",
    )
    args = parser.parse_args()

    _setup_style()
    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Uncorrected
    plot_scatter(df, out, correction=1.0, tag="_uncorrected")
    plot_confusion(df, out, correction=1.0, tag="_uncorrected")
    plot_histogram(df, out, correction=1.0, tag="_uncorrected")

    # Corrected
    if args.correction != 1.0:
        plot_scatter(df, out, correction=args.correction, tag="_corrected")
        plot_confusion(df, out, correction=args.correction, tag="_corrected")
        plot_histogram(df, out, correction=args.correction, tag="_corrected")

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
