"""Compute and display zero-shot APP evaluation metrics.

Reads a predictions CSV produced by the evaluation pipeline and prints
a summary table of Spearman, AUROC, F1, and PR-AUC for the top-fraction
classification task. Optionally produces a scatter plot.

Expected CSV columns
--------------------
mean_pred             : float  (ensemble mean predicted KD)
True KD               : float  (experimentally measured mRNA remaining %)
model_condition       : str    (model label for grouping, optional)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, r2_score,
)


def calc_key_stats_frac(y_true, y_pred, frac=0.25):
    """Classify samples by top-fraction quantile and compute metrics."""
    thresh_true = np.quantile(y_true, frac)
    thresh_pred = np.quantile(y_pred, frac)
    y_true_cls = (y_true <= thresh_true).astype(int)
    y_pred_cls = (y_pred <= thresh_pred).astype(int)

    spear = round(spearmanr(y_true, y_pred).correlation, 4)
    auc = round(roc_auc_score(y_true_cls, y_pred_cls), 4)
    r2 = round(r2_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true_cls, y_pred_cls), 4)
    pr_auc = round(average_precision_score(y_true_cls, y_pred_cls), 4)
    return spear, auc, r2, f1, pr_auc, y_true_cls, y_pred_cls


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot APP evaluation table and scatter plot."
    )
    parser.add_argument(
        "predictions_csv",
        help="Path to predictions CSV (must contain 'mean_pred' and "
             "'True KD' columns).",
    )
    parser.add_argument(
        "--frac", type=float, default=0.25,
        help="Top-fraction for classification metrics (default: 0.25).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show a scatter plot of predicted vs. actual KD.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save scatter plot to this path.",
    )
    args = parser.parse_args()

    plt.style.use("ggplot")

    df = pd.read_csv(args.predictions_csv)

    target_col = "model_condition"
    groups = df[target_col].unique() if target_col in df.columns else ["all"]

    for group_label in groups:
        subset = (
            df[df[target_col] == group_label] if target_col in df.columns else df
        )
        subset = subset.dropna(subset=["True KD", "mean_pred"])

        spear, auc, r2, f1, pr_auc, y_true_cls, y_pred_cls = (
            calc_key_stats_frac(
                subset["True KD"].to_numpy(),
                subset["mean_pred"].to_numpy(),
                frac=args.frac,
            )
        )

        print(f"\n--- {group_label} (N={len(subset)}) ---")
        print(f"  Spearman          : {spear}")
        print(f"  Top {args.frac*100:.0f}% AUROC   : {auc}")
        print(f"  Top {args.frac*100:.0f}% F1       : {f1}")
        print(f"  Top {args.frac*100:.0f}% PR-AUC   : {pr_auc}")
        print(f"  R2                : {r2}")

    if args.plot or args.output:
        n_groups = len(groups)
        n_cols = min(n_groups, 3)
        n_rows = (n_groups + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows),
            squeeze=False, sharex=True, sharey=True,
        )
        axes_flat = axes.flatten()

        for idx, group_label in enumerate(groups):
            ax = axes_flat[idx]
            subset = (
                df[df[target_col] == group_label]
                if target_col in df.columns else df
            ).dropna(subset=["True KD", "mean_pred"])

            ax.scatter(
                subset["mean_pred"], subset["True KD"],
                alpha=0.7, edgecolor="none", s=20,
            )
            ax.set_title(group_label)
            ax.set_xlabel("Predicted KD")
            ax.set_ylabel("Actual KD")

            if not subset.empty:
                xy_min = min(
                    subset["mean_pred"].min(), subset["True KD"].min()
                )
                xy_max = max(
                    subset["mean_pred"].max(), subset["True KD"].max()
                )
                ax.plot(
                    [xy_min, xy_max], [xy_min, xy_max],
                    linestyle="--", color="black", linewidth=1,
                )

            spear, auc, r2, f1, pr_auc, _, _ = calc_key_stats_frac(
                subset["True KD"].to_numpy(),
                subset["mean_pred"].to_numpy(),
                frac=args.frac,
            )
            summary = (
                f"N={len(subset)}\n"
                f"Spearman: {spear:.2f}\n"
                f"Top {args.frac*100:.0f}% AUC: {auc:.2f}\n"
                f"Top {args.frac*100:.0f}% F1: {f1:.2f}\n"
                f"Top {args.frac*100:.0f}% PR-AUC: {pr_auc:.2f}"
            )
            ax.text(
                0.02, 0.98, summary, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        for j in range(len(groups), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout()

        if args.output:
            fig.savefig(args.output, bbox_inches="tight")
            print(f"\nSaved plot to {args.output}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
