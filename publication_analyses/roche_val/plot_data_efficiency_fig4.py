"""Plot JAK1 data-efficiency barplots (Fig 4, JAK1 panel).

Reads a CSV of per-seed evaluation statistics across training-set fractions
and produces side-by-side AUROC and Spearman barplots, comparing FENNEC with
Orthrus embeddings vs. no-embedding baseline.

Expected CSV columns
--------------------
model_type            : str   (e.g. 'Orthrus_100bp', 'NoEmb')
jak1 Fraction         : float (fraction of JAK1 training data, 0..1)
seed                  : int/str
spearman              : float
auc                   : float
siRNA_concentration   : float (nM)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_TYPE_RENAME = {
    "Orthrus_100bp": "FENNEC Orthrus 5UTR+CDS+3UTR 100bp Windows",
    "Orthrus_center_100bp": "FENNEC Orthrus Center 100bp Window",
    "RNAFM_100bp": "FENNEC RNA-FM 5UTR+CDS+3UTR 100bp Windows",
    "NoEmb": "FENNEC No Embeddings",
}

KEEP_MODELS = [
    "FENNEC Orthrus 5UTR+CDS+3UTR 100bp Windows",
    "FENNEC No Embeddings",
]


def main():
    parser = argparse.ArgumentParser(
        description="Plot JAK1 data-efficiency curves (Fig 4)."
    )
    parser.add_argument(
        "stats_csv",
        help="Path to CSV with per-seed stats (model_type, jak1 Fraction, "
             "seed, spearman, auc, siRNA_concentration).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    args = parser.parse_args()

    plt.style.use("ggplot")

    df = pd.read_csv(args.stats_csv)
    df["model_type"] = df["model_type"].map(MODEL_TYPE_RENAME).fillna(df["model_type"])
    df = df[df["model_type"].isin(KEEP_MODELS)]

    concs = sorted(df["siRNA_concentration"].unique())
    n_concs = len(concs)

    fig, axes = plt.subplots(
        n_concs, 2, figsize=(12, 5 * n_concs), squeeze=False, dpi=200,
    )

    for idx, conc in enumerate(concs):
        tdf_c = df[df["siRNA_concentration"] == conc]

        sns.barplot(
            data=tdf_c, x="jak1 Fraction", y="auc", hue="model_type",
            ax=axes[idx, 0], alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
        )
        axes[idx, 0].set_title(
            f"AUROC at {conc} nM\n(error = +/-1.96 SE across seeds)"
        )
        axes[idx, 0].set_ylim(0.4, 1)
        axes[idx, 0].axhline(0.5, color="red", linestyle="--")
        axes[idx, 0].legend(fontsize=10)
        axes[idx, 0].set_ylabel("AUROC")

        sns.barplot(
            data=tdf_c, x="jak1 Fraction", y="spearman", hue="model_type",
            ax=axes[idx, 1], alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
        )
        axes[idx, 1].set_title(
            f"Spearman at {conc} nM\n(error = +/-1.96 SE across seeds)"
        )
        axes[idx, 1].set_ylim(-0.05, 1)
        axes[idx, 1].axhline(0.0, color="red", linestyle="--")
        axes[idx, 1].legend(fontsize=10)
        axes[idx, 1].set_ylabel("Spearman")

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
