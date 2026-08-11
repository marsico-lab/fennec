"""Plot ACVR1C data-efficiency barplots (Fig 4, ACVR1C panel).

Reads a CSV of per-seed evaluation statistics across training-set fractions
and produces side-by-side AUROC and Spearman barplots at a chosen siRNA
concentration (default 10 nM), comparing FENNEC with Orthrus embeddings
vs. no-embedding baseline.

Expected CSV columns
--------------------
model_type            : str   (e.g. 'Orthrus_100bp', 'NoEmb')
arc1 Fraction         : float (fraction of ACVR1C training data, 0..1)
seed                  : int/str
spearman              : float
auc                   : float
siRNA_concentration   : float (nM)
"""

import argparse
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
        description="Plot ACVR1C data-efficiency curves (Fig 4)."
    )
    parser.add_argument(
        "stats_csv",
        help="Path to CSV with per-seed stats (model_type, arc1 Fraction, "
             "seed, spearman, auc, siRNA_concentration).",
    )
    parser.add_argument(
        "--concentration", type=float, default=10.0,
        help="siRNA concentration in nM to plot (default: 10).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    args = parser.parse_args()

    plt.style.use("ggplot")

    df = pd.read_csv(args.stats_csv)
    df = df[df["siRNA_concentration"] == args.concentration].copy()
    df["model_type"] = df["model_type"].map(MODEL_TYPE_RENAME)
    df = df[df["model_type"].isin(KEEP_MODELS)]

    fig, (ax_auc, ax_spear) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    sns.barplot(
        data=df, x="arc1 Fraction", y="auc", hue="model_type",
        ax=ax_auc, alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
    )
    ax_auc.set_title(
        f"AUROC at {args.concentration} nM\n(error = +/-1.96 SE across seeds)"
    )
    ax_auc.set_ylim(0.4, 1)
    ax_auc.axhline(0.5, color="red", linestyle="--")
    ax_auc.legend(fontsize=10)
    ax_auc.set_ylabel("AUROC")

    sns.barplot(
        data=df, x="arc1 Fraction", y="spearman", hue="model_type",
        ax=ax_spear, alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
    )
    ax_spear.set_title(
        f"Spearman at {args.concentration} nM\n"
        f"(error = +/-1.96 SE across seeds)"
    )
    ax_spear.set_ylim(-0.05, 1)
    ax_spear.axhline(0.0, color="red", linestyle="--")
    ax_spear.legend(fontsize=10)
    ax_spear.set_ylabel("Spearman")

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
