"""Plot ACVR1C data-efficiency barplots at multiple concentrations (SFig 9).

Reads a CSV of per-seed evaluation statistics and produces AUROC and Spearman
barplots broken down by siRNA concentration (default: 0.1, 1, 10 nM),
showing all model types.

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


def main():
    parser = argparse.ArgumentParser(
        description="Plot ACVR1C data-efficiency per concentration (SFig 9)."
    )
    parser.add_argument(
        "stats_csv",
        help="Path to CSV with per-seed stats (model_type, arc1 Fraction, "
             "seed, spearman, auc, siRNA_concentration).",
    )
    parser.add_argument(
        "--concentrations", nargs="+", type=float, default=[0.1, 1, 10],
        help="siRNA concentrations to plot (default: 0.1 1 10).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to this path instead of showing interactively.",
    )
    args = parser.parse_args()

    plt.style.use("ggplot")

    df = pd.read_csv(args.stats_csv)
    df["model_type"] = df["model_type"].map(MODEL_TYPE_RENAME).fillna(df["model_type"])

    concs = sorted(args.concentrations)
    df = df[df["siRNA_concentration"].isin(concs)]

    n_concs = len(concs)
    fig, axes = plt.subplots(n_concs, 2, figsize=(14, 5 * n_concs))
    axes = np.array(axes).reshape(-1, 2)

    for idx, conc in enumerate(concs):
        tdf_c = df[df["siRNA_concentration"] == conc]

        sns.barplot(
            data=tdf_c, x="arc1 Fraction", y="auc", hue="model_type",
            ax=axes[idx, 0], alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
        )
        axes[idx, 0].set_title(
            f"AUROC at {conc} nM  (error = +/-1.96 SE across seeds)"
        )
        axes[idx, 0].set_ylim(0.4, 1)
        axes[idx, 0].axhline(0.5, color="red", linestyle="--")
        axes[idx, 0].legend(fontsize=7)
        axes[idx, 0].set_ylabel("AUROC")

        sns.barplot(
            data=tdf_c, x="arc1 Fraction", y="spearman", hue="model_type",
            ax=axes[idx, 1], alpha=0.7, errorbar=("se", 1.96), capsize=0.05,
        )
        axes[idx, 1].set_title(
            f"Spearman at {conc} nM  (error = +/-1.96 SE across seeds)"
        )
        axes[idx, 1].set_ylim(-0.05, 1)
        axes[idx, 1].axhline(0.0, color="red", linestyle="--")
        axes[idx, 1].legend(fontsize=7)
        axes[idx, 1].set_ylabel("Spearman")

    plt.suptitle(
        "FENNEC ACVR1C -- Model Comparison per Concentration", y=1.01
    )
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
