#!/usr/bin/env python
"""Plot dataset overview panels for Fig 2.

Produces four figure files:
  1. gene_distribution.png  -- Two-row stacked bar chart of gene counts by
     data source (Patent / Huesken / Ichihara / Takayuki).
  2. categorical_summaries.png -- Horizontal bar charts for transfection
     method, siRNA concentration, cell line, and duration.
  3. mrna_remaining_distribution.png -- Histogram of mRNA remaining (%).
  4. composition.png -- Strand-resolved nucleotide and chemical-modification
     composition (4-panel: nucleotides, nucleotide class, modification type,
     backbone chemistry).

Usage
-----
    python plot_dataset_overview.py \
        --data-path /path/to/train_data.csv.gz \
        --output-dir ./figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from fennec.featurizer import categorize_component

# ── Label harmonisation maps ────────────────────────────────────────────────
MODIFICATION_MAP = {
    "f": "2'-F",
    "hd": "Hexadecyl",
    "": "Unmodified",
    "2OMe": "2'-OMe",
    "da": "Docosyloxy",
}
NUCLEOTIDE_CLASS_MAP = {"r": "RNA", "d": "DNA", "gn": "Glycol"}
BACKBONE_MAP = {
    "P": "Phosphate",
    "p": "Phosphate",
    "s": "Phosphorothioate",
    "VP": "Vinyl-Phosphate",
    "2p": "2-Phosphate",
    "L96": "GalNAc",
}


# ── Shared style ────────────────────────────────────────────────────────────
def _apply_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "savefig.transparent": False,
            "axes.facecolor": "whitesmoke",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "text.color": "#1a1a1a",
        }
    )


# ── 1. Gene distribution ────────────────────────────────────────────────────
def plot_gene_distribution(data: pd.DataFrame, output_dir: str) -> None:
    """Two-row stacked bar chart: patent genes (top) vs other genes (bottom)."""
    gene_counts_by_source = data.groupby("meta_group")[
        "gene_target_symbol_name"
    ].nunique()
    plot_df = data.copy()
    plot_df["source_group"] = (
        plot_df["meta_group"]
        .fillna("Other/Unknown")
        .replace(
            {
                "Huesken": f'Huesken (N={gene_counts_by_source.get("Huesken", 0)})',
                "ichihara_mix": f'Ichihara (N={gene_counts_by_source.get("ichihara_mix", 0)})',
                "Takayuki": f'Takayuki (N={gene_counts_by_source.get("Takayuki", 0)})',
                "Patent": f'Patent (N={gene_counts_by_source.get("Patent", 0)})',
            }
        )
    )

    gene_order = plot_df["gene_target_symbol_name"].value_counts().index
    counts_by_source = (
        plot_df.groupby(["gene_target_symbol_name", "source_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(gene_order)
    )

    source_palette = {
        "Patent": "#082235",
        "Huesken": "#ff7f0e",
        "Ichihara": "#2ca02c",
        "Takayuki": "#d62728",
        "Other/Unknown": "#7f7f7f",
    }

    patent_col = next(
        (c for c in counts_by_source.columns if c.startswith("Patent")), None
    )
    if patent_col is None:
        patent_genes = counts_by_source.index
        other_genes = pd.Index([])
    else:
        patent_genes = counts_by_source.index[counts_by_source[patent_col] > 0]
        other_genes = counts_by_source.index[counts_by_source[patent_col] == 0]

    patent_data = counts_by_source.loc[patent_genes]
    other_data = counts_by_source.loc[other_genes]
    colors = [
        source_palette[s.split(" (N=")[0]] for s in counts_by_source.columns
    ]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        dpi=200,
        sharey=True,
        sharex=False,
        gridspec_kw={"hspace": 0.6},
    )

    patent_data.plot(
        kind="bar", stacked=True, color=colors, width=0.9, ax=ax1, legend=False
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax1.set_ylabel("siRNA datapoints #", fontsize=16)
    ax1.set_xlabel(f"Patent Genes (N={len(patent_genes)})", fontsize=16)
    ax1.set_title("Patent", fontsize=18)

    if len(other_genes) > 0:
        other_data.plot(
            kind="bar", stacked=True, color=colors, width=0.9, ax=ax2, legend=False
        )
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=45, ha="right", fontsize=11
        )
        ax2.set_ylabel("siRNA datapoints #", fontsize=16)
        ax2.set_xlabel(f"Other Genes (N={len(other_genes)})", fontsize=16)
        ax2.set_title(
            "Other (Huesken / Ichihara / Takayuki)", fontsize=18, y=0.95
        )

    legend_handles = [
        Patch(facecolor=c, label=s)
        for c, s in zip(colors, counts_by_source.columns)
    ]
    ax1.legend(handles=legend_handles, title="Source", ncol=2, frameon=True)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "gene_distribution.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


# ── 2. Categorical summaries ────────────────────────────────────────────────
def plot_categorical_summaries(data: pd.DataFrame, output_dir: str) -> None:
    """Horizontal bar charts for key categorical features."""
    targets = [
        ("Transfection_method", "Transfection method"),
        ("siRNA_concentration", "siRNA concentration"),
        ("cell_line_donor", "Cell line / donor"),
        ("Duration_after_transfection_h", "Duration (h)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes_flat = axes.flatten()

    for idx, (column, label) in enumerate(targets):
        ax = axes_flat[idx]
        counts = data[column].value_counts(dropna=False)
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="gray")
        ax.set_title(label)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ytick_size = 12 if column == "cell_line_donor" else 20
        ax.tick_params(axis="y", labelsize=ytick_size)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Key Categorical Summaries", fontsize=18, y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "categorical_summaries.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# ── 3. mRNA remaining distribution ──────────────────────────────────────────
def plot_mrna_distribution(data: pd.DataFrame, output_dir: str) -> None:
    """Histogram of mRNA remaining percentage with median line."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=300)
    series = data["mRNA_remaining_pct"].dropna()
    sns.histplot(series, bins=100, kde=True, ax=ax, color="#4C72B0")
    if not series.empty:
        median_val = series.median()
        ax.axvline(median_val, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.text(
            median_val,
            ax.get_ylim()[1] * 0.95,
            "Median",
            color="#C44E52",
            ha="right",
            va="top",
            fontsize=11,
            rotation=90,
        )
    ax.set_xlabel("mRNA remaining (%)", fontsize=14)
    ax.set_ylabel("siRNA datapoints (N)", fontsize=14)
    ax.set_xlim(0, 200)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mrna_remaining_distribution.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


# ── 4. Strand-resolved composition ──────────────────────────────────────────
def _categorize_strand(sequences: pd.Series) -> tuple:
    """Run categorize_component on every token of every sequence in a strand."""
    nucs, nuc_types, mod_types, backbones = [], [], [], []
    for seq in sequences:
        seq = seq.replace(" p", "")
        for token in seq.split(" "):
            nu, nu_type, mod_t, backb = categorize_component(token)
            if token in BACKBONE_MAP:
                nu, nu_type, mod_t = None, None, None
            nucs.append(nu)
            nuc_types.append(nu_type)
            mod_types.append(mod_t)
            backbones.append(backb)
    return nucs, nuc_types, mod_types, backbones


def plot_composition(data: pd.DataFrame, output_dir: str) -> None:
    """Strand-resolved composition of siRNA components (2x2 panel)."""
    g_nuc, g_type, g_mod, g_bb = _categorize_strand(data["siRNA_list"])
    p_nuc, p_type, p_mod, p_bb = _categorize_strand(data["siRNA_passenger_list"])

    records = []
    for nu, nt, mod, bb in zip(g_nuc, g_type, g_mod, g_bb):
        records.append(
            {"strand": "Guide", "nuc": nu, "nuc_type": nt, "mod_type": mod, "backbone": bb}
        )
    for nu, nt, mod, bb in zip(p_nuc, p_type, p_mod, p_bb):
        records.append(
            {"strand": "Passenger", "nuc": nu, "nuc_type": nt, "mod_type": mod, "backbone": bb}
        )

    comp_df = pd.DataFrame(records)
    comp_df["nuc"] = comp_df["nuc"].replace("", None)
    comp_df["strand"] = pd.Categorical(
        comp_df["strand"], categories=["Guide", "Passenger"], ordered=True
    )
    comp_df["mod_type"] = comp_df["mod_type"].map(
        lambda x: MODIFICATION_MAP.get(x, x)
    )
    comp_df["nuc_type"] = comp_df["nuc_type"].map(
        lambda x: NUCLEOTIDE_CLASS_MAP.get(x, x)
    )
    comp_df["backbone"] = comp_df["backbone"].map(
        lambda x: BACKBONE_MAP.get(x, x)
    )

    summary_specs = [
        ("nuc", "Canonical Nucleotides"),
        ("nuc_type", "Nucleotide Class"),
        ("mod_type", "Modification Type"),
        ("backbone", "Backbone Chemistry"),
    ]

    strand_palette = {"Guide": "#0098b3", "Passenger": "#c252527c"}
    sns.set_theme(style="whitegrid", context="talk")
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "axes.facecolor": "whitesmoke",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            "text.color": "#1a1a1a",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 8), dpi=300)
    axes_flat = axes.flatten()

    for ax, (feature, title) in zip(axes_flat, summary_specs):
        counts = (
            comp_df.groupby([feature, "strand"]).size().reset_index(name="count")
        )
        order = (
            counts.groupby(feature)["count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        sns.barplot(
            data=counts,
            y=feature,
            x="count",
            hue="strand",
            order=order,
            hue_order=strand_palette.keys(),
            palette=strand_palette,
            ax=ax,
            saturation=0.9,
            alpha=0.75,
            dodge=True,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3, fontsize=13)
        ax.set_title(title, pad=22, fontsize=30)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.legend_.set_title("")
        ax.set_xscale("log")
        ax.set_yticklabels(ax.get_yticklabels(), size=25)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.legend(bbox_to_anchor=(0.70, 0), loc="lower left")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "composition.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dataset overview panels for Fig 2."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data CSV (.csv or .csv.gz).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where figure PNGs are saved (default: current dir).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _apply_style()

    compression = "gzip" if args.data_path.endswith(".gz") else None
    data = pd.read_csv(
        args.data_path, index_col=0, low_memory=False, compression=compression
    )

    # Harmonise cell-line labels
    data["cell_line_donor"] = (
        data["cell_line_donor"]
        .replace(r"cyno|Cynomolgous", "Primary Monkey Hepatocytes", regex=True)
        .fillna("Unknown_cell_line")
        .replace("Unknown_cell_line", "NA")
    )
    data["Transfection_method"] = data["Transfection_method"].str.strip()
    data["siRNA_concentration"] = data["siRNA_concentration"].fillna("NA")

    print("Plotting gene distribution ...")
    plot_gene_distribution(data, args.output_dir)

    print("Plotting categorical summaries ...")
    plot_categorical_summaries(data, args.output_dir)

    print("Plotting mRNA remaining distribution ...")
    plot_mrna_distribution(data, args.output_dir)

    print("Plotting strand-resolved composition ...")
    plot_composition(data, args.output_dir)

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
