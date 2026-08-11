#!/usr/bin/env python
"""Plot embedding-correlation analysis for Supplementary Figures 1-3.

Produces five figure files:
  1. pca_scatter.png        -- PCA scatter of Orthrus embeddings coloured
                               by gene target.
  2. explained_variance.png -- Bar chart of PCA explained variance ratios
                               (requires --explained-variance-path).
  3. spearman_heatmap.png   -- Spearman correlation heatmap between PCA
                               components and meta features, split into
                               gene-level (static) and siRNA-level (dynamic).
  4. mi_heatmap.png         -- Mutual-information heatmap with the same
                               static / dynamic split.
  5. top_correlations.png   -- Bar chart of the top 80 most strongly
                               correlated (PCA component, meta feature) pairs.

Input
-----
A CSV file whose columns include:

  * ``gene_target_symbol_name`` -- gene labels for colouring / grouping.
  * ``PCA_1``, ``PCA_2``, ... , ``PCA_N`` -- pre-computed PCA coordinates
    (column names are auto-detected).
  * Any remaining numeric columns are treated as meta features.

Optionally, a JSON file containing an array of explained-variance ratios
(one float per PCA component) for the explained-variance bar chart.

Usage
-----
    python plot_embedding_correlations.py \
        --data-path pca_meta_features.csv \
        --explained-variance-path explained_variance.json \
        --output-dir ./figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.feature_selection import mutual_info_regression

# Features excluded from correlation analysis (experiment-level or redundant)
DROP_COLS = [
    "siRNA_concentration",
    "Duration_after_transfection_h",
    "homodimer_dG",
    "heterodimer_dG",
    "seed_toxicity",
    "tm",
    "ends_3nt_tm_asymetry",
    "ends_2nt_tm_asymetry",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _detect_pca_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("PCA_")],
        key=lambda c: int(c.split("_")[1]),
    )


def _detect_meta_cols(
    df: pd.DataFrame, pca_cols: list[str]
) -> list[str]:
    skip = set(pca_cols) | {"gene_target_symbol_name"} | set(DROP_COLS)
    return [
        c
        for c in df.columns
        if c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _cluster_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if matrix.empty:
        return matrix
    if matrix.shape[1] > 1:
        d = pdist(matrix.abs().fillna(0).T, metric="euclidean")
        col_order = matrix.columns[leaves_list(linkage(d, method="average"))]
    else:
        col_order = matrix.columns
    if matrix.shape[0] > 1:
        d = pdist(matrix.abs().fillna(0), metric="euclidean")
        row_order = matrix.index[leaves_list(linkage(d, method="average"))]
    else:
        row_order = matrix.index
    return matrix.loc[row_order, col_order]


def _build_spearman_matrix(
    frame: pd.DataFrame,
    pca_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for pca_col in pca_cols:
        corr_values = {}
        for feat in feature_cols:
            joined = frame[[pca_col, feat]].dropna()
            corr_values[feat] = (
                joined[pca_col].corr(joined[feat], method="spearman")
                if len(joined) > 2
                else np.nan
            )
        rows.append(pd.Series(corr_values, name=pca_col))
    return pd.DataFrame(rows)


def _build_mi_matrix(
    frame: pd.DataFrame,
    pca_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for pca_col in pca_cols:
        joined = frame[[pca_col, *feature_cols]].dropna()
        if len(joined) < 5:
            rows.append(pd.Series(np.nan, index=feature_cols, name=pca_col))
            continue
        mi = mutual_info_regression(
            joined[feature_cols], joined[pca_col], random_state=42
        )
        rows.append(pd.Series(mi, index=feature_cols, name=pca_col))
    return pd.DataFrame(rows)


def _split_static_dynamic(
    analysis_df: pd.DataFrame,
    pca_cols: list[str],
    numeric_meta_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Split meta features into gene-level (static) and siRNA-level (dynamic).

    A feature is 'static' if >= 95 % of genes have at most one unique value for
    it (i.e. it varies across genes but not within a gene).
    """
    per_gene_nunique = analysis_df.groupby("gene_target_symbol_name")[
        numeric_meta_cols
    ].nunique(dropna=True)
    static_cols = [
        col
        for col in numeric_meta_cols
        if (per_gene_nunique[col] <= 1).mean() >= 0.95
    ]
    dynamic_cols = [c for c in numeric_meta_cols if c not in static_cols]

    per_gene_df = analysis_df.groupby("gene_target_symbol_name", as_index=False).agg(
        {
            **{pc: "median" for pc in pca_cols},
            **{col: "median" for col in static_cols},
        }
    )
    return per_gene_df, static_cols, dynamic_cols


# ── Plotting functions ───────────────────────────────────────────────────────
def plot_pca_scatter(
    df: pd.DataFrame,
    pca_cols: list[str],
    output_dir: str,
    explained_variance: list[float] | None = None,
) -> None:
    """PCA scatter coloured by gene (top 30)."""
    gene_series = df["gene_target_symbol_name"]
    top_genes = gene_series.value_counts().head(30).index
    display_labels = gene_series.where(gene_series.isin(top_genes), "Other Genes")

    palette = dict(
        zip(top_genes, sns.color_palette("Spectral", n_colors=len(top_genes)))
    )
    palette["Other Genes"] = "#b7b7b7"

    plot_df = pd.DataFrame(
        {
            "PCA_1": df[pca_cols[0]],
            "PCA_2": df[pca_cols[1]],
            "Gene": display_labels.values,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300, facecolor="white")
    sns.scatterplot(
        data=plot_df,
        x="PCA_1",
        y="PCA_2",
        hue="Gene",
        palette=palette,
        s=7,
        alpha=0.4,
        linewidth=0.15,
        edgecolor="white",
        ax=ax,
        rasterized=True,
    )

    xlabel = "PCA 1"
    ylabel = "PCA 2"
    if explained_variance is not None and len(explained_variance) >= 2:
        xlabel = f"PCA 1 ({explained_variance[0]:.2%})"
        ylabel = f"PCA 2 ({explained_variance[1]:.2%})"

    ax.set_title(
        "PCA of local mRNA Language Model Embeddings", fontsize=20, pad=1, alpha=0.8
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(False)

    legend = ax.legend(
        title="Gene (Top 30 by count)",
        loc="upper right",
        title_fontsize=7,
        bbox_to_anchor=(1.2, 1),
        frameon=True,
        framealpha=0.95,
        fontsize=6,
    )
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_facecolor("white")

    fig.savefig(
        os.path.join(output_dir, "pca_scatter.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig)


def plot_explained_variance(
    variance_ratios: list[float], output_dir: str
) -> None:
    """Bar chart of PCA explained variance with cumulative annotations."""
    n = len(variance_ratios)
    component_names = [f"PCA {i + 1}" for i in range(n)]
    cumulative = np.cumsum(variance_ratios)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    barplot = sns.barplot(x=component_names, y=variance_ratios, ax=ax, palette="viridis")

    for i, p in enumerate(barplot.patches):
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            f"Cumulative: {cumulative[i]:.0%}",
            fontsize=9,
            color="dimgrey",
            ha="center",
            va="bottom",
            rotation=90,
        )

    ax.set_title("Explained Variance by Principal Component", fontsize=16)
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_xlabel("Principal Component")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "explained_variance.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


def plot_spearman_heatmap(
    analysis_df: pd.DataFrame,
    per_gene_df: pd.DataFrame,
    pca_cols: list[str],
    static_cols: list[str],
    dynamic_cols: list[str],
    output_dir: str,
) -> None:
    """Split Spearman correlation heatmap (gene-level vs siRNA-level)."""
    static_corr = _cluster_matrix(
        _build_spearman_matrix(per_gene_df, pca_cols, static_cols)
    )
    dynamic_corr = _cluster_matrix(
        _build_spearman_matrix(analysis_df, pca_cols, dynamic_cols)
    )

    static_max = static_corr.abs().max().max() if not static_corr.empty else 0
    dynamic_max = dynamic_corr.abs().max().max() if not dynamic_corr.empty else 0

    ncols_static = max(1, len(static_corr.columns))
    ncols_dynamic = max(1, len(dynamic_corr.columns))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10, 0.55 * ncols_static + 0.35 * ncols_dynamic), 8),
        dpi=200,
        gridspec_kw={"width_ratios": [ncols_static, ncols_dynamic]},
    )

    panels = [
        (axes[0], static_corr, "Gene-level static features\nSpearman", static_max),
        (
            axes[1],
            dynamic_corr,
            "siRNA-centred embeddings + UTR slices\ndynamic features Spearman",
            dynamic_max,
        ),
    ]

    for ax, matrix, title, max_abs in panels:
        if matrix.empty:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(no usable features)")
            continue
        sns.heatmap(
            matrix,
            cmap="vlag",
            center=0,
            annot=True,
            vmin=-max_abs,
            vmax=max_abs,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            cbar_kws={"label": "Spearman correlation"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Meta Features")
        ax.set_ylabel("PCA Components")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    plt.suptitle(
        "Orthrus PCA Component vs. Meta Feature Spearman Correlations", y=1.02
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "spearman_heatmap.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


def plot_mi_heatmap(
    analysis_df: pd.DataFrame,
    per_gene_df: pd.DataFrame,
    pca_cols: list[str],
    static_cols: list[str],
    dynamic_cols: list[str],
    output_dir: str,
) -> None:
    """Split mutual-information heatmap (gene-level vs siRNA-level)."""
    static_mi = _cluster_matrix(
        _build_mi_matrix(per_gene_df, pca_cols, static_cols)
    )
    dynamic_mi = _cluster_matrix(
        _build_mi_matrix(analysis_df, pca_cols, dynamic_cols)
    )

    ncols_static = max(1, len(static_mi.columns))
    ncols_dynamic = max(1, len(dynamic_mi.columns))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10, 0.55 * ncols_static + 0.35 * ncols_dynamic), 8),
        dpi=200,
        gridspec_kw={"width_ratios": [ncols_static, ncols_dynamic]},
    )

    panels = [
        (axes[0], static_mi, "Gene-level static features MI"),
        (
            axes[1],
            dynamic_mi,
            "siRNA-centred embeddings + UTR slices\ndynamic features MI",
        ),
    ]

    for ax, matrix, title in panels:
        if matrix.empty:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(no usable features)")
            continue
        sns.heatmap(
            matrix,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            cbar_kws={"label": "Mutual Information"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Meta Features")
        ax.set_ylabel("PCA Components")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    plt.suptitle(
        "Orthrus PCA Component vs. Meta Feature Mutual Information", y=1.02
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mi_heatmap.png"), bbox_inches="tight", dpi=200
    )
    plt.close(fig)


def plot_top_correlations(
    analysis_df: pd.DataFrame,
    per_gene_df: pd.DataFrame,
    pca_cols: list[str],
    static_cols: list[str],
    dynamic_cols: list[str],
    output_dir: str,
    top_n: int = 80,
) -> None:
    """Bar chart of the top-N most strongly correlated pairs."""
    # Combine static (gene-level) and dynamic (siRNA-level) correlations
    static_corr = _build_spearman_matrix(per_gene_df, pca_cols, static_cols)
    dynamic_corr = _build_spearman_matrix(analysis_df, pca_cols, dynamic_cols)
    combined = pd.concat([static_corr, dynamic_corr], axis=1)
    sorted_corr = _cluster_matrix(combined)

    corr_melt = (
        sorted_corr.stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_0": "PCA_component", "level_1": "meta_feature"})
    )
    corr_melt["abs_value"] = corr_melt["value"].abs()
    top_pairs = corr_melt.nlargest(top_n, "abs_value").copy()

    feature_levels = [
        f for f in sorted_corr.columns if f in top_pairs["meta_feature"].unique()
    ]
    component_levels = [
        c for c in sorted_corr.index if c in top_pairs["PCA_component"].unique()
    ]

    top_pairs["meta_feature"] = pd.Categorical(
        top_pairs["meta_feature"], categories=feature_levels, ordered=True
    )
    top_pairs["PCA_component"] = pd.Categorical(
        top_pairs["PCA_component"], categories=component_levels, ordered=True
    )
    top_pairs = top_pairs.sort_values(["meta_feature", "PCA_component"])

    fig = plt.figure(figsize=(10, 6), dpi=200)
    sns.barplot(
        data=top_pairs,
        x="meta_feature",
        y="value",
        hue="PCA_component",
        order=feature_levels,
        hue_order=component_levels,
        dodge=True,
        palette="Spectral",
        alpha=0.7,
    )
    for i in range(1, len(feature_levels)):
        plt.axvline(i - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.title(
        f"Top {top_n} Most Highly Correlated Feature x PCA Component Pairs"
    )
    plt.xlabel("Meta Feature")
    plt.ylabel("Spearman Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="PCA Component")
    fig.savefig(
        os.path.join(output_dir, "top_correlations.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot embedding-correlation analysis (SFig 1-3)."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "CSV with columns: gene_target_symbol_name, PCA_1 ... PCA_N, "
            "and numeric meta-feature columns."
        ),
    )
    parser.add_argument(
        "--explained-variance-path",
        default=None,
        help="JSON file with an array of PCA explained-variance ratios.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where figure PNGs are saved (default: current dir).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plt.style.use("ggplot")
    plt.rcParams.update({"axes.facecolor": "whitesmoke"})

    df = pd.read_csv(args.data_path, low_memory=False)
    pca_cols = _detect_pca_cols(df)
    meta_cols = _detect_meta_cols(df, pca_cols)

    if not pca_cols:
        raise ValueError("No PCA_* columns found in the input CSV.")

    explained_variance = None
    if args.explained_variance_path:
        with open(args.explained_variance_path) as fh:
            explained_variance = json.load(fh)

    # Build analysis DataFrame
    analysis_df = df[["gene_target_symbol_name"] + pca_cols + meta_cols].copy()
    per_gene_df, static_cols, dynamic_cols = _split_static_dynamic(
        analysis_df, pca_cols, meta_cols
    )

    print("Plotting PCA scatter ...")
    plot_pca_scatter(df, pca_cols, args.output_dir, explained_variance)

    if explained_variance is not None:
        print("Plotting explained variance ...")
        plot_explained_variance(explained_variance, args.output_dir)

    print("Plotting Spearman heatmap ...")
    plot_spearman_heatmap(
        analysis_df, per_gene_df, pca_cols, static_cols, dynamic_cols, args.output_dir
    )

    print("Plotting MI heatmap ...")
    plot_mi_heatmap(
        analysis_df, per_gene_df, pca_cols, static_cols, dynamic_cols, args.output_dir
    )

    print("Plotting top correlations ...")
    plot_top_correlations(
        analysis_df, per_gene_df, pca_cols, static_cols, dynamic_cols, args.output_dir
    )

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
