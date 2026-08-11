#!/usr/bin/env python3
"""
Plot Figure 3 -- FENNEC model performance, ablation study, and ensembling gain.

Reads pre-computed metric CSVs (one row per model-variant x seed or
model-variant x ensemble) and produces publication-quality bar-chart panels.

Expected CSV schemas
--------------------
seed_metrics.csv
    model_name, seed, Epoch, variable, value
    where *variable* is one of: Spearman, ROC AUC, MAE, R2,
    Spearman All, ROC AUC All, R2 All

ensemble_metrics.csv   (optional)
    model_name, variable, value
    same metric names as above

Usage
-----
    python plot_ablation_fig3.py \\
        --seed-metrics seed_metrics.csv \\
        --ensemble-metrics ensemble_metrics.csv \\
        --output fig3.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Metric display ranges (y-axis limits) used in the publication figure.
# Keys must match the *variable* column in the input CSVs.
# ---------------------------------------------------------------------------
METRIC_RANGES = {
    "Spearman": (0.14, 0.55),
    "ROC AUC": (0.49, 0.80),
    "MAE": (20, 45),
    "R2": (-0.20, 0.40),
    "Spearman All": (0.14, 0.55),
    "ROC AUC All": (0.49, 0.80),
    "R2 All": (-0.20, 0.40),
}

# Metrics shown in the main figure panels (4 primary metrics).
DEFAULT_METRICS = ["Spearman", "ROC AUC", "MAE", "R2"]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    y_range: tuple[float, float] | None = None,
    palette: str = "Spectral",
    errorbar: tuple[str, float] = ("se", 1.96),
    label_fmt: str = "%.3f",
    label_size: int = 7,
) -> None:
    """Draw a single bar-chart panel for one metric.

    Parameters
    ----------
    df : DataFrame
        Must contain columns *model_name*, *variable*, and *value*.
    """
    df_metric = df[df["variable"] == metric].copy()
    if df_metric.empty:
        ax.set_title(metric)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return

    # Sort models by ascending mean so the best model appears on the right.
    group_means = df_metric.groupby("model_name")["value"].mean().sort_values()
    df_metric["group_mean"] = df_metric["model_name"].map(group_means)
    df_metric = df_metric.sort_values("group_mean")

    sns.barplot(
        data=df_metric,
        x="model_name",
        y="value",
        hue="model_name",
        palette=palette,
        alpha=0.7,
        errorbar=errorbar,
        ax=ax,
    )

    # Annotate bars with their mean value.
    for container in ax.containers:
        ax.bar_label(
            container, fmt=label_fmt, label_type="center",
            padding=0, size=label_size, rotation=90,
        )

    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45)

    if y_range is not None:
        ax.set_ylim(*y_range)
        y_ticks = np.linspace(y_range[0], y_range[1], 9)
        y_minor = np.linspace(y_range[0], y_range[1], 17)
        ax.set_yticks(y_ticks)
        ax.set_yticks(y_minor, minor=True)
        ax.tick_params(axis="y", which="major", length=4, width=0.9)
        ax.tick_params(axis="y", which="minor", length=2, width=0.5)
        for yt in y_ticks:
            ax.axhline(yt, color="grey", linewidth=0.3, alpha=0.5)

    # Hide the auto-generated legend (model names are on x-axis).
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def plot_metric_panels(
    df: pd.DataFrame,
    metrics: list[str],
    suptitle: str = "",
    dpi: int = 200,
    palette: str = "Spectral",
    errorbar: tuple[str, float] = ("se", 1.96),
) -> plt.Figure:
    """Create a row of bar-chart panels, one per metric."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        y_range = METRIC_RANGES.get(metric)
        _bar_panel(ax, df, metric, y_range=y_range, palette=palette,
                   errorbar=errorbar)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot FENNEC Figure 3 (ablation / model performance).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--seed-metrics", required=True, type=Path,
        help="CSV with per-seed metrics (model_name, seed, Epoch, variable, value).",
    )
    p.add_argument(
        "--ensemble-metrics", default=None, type=Path,
        help="CSV with ensemble metrics (model_name, variable, value). "
             "If omitted, only the per-seed panel is plotted.",
    )
    p.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help="Which metrics to plot (default: %(default)s).",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=Path("fig3.pdf"),
        help="Output figure path (default: fig3.pdf).",
    )
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    p.add_argument(
        "--style", default="ggplot",
        help="Matplotlib style (default: ggplot).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use(args.style)

    # -- Load data ----------------------------------------------------------
    seed_df = pd.read_csv(args.seed_metrics)
    seed_df["value"] = pd.to_numeric(seed_df["value"], errors="coerce")

    ensemble_df = None
    if args.ensemble_metrics is not None:
        ensemble_df = pd.read_csv(args.ensemble_metrics)
        ensemble_df["value"] = pd.to_numeric(ensemble_df["value"], errors="coerce")

    # -- Plot ---------------------------------------------------------------
    figures: list[tuple[str, plt.Figure]] = []

    fig_seed = plot_metric_panels(
        seed_df, args.metrics,
        suptitle="Per-seed model performance",
        dpi=args.dpi,
    )
    figures.append(("seed", fig_seed))

    if ensemble_df is not None:
        fig_ens = plot_metric_panels(
            ensemble_df, args.metrics,
            suptitle="Ensemble model performance",
            dpi=args.dpi,
        )
        figures.append(("ensemble", fig_ens))

    # -- Save ---------------------------------------------------------------
    out = args.output
    if len(figures) == 1:
        figures[0][1].savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        # When we have multiple figure groups, save each with a suffix.
        for label, fig in figures:
            stem = out.stem
            suffix = out.suffix or ".pdf"
            path = out.with_name(f"{stem}_{label}{suffix}")
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved {path}")

    plt.close("all")


if __name__ == "__main__":
    main()
