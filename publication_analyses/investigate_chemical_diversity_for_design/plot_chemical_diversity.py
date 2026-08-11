#!/usr/bin/env python
"""Plot chemical-modification diversity analysis for a Fig 2 panel.

Produces three figure files:
  1. top25_guide_patterns.png    -- 5x5 grid of heatmaps showing the 25 most
                                    frequent guide-strand chemical patterns.
  2. top25_passenger_patterns.png -- same for the passenger strand.
  3. pattern_similarity.png      -- Pairwise edit-distance similarity matrix
                                    of the top guide-strand chemical patterns.

Input
-----
A NumPy ``.npz`` archive containing pre-computed one-hot tensors:

  * ``X`` -- full one-hot tensor of shape (N, features, seq_len) as produced
    by ``fennec.preprocessing.prepare_meta_features_and_one_hot``.

Generate this file from training data with::

    import numpy as np
    from fennec.preprocessing import prepare_meta_features_and_one_hot
    X, y, X_meta = prepare_meta_features_and_one_hot(train_data)
    np.savez_compressed("one_hot_tensors.npz", X=X.numpy())

Usage
-----
    python plot_chemical_diversity.py \
        --data-path one_hot_tensors.npz \
        --output-dir ./figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
# Column labels for the 21-dimensional one-hot encoding per strand
X_ATTR_COLUMNS = [
    "A-nuc", "C-nuc", "G-nuc", "U-nuc", "T-nuc",
    "ribose_sugar", "deoxyribose_sugar", "glycol_sugar", "hexitol_sugar",
    "no_sugar_mod", "2OMe_mod", "fluro_mod", "4-thioribose_mod",
    "hexadecyl_mod", "LNA_mod", "UNA_mod",
    "N_backbone", "normal_backbone", "phosphorothioate_backbone",
    "VinylP_backbone", "L96_backbone",
]

# Shortened labels for the chemistry-string representation
SIMPLE_CHEM_MAP = {
    "ribose_sugar": "r",
    "deoxyribose_sugar": "d",
    "glycol_sugar": "g",
    "hexitol_sugar": "h",
    "no_sugar_mod": "",
    "2OMe_mod": "OMe",
    "fluro_mod": " F ",
    "4-thioribose_mod": "4td",
    "hexadecyl_mod": "Hd ",
    "LNA_mod": "+  ",
    "UNA_mod": "-  ",
    "N_backbone": "p",
    "normal_backbone": "p",
    "phosphorothioate_backbone": "s",
    "VinylP_backbone": "V",
    "L96_backbone": "L",
}

NUC_END = 5  # first 5 rows are nucleotide identity, rest is chemistry


# ── Pattern counting ─────────────────────────────────────────────────────────
def count_unique_chemical_patterns(
    X_chem: np.ndarray,
) -> tuple[dict, dict, pd.DataFrame]:
    """Count unique chemical-modification patterns across sequences.

    Parameters
    ----------
    X_chem : ndarray of shape (N, chem_dim, seq_len)

    Returns
    -------
    chem_pat : dict mapping flattened pattern tuple -> [count]
    train_chem_keys : dict mapping sequence index -> [pattern tuple]
    chem_pat_df : DataFrame sorted by frequency (index = pattern tuple)
    """
    chem_pat: dict[tuple, list[int]] = {}
    train_chem_keys: dict[int, list[tuple]] = {}

    for i, mat in enumerate(X_chem):
        mat_flat = tuple(mat.flatten())
        if mat_flat in chem_pat:
            chem_pat[mat_flat][0] += 1
        else:
            chem_pat[mat_flat] = [1]
        train_chem_keys[i] = [mat_flat]

    chem_pat_df = pd.DataFrame(chem_pat).T.sort_values(by=0, ascending=False)
    return chem_pat, train_chem_keys, chem_pat_df


# ── Chemistry-string helper ─────────────────────────────────────────────────
def _chem_mat_to_simple_string(chem_mat: pd.DataFrame) -> str:
    """Convert a chemical-pattern matrix into a compact string representation."""
    chem_string = ""
    for position in chem_mat:
        chem_char = ""
        series = chem_mat[position]
        for mod, v in series.items():
            if v == 0:
                continue
            if mod.endswith("_sugar"):
                chem_char += "(" + SIMPLE_CHEM_MAP[mod] + "."
            elif "mod" in mod:
                if not chem_char:
                    chem_char += "("
                chem_char += SIMPLE_CHEM_MAP[mod] + "."
            elif "backbone" in mod:
                chem_char += SIMPLE_CHEM_MAP[mod] + ")-"
        chem_string += chem_char
    return chem_string.rstrip("-")


# ── Plotting functions ───────────────────────────────────────────────────────
def plot_top25_patterns(
    chem_pat_df: pd.DataFrame,
    chem_dim: int,
    seq_len: int,
    row_labels: list[str],
    title: str,
    output_path: str,
) -> dict[str, dict]:
    """5x5 grid of the 25 most frequent chemical patterns."""
    fig, axes = plt.subplots(5, 5, figsize=(30, 20), sharex=True, sharey=True, dpi=300)
    axes_flat = axes.flatten()

    chem_pat_strings: dict[str, dict] = {}

    n_patterns = min(25, len(chem_pat_df))
    for i in range(n_patterns):
        chem_mat = pd.DataFrame(
            np.array(chem_pat_df.index[i]).reshape(chem_dim, seq_len)
        )
        seqs_with_pat = chem_pat_df.iloc[i][0]
        ax = axes_flat[i]
        chem_mat.index = row_labels
        chem_string = _chem_mat_to_simple_string(chem_mat)
        chem_pat_strings[chem_string] = {"N_seqs": seqs_with_pat, "id_mat": i}
        ax.set_title(f"Chemical Pattern N={seqs_with_pat}")
        sns.heatmap(chem_mat, ax=ax)

    # Hide unused axes
    for j in range(n_patterns, 25):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, size=25, verticalalignment="bottom")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return chem_pat_strings


def plot_pattern_similarity(
    chem_pat_strings: dict[str, dict],
    output_path: str,
) -> None:
    """Pairwise edit-distance similarity heatmap of chemical-pattern strings."""
    try:
        from edlib import align
    except ImportError:
        print(
            "WARNING: edlib not installed -- skipping pattern similarity plot. "
            "Install with: pip install edlib"
        )
        return

    chem_string_list = list(chem_pat_strings.keys())
    pairwise: dict[str, dict[str, float]] = {}

    for i, a in enumerate(chem_string_list):
        n_a = chem_pat_strings[a]["N_seqs"]
        row_key = f"{i}  N seqs={n_a}"
        pairwise[row_key] = {}
        for j, b in enumerate(chem_string_list):
            n_b = chem_pat_strings[b]["N_seqs"]
            col_key = f"{j}  N seqs={n_b}"
            slen = max(len(a), len(b))
            edit = align(a, b)["editDistance"]
            pairwise[row_key][col_key] = 1 - edit / slen if slen > 0 else 1.0

    sim_df = pd.DataFrame(pairwise)

    # Sort by total similarity
    col_order = sim_df.sum().sort_values().index
    sim_df = sim_df[col_order]
    row_order = sim_df.sum(axis=1).sort_values().index[::-1]
    sim_df = sim_df.loc[row_order]

    fig = plt.figure(dpi=200, figsize=(10, 8))
    sns.heatmap(sim_df, vmax=1, vmin=0)
    plt.title("Percent Similarity Between Chemical Patterns")
    plt.ylabel("Chemical Sequence Size Order")
    plt.xlabel("Chemical Sequence Size Order")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot chemical-modification diversity (Fig 2 panel)."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to .npz file containing the one-hot tensor 'X' as produced "
            "by fennec.preprocessing.prepare_meta_features_and_one_hot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where figure PNGs are saved (default: current dir).",
    )
    parser.add_argument(
        "--guide-seq-limit",
        type=int,
        default=None,
        help="Truncate guide-strand sequences to this length (default: full).",
    )
    parser.add_argument(
        "--passenger-seq-limit",
        type=int,
        default=25,
        help="Truncate passenger-strand sequences to this length (default: 25).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.style.use("ggplot")

    # Load one-hot tensor
    npz = np.load(args.data_path, allow_pickle=True)
    X = npz["X"]  # shape: (N, features, seq_len)

    # Guide strand: features NUC_END..21 (chemical dims only)
    guide_end = 21
    X_chem_guide = X[:, NUC_END:guide_end, :]
    if args.guide_seq_limit is not None:
        X_chem_guide = X_chem_guide[:, :, : args.guide_seq_limit]

    # Passenger strand: features (21+NUC_END).. (chemical dims only)
    passenger_start = guide_end + NUC_END
    X_chem_pass = X[:, passenger_start:, :]
    if args.passenger_seq_limit is not None:
        X_chem_pass = X_chem_pass[:, :, : args.passenger_seq_limit]

    _, chem_dim_g, seq_len_g = X_chem_guide.shape
    _, chem_dim_p, seq_len_p = X_chem_pass.shape

    guide_labels = X_ATTR_COLUMNS[NUC_END : NUC_END + chem_dim_g]
    passenger_labels = X_ATTR_COLUMNS[NUC_END : NUC_END + chem_dim_p]

    # Count unique patterns
    print("Counting guide-strand chemical patterns ...")
    _, _, guide_pat_df = count_unique_chemical_patterns(X_chem_guide)

    print("Counting passenger-strand chemical patterns ...")
    _, _, pass_pat_df = count_unique_chemical_patterns(X_chem_pass)

    # Plot top 25 guide patterns
    print("Plotting top 25 guide-strand patterns ...")
    guide_strings = plot_top25_patterns(
        guide_pat_df,
        chem_dim_g,
        seq_len_g,
        guide_labels,
        "Top 25 Chemical Patterns (Guide Strand)",
        os.path.join(args.output_dir, "top25_guide_patterns.png"),
    )

    # Plot top 25 passenger patterns
    print("Plotting top 25 passenger-strand patterns ...")
    plot_top25_patterns(
        pass_pat_df,
        chem_dim_p,
        seq_len_p,
        passenger_labels,
        f"Top 25 Chemical Patterns (Passenger Strand, limit {seq_len_p} bp)",
        os.path.join(args.output_dir, "top25_passenger_patterns.png"),
    )

    # Plot similarity matrix for guide patterns
    if guide_strings:
        print("Plotting pattern similarity matrix ...")
        plot_pattern_similarity(
            guide_strings,
            os.path.join(args.output_dir, "pattern_similarity.png"),
        )

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
