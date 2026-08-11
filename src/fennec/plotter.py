import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.metrics import roc_auc_score, r2_score, f1_score, average_precision_score
from scipy.stats import spearmanr

import click


def to_numeric_1d_array(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([], dtype=float)
    arr = arr.reshape(-1)
    arr = pd.to_numeric(arr, errors='coerce')
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def compute_auc_from_regression_targets(y_true, y_pred, response_threshold=30.0):
    y_true_arr = to_numeric_1d_array(y_true)
    y_pred_arr = to_numeric_1d_array(y_pred)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return float('nan')
    n = min(y_true_arr.size, y_pred_arr.size)
    y_true_arr = y_true_arr[:n]
    y_pred_arr = y_pred_arr[:n]

    y_true_bin = (y_true_arr <= response_threshold).astype(int)
    if np.unique(y_true_bin).size < 2:
        return float('nan')

    y_score = -y_pred_arr
    return round(float(roc_auc_score(y_true_bin, y_score)), 4)

def calc_key_stats(y_test, y_pred, response_threshold=30.0):
    """
    Calculate Spearman correlation, AUC for top half, and R2 score between true and predicted values.

    """

    y_test_arr = to_numeric_1d_array(y_test)
    y_pred_arr = to_numeric_1d_array(y_pred)
    if y_test_arr.size == 0 or y_pred_arr.size == 0:
        return float('nan'), float('nan'), float('nan'), []
    n = min(y_test_arr.size, y_pred_arr.size)
    y_test_arr = y_test_arr[:n]
    y_pred_arr = y_pred_arr[:n]

    y_test_class = (y_test_arr <= response_threshold).astype(int).tolist()
    auc = compute_auc_from_regression_targets(y_test_arr, y_pred_arr, response_threshold=response_threshold)
    res = spearmanr(y_test_arr, y_pred_arr)
    spear = round(res.correlation, 4)
    r2 = r2_score(y_test_arr, y_pred_arr)
    r2 = round(r2, 4)

    return spear, auc, r2, y_test_class


def calc_key_stats_frac(y_test, y_pred, frac=0.25):
    """
    Calculate Spearman correlation, AUC for top half, and R2 score between true and predicted values.
    There are multiple frac to define the in group for istance best (lowest) 50% (frac=0.5) or best(lowest) 25% (frac=0.25).
    """

    assert 0 < frac < 1, "Fraction must be between 0 and 1."

    threshold_test = np.percentile(y_test, 100 * (frac))
    threshold_pred = np.percentile(y_pred, 100 * (frac))
    y_test_class = [1 if i <= threshold_test else 0 for i in y_test]
    y_pred_class = [1 if i <= threshold_pred else 0 for i in y_pred]
    # For rank-based metrics use continuous prediction scores, not thresholded labels.
    auc = compute_auc_from_regression_targets(y_test, y_pred, response_threshold=threshold_test)
    f1 = round(f1_score(y_test_class, y_pred_class), 4)
    pr_auc = round(average_precision_score(y_test_class, -np.asarray(y_pred).reshape(-1)), 4)
    res = spearmanr(y_test, y_pred)
    spear = round(res.correlation, 4)
    r2 = r2_score(y_test, y_pred)
    r2 = round(r2, 4)

    return spear, auc, r2, f1, pr_auc, y_test_class, y_pred_class

def calc_key_stats_weight_split_by_gene(y_test_inp, y_pred_inp, genes, rounder=4, frac=0.25):

    assert 0 < frac < 1, "Fraction must be between 0 and 1."

    unique_genes = np.unique(genes)
    f1s_all = 0
    spear_all = 0
    r2_all = 0
    mae = 0

    all_len = len(y_test_inp)
    
    for ugene in unique_genes:
        genes_mask = genes == ugene
        y_test = y_test_inp[genes_mask]
        y_pred = y_pred_inp[genes_mask]

        weight = len(y_test) / all_len
        threshold_test = np.percentile(y_test, 100 * frac)
        threshold_pred = np.percentile(y_pred, 100 * frac)

        y_test_class = [1 if i <= threshold_test else 0 for i in y_test]
        y_pred_class = [1 if i <= threshold_pred else 0 for i in y_pred]
        
        f1s = f1_score(y_test_class, y_pred_class)
        
        res = spearmanr(y_test, y_pred)
        spear = res.correlation
        
        
        r2 = r2_score(y_test, y_pred)
        r2 = r2

        f1s_all += f1s * weight
        spear_all += spear * weight
        r2_all += r2 * weight

    return spear_all, f1s_all, r2_all

def plot_cv_loss(all_stats2_df: pd.DataFrame, name: str):

    # Number of unique CVs
    n_folds = all_stats2_df['cv'].nunique()

    # Determine number of rows and columns dynamically
    ncols = min(5, n_folds)  # Cap the number of columns to keep layout readable
    nrows = math.ceil(n_folds / ncols)

    # Adjust figure size proportionally
    fig_width = 5 * ncols
    fig_height = 4 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)

    # Flatten the axes array (even if it's 1D or 2D)
    axes = np.array(axes).reshape(-1)

    for i, cv in enumerate(sorted(all_stats2_df['cv'].unique())):
        tdf = all_stats2_df[all_stats2_df['cv'] == cv]
        ax = axes[i]

        sns.lineplot(data=tdf, x='epoch', y='test_mse', label='Test MSE', ax=ax)
        sns.lineplot(data=tdf, x='epoch', y='train_loss', label='Train Loss', ax=ax)

        ax.set_xlabel('Epoch #')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'CV Fold: {cv}')
        ax.legend()

    # Hide any unused axes
    for j in range(n_folds, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{name}_all_cv_losses.png')

def plot_density(best_statse, name, auc_response_threshold=30.0):
    if best_statse.empty:
        plt.figure(dpi=150)
        plt.text(0.05, 0.5, 'No per-epoch test predictions available for density plot.')
        plt.axis('off')
        plt.savefig(f'{name}_density_stats.png')
        return

    spear, auc, r2, _ = calc_key_stats(best_statse['test_y'].values,
                                       best_statse['test_pred'].values,
                                       response_threshold=auc_response_threshold)

    plt.figure(dpi=150)
    sns.kdeplot(data=best_statse, x='test_pred', y='test_y', levels=50, thresh=0.1, fill=True, palette='Spectral', alpha=0.3)
    #sns.scatterplot(x=y_test, y=y_pred, hue=X_test_df.gene_target_symbol_name.fillna('Unknown').values, palette='Set1', alpha=0.05)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.text(0, 0, f'Spearman={spear} \nAUC@<= {auc_response_threshold}={auc}\nR2={r2}\nN={len(best_statse)}')
    plt.plot([0, 150], [0, 150], alpha=0.3, linestyle='--')
    plt.xlim(-10, 110)
    plt.savefig(f'{name}_density_stats.png')

@click.group()
def cli():
    """fennec CLI for plotting CV split results."""
    pass

@cli.command()
@click.argument('results_pkl_paths', nargs=-1, type=click.Path())
@click.option('--auc-response-threshold', type=float, default=30.0, help='Fixed mRNA_remaining_pct threshold to define positive class for ROC AUC (positive means <= threshold).')
def plot(results_pkl_paths: list, auc_response_threshold: float):
    all_dfs = []
    for pkl in results_pkl_paths:
        tdf = pd.read_pickle(pkl).reset_index().rename(columns={'index': 'epoch'})
        all_dfs.append(tdf)

    name = results_pkl_paths[0].split('_stat')[0]
    all_stats2_df = pd.concat(all_dfs)

    # Publication mode: never select rows based on outer-test performance.
    # We select the latest epoch per fold, preferring rows that carry
    # prediction payloads for density plotting.
    all_stats3_df = all_stats2_df.copy()

    def has_prediction_payload(row):
        y_arr = to_numeric_1d_array(row['test_y']) if 'test_y' in row else np.array([], dtype=float)
        p_arr = to_numeric_1d_array(row['test_pred']) if 'test_pred' in row else np.array([], dtype=float)
        return min(y_arr.size, p_arr.size) > 0

    all_stats3_df['has_preds'] = all_stats3_df.apply(has_prediction_payload, axis=1)
    candidate_rows = all_stats3_df[all_stats3_df['has_preds']].copy()
    if candidate_rows.empty:
        candidate_rows = all_stats3_df.copy()

    best_stats = (
        candidate_rows
        .sort_values(by=['has_preds', 'epoch'], ascending=[False, False])
        .groupby('cv')
        .head(1)
        .sort_values(by='cv')
    )
    best_records = []
    for _, row in best_stats.iterrows():
        y_arr = to_numeric_1d_array(row['test_y']) if 'test_y' in row else np.array([], dtype=float)
        p_arr = to_numeric_1d_array(row['test_pred']) if 'test_pred' in row else np.array([], dtype=float)
        n = min(y_arr.size, p_arr.size)
        for i in range(n):
            best_records.append({
                'cv': row.get('cv', np.nan),
                'epoch': row.get('epoch', np.nan),
                'test_y': y_arr[i],
                'test_pred': p_arr[i],
                'Name': name,
            })

    if best_records:
        best_statse = pd.DataFrame(best_records)
    else:
        best_statse = pd.DataFrame(columns=['cv', 'epoch', 'test_y', 'test_pred', 'Name'])

    best_statse.to_csv(f'{name}_best_stats.csv')

    plot_cv_loss(all_stats2_df, name)
    plot_density(best_statse, name, auc_response_threshold=auc_response_threshold)

if __name__ == '__main__':
    cli()
