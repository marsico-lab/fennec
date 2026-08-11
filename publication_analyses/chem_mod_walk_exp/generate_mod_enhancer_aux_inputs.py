#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def canonicalize_sirna(value: object) -> str:
    text = "" if value is None else str(value)
    return text.strip().upper().replace("T", "U")


def model_key_from_stats_file(path: Path) -> str:
    stem = path.name.replace("_best_stats.csv", "")
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def token_edit_distance(seq_a: str, seq_b: str) -> int:
    a = str(seq_a).split()
    b = str(seq_b).split()
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return int(dp[n])


def load_test_index_map(stats_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    npy_paths = sorted(stats_dir.glob("test_index_*.npy"))
    name_re = re.compile(r"test_index_(.+)_(\d+)\.npy$")
    by_model: dict[str, dict[int, np.ndarray]] = {}

    for path in npy_paths:
        match = name_re.match(path.name)
        if match is None:
            continue
        model_key, fold = match.groups()
        by_model.setdefault(model_key, {})[int(fold)] = np.load(path)
    return by_model


def build_test_df_from_model_key(
    model_key: str,
    train_data: pd.DataFrame,
    test_indices_by_model: dict[str, dict[int, np.ndarray]],
) -> pd.DataFrame:
    fold_map = test_indices_by_model.get(model_key)
    if not fold_map:
        raise KeyError(f"No test_index npy files found for model_key={model_key}")

    fold_ids = sorted(fold_map.keys())
    test_order = np.concatenate([fold_map[k] for k in fold_ids])
    cv_ids = np.concatenate([np.full(len(fold_map[k]), k, dtype=int) for k in fold_ids])

    out = train_data.iloc[test_order].copy()
    out["train_data_index"] = out.index
    out["CV"] = cv_ids
    out["test_y"] = out["mRNA_remaining_pct"]
    return out


def build_ablation_table(
    stats_dir: Path,
    train_data: pd.DataFrame,
) -> pd.DataFrame:
    test_indices_by_model = load_test_index_map(stats_dir)
    stats_files = sorted(stats_dir.glob("*stats.csv"))
    rows: list[pd.DataFrame] = []
    test_df_cache: dict[str, pd.DataFrame] = {}

    for stats_file in stats_files:
        try:
            stats_df = pd.read_csv(stats_file, index_col=0)
        except Exception:
            continue

        if "Name" not in stats_df.columns:
            continue

        model_key = model_key_from_stats_file(stats_file)
        if model_key not in test_df_cache:
            try:
                test_df_cache[model_key] = build_test_df_from_model_key(
                    model_key=model_key,
                    train_data=train_data,
                    test_indices_by_model=test_indices_by_model,
                )
            except Exception:
                continue

        test_df = test_df_cache[model_key]
        if len(stats_df) != len(test_df):
            continue

        stats_df = stats_df.copy()
        stats_df["seed"] = stats_df["Name"].map(lambda x: int(str(x).split("_")[-1]))
        stats_df["model_name"] = stats_df["Name"].map(lambda x: "_".join(str(x).split("_")[:-1]))

        merged = pd.concat(
            [stats_df.reset_index(drop=False), test_df.drop(columns=["test_y"]).reset_index(drop=True)],
            axis=1,
        )
        rows.append(merged)

    if not rows:
        raise RuntimeError(f"No usable *stats.csv rows found in {stats_dir}")
    return pd.concat(rows, ignore_index=True)


def build_condition_metrics(ablat_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "model_name",
        "index",
        "train_data_index",
        "test_y",
        "test_pred",
        "siRNA",
        "cell_line_donor",
        "siRNA_concentration",
        "gene_target_species",
        "Transfection_method",
        "gene_target_symbol_name",
        "siRNA_list",
        "siRNA_passenger_list",
    }
    missing = sorted(list(required_cols - set(ablat_df.columns)))
    if missing:
        raise ValueError(f"ablation table missing required columns: {missing}")

    group_cols = ["model_name", "index", "train_data_index"]
    ensemble = ablat_df.groupby(group_cols)[["test_y", "test_pred"]].mean().reset_index()

    metadata_cols = [
        c
        for c in [
            "cv",
            "gene_target_symbol_name",
            "patent_id",
            "siRNA",
            "cell_line_donor",
            "siRNA_concentration",
            "gene_target_species",
            "Transfection_method",
            "table_title",
            "Duplex_ID",
            "siRNA_list",
            "siRNA_passenger_list",
            "meta_group",
        ]
        if c in ablat_df.columns
    ]
    metadata_df = (
        ablat_df[["model_name", "train_data_index", *metadata_cols]]
        .drop_duplicates(["model_name", "train_data_index"])
        .copy()
    )

    ensemble = ensemble.merge(
        metadata_df,
        on=["model_name", "train_data_index"],
        how="left",
        validate="many_to_one",
    )
    ensemble["Gene"] = ensemble["gene_target_symbol_name"]
    ensemble["siRNA"] = ensemble["siRNA"].map(canonicalize_sirna)

    condition_rows: list[dict[str, object]] = []
    condition_group_cols = [
        "model_name",
        "siRNA",
        "cell_line_donor",
        "siRNA_concentration",
        "gene_target_species",
        "Transfection_method",
        "gene_target_symbol_name",
        "table_title",
    ]

    for condition, group_df in ensemble.groupby(condition_group_cols):
        dedup = group_df.drop_duplicates(subset=["siRNA_list", "siRNA_passenger_list"]).copy()
        if len(dedup) < 2:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(dedup["test_y"], dedup["test_pred"])

        condition_rows.append(
            {
                "model_name": condition[0],
                "siRNA": condition[1],
                "cell_line_donor": condition[2],
                "siRNA_concentration": condition[3],
                "gene_target_species": condition[4],
                "Transfection_method": condition[5],
                "Gene": condition[6],
                "table_title": condition[7],
                "true_y_stdev": dedup["test_y"].std(),
                "n": len(dedup),
                "n_full": len(group_df),
                "spearman": rho,
                "p_value": p_value,
            }
        )

    condition_df = pd.DataFrame(condition_rows)
    condition_df = condition_df.sort_values(by=["n", "spearman"], ascending=[False, False]).reset_index(drop=True)
    return condition_df


def build_distance_json(
    train_data: pd.DataFrame,
    condition_df: pd.DataFrame,
    min_n: int,
) -> dict[str, dict[str, object]]:
    candidate_sirnas = condition_df.loc[condition_df["n"] > min_n, "siRNA"].dropna().unique()
    train = train_data.copy()
    train["siRNA"] = train["siRNA"].map(canonicalize_sirna)
    train = train[train["siRNA"].isin(candidate_sirnas)].copy()
    train = train.dropna(subset=["siRNA_list", "siRNA_passenger_list"])
    train["guide_and_passenger_list"] = train["siRNA_list"].astype(str) + " " + train["siRNA_passenger_list"].astype(str)

    all_matrices: dict[str, dict[str, object]] = {}
    for sirna in train["siRNA"].dropna().unique():
        sub = train[train["siRNA"] == sirna]
        unique_lists = pd.Series(sub["guide_and_passenger_list"].unique()).dropna().astype(str).tolist()
        n = len(unique_lists)
        if n == 0:
            continue

        dist = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                value = token_edit_distance(unique_lists[i], unique_lists[j])
                dist[i, j] = value
                dist[j, i] = value

        all_matrices[sirna] = {
            "dist": dist.tolist(),
            "list_full": unique_lists,
        }
    return all_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Mod Enhancer aux inputs from ablation artifacts")
    parser.add_argument(
        "--stats-dir",
        required=True,
        help="Directory containing *stats.csv and test_index_*.npy files",
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Train data csv.gz used for test-index joins",
    )
    parser.add_argument(
        "--output-condition-metrics",
        default="ablation/vizzes/ensemble_chemicially_numerous_siRNA_condition_spearman.csv",
        help="Output CSV path for condition metrics",
    )
    parser.add_argument(
        "--output-distance-json",
        default="ablation/vizzes/siRNA_guide_passenger_token_edit_distance_matrices.json",
        help="Output JSON path for per-siRNA token-distance matrices",
    )
    parser.add_argument(
        "--min-condition-n",
        type=int,
        default=4,
        help="Use siRNAs with condition-metric n > this value when building distance JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    train_data_path = Path(args.train_data)
    output_condition_metrics = Path(args.output_condition_metrics)
    output_distance_json = Path(args.output_distance_json)

    train_data = pd.read_csv(train_data_path, compression="gzip")
    train_data["siRNA"] = train_data["siRNA"].map(canonicalize_sirna)

    ablat_df = build_ablation_table(stats_dir=stats_dir, train_data=train_data)
    condition_df = build_condition_metrics(ablat_df)

    output_condition_metrics.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(output_condition_metrics, index=False)

    distance_payload = build_distance_json(
        train_data=train_data,
        condition_df=condition_df,
        min_n=int(args.min_condition_n),
    )
    output_distance_json.parent.mkdir(parents=True, exist_ok=True)
    with output_distance_json.open("w", encoding="utf-8") as handle:
        json.dump(distance_payload, handle, indent=2)

    print({
        "stats_dir": str(stats_dir),
        "ablation_rows": int(len(ablat_df)),
        "condition_rows": int(len(condition_df)),
        "distance_sirnas": int(len(distance_payload)),
        "output_condition_metrics": str(output_condition_metrics),
        "output_distance_json": str(output_distance_json),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())