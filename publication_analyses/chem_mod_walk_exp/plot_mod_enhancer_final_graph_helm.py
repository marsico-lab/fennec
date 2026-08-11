#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import roc_auc_score
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from chem_mod_walk_exp.mod_enhancer_closed_world_clean_helm import (
    BeamSearchConfig,
    ClosedWorldModEnhancerSearch,
    HELM_COLUMN,
    describe_helm_position,
    ensure_helm_sequence_columns,
    guide_helm_tokens,
    helm_tokens_to_fennec_pair,
    load_train_data,
    make_pair_key,
    passenger_helm_tokens,
    render_helm_position_from_template,
    resolve_model_paths,
    DEFAULT_DISTANCE_JSON,
    load_distance_payload,
)


DESIGN_STATUS_ORDER = [
    "Unoptimized Least Effective siRNA Chemical Design",
    "Redesigned Match Recovered",
    "Experimental siRNA Match missing from AdaLead designs",
]

DESIGN_STATUS_PALETTE = {
    "Unoptimized Least Effective siRNA Chemical Design": "#4C78A8",
    "Redesigned Match Recovered": "#54A24B",
    "Experimental siRNA Match missing from AdaLead designs": "#E45756",
}


def _format_compact_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    value_int = int(value)
    abs_value = abs(value_int)
    if abs_value >= 1_000_000_000:
        return f"{value_int / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value_int / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value_int / 1_000:.1f}K"
    return str(value_int)


def _abbreviate_sequence(seq: str, head: int = 9, tail: int = 9) -> str:
    token = str(seq)
    if len(token) <= head + tail + 3:
        return token
    return f"{token[:head]}...{token[-tail:]}"


def _format_percent(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _format_multiplier_x(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    value_float = float(value)
    if np.isposinf(value_float):
        return "infx"
    if np.isneginf(value_float):
        return "-infx"
    return f"{value_float:.2f}x"


def _format_recovery_fraction(value: float | int | None, observed_count: float | int | None) -> str:
    if observed_count is None or (isinstance(observed_count, float) and np.isnan(observed_count)):
        return _format_percent(value)
    if float(observed_count) <= 0:
        return "none reachable"
    return _format_percent(value)


def _token_distance(
    seed_guide: list[str],
    seed_passenger: list[str],
    guide_tokens: list[str],
    passenger_tokens: list[str],
) -> int | None:
    if len(seed_guide) != len(guide_tokens) or len(seed_passenger) != len(passenger_tokens):
        return None
    distance = 0
    for left, right in zip(seed_guide, guide_tokens):
        if left != right:
            distance += 1
    for left, right in zip(seed_passenger, passenger_tokens):
        if left != right:
            distance += 1
    return distance


def _mutation_operation_distance(
    seed_guide: list[str],
    seed_passenger: list[str],
    guide_tokens: list[str],
    passenger_tokens: list[str],
) -> int | None:
    return _token_distance(seed_guide, seed_passenger, guide_tokens, passenger_tokens)


def _alternative_counts_for_mode(
    seed_guide_tokens: list[str],
    seed_passenger_tokens: list[str],
    token_choices: dict[str, list[list[str]]],
) -> list[int]:
    guide_alternatives = []
    for position, choices in enumerate(token_choices["guide"]):
        current_token = seed_guide_tokens[position]
        guide_alternatives.append(sum(token != current_token for token in choices))

    passenger_alternatives = []
    for position, choices in enumerate(token_choices["passenger"]):
        current_token = seed_passenger_tokens[position]
        passenger_alternatives.append(sum(token != current_token for token in choices))

    return guide_alternatives + passenger_alternatives


def _token_space_total_est_for_mode(
    alternative_counts: list[int],
    token_choices: dict[str, list[list[str]]],
) -> int:
    per_position_choice_counts = [
        max(len(position_choices), 1) for position_choices in token_choices["guide"]
    ] + [
        max(len(position_choices), 1) for position_choices in token_choices["passenger"]
    ]
    return int(math.prod(per_position_choice_counts)) if per_position_choice_counts else 0


def _is_token_space_compatible(
    guide_tokens: list[str],
    passenger_tokens: list[str],
    token_choices: dict[str, list[list[str]]],
) -> bool:
    if len(guide_tokens) != len(token_choices["guide"]):
        return False
    if len(passenger_tokens) != len(token_choices["passenger"]):
        return False

    for position, token in enumerate(guide_tokens):
        if token not in token_choices["guide"][position]:
            return False
    for position, token in enumerate(passenger_tokens):
        if token not in token_choices["passenger"][position]:
            return False
    return True


def _normalize_tokens_to_seed_bases(tokens: list[str], seed_tokens: list[str]) -> list[str] | None:
    if len(tokens) != len(seed_tokens):
        return None

    normalized_tokens: list[str] = []
    for token, seed_token in zip(tokens, seed_tokens):
        token_descriptor = describe_helm_position(token)
        seed_descriptor = describe_helm_position(seed_token)

        if seed_descriptor.category == "nucleotide":
            anchor_descriptor = seed_descriptor.anchor_descriptor
            if token_descriptor.category != "nucleotide" or anchor_descriptor is None or anchor_descriptor.base is None:
                return None
            remapped = render_helm_position_from_template(token, anchor_descriptor.base)
            if remapped is None:
                return None
            normalized_tokens.append(remapped)
            continue

        if seed_descriptor.category == "prefix_attachment":
            if token_descriptor.category != "prefix_attachment":
                return None
            normalized_tokens.append(str(token).strip())
            continue

        if seed_descriptor.category == "abasic":
            if token_descriptor.category != "abasic":
                return None
            normalized_tokens.append(str(token).strip())
            continue

        return None

    return normalized_tokens


def _reachable_count_within_steps(alternative_counts: list[int], max_steps: int) -> int:
    if max_steps < 0:
        return 0
    coefficients = [1]
    for alternatives in alternative_counts:
        next_coefficients = [0] * (len(coefficients) + 1)
        for degree, value in enumerate(coefficients):
            next_coefficients[degree] += value
            next_coefficients[degree + 1] += value * max(alternatives, 0)
        coefficients = next_coefficients
    capped_steps = min(max_steps, len(alternative_counts))
    return int(sum(coefficients[: capped_steps + 1]))


def build_origin_search_metrics(
    search_results: pd.DataFrame,
    search: ClosedWorldModEnhancerSearch,
) -> pd.DataFrame:
    search_results = ensure_helm_sequence_columns(search_results)
    rows: list[dict[str, float | int]] = []

    for origin_id, group in search_results.groupby("origin_id", dropna=False):
        origin_int = int(origin_id)
        if origin_int not in search.token_choices_by_origin or origin_int not in search.seed_pair_by_origin:
            continue
        max_search_iteration = int(group["search_iteration"].max())
        if "effective_beam_width" in group.columns and group["effective_beam_width"].notna().any():
            effective_beam_width = int(group["effective_beam_width"].dropna().iloc[0])
        else:
            effective_beam_width = int(
                search.effective_beam_width_by_origin.get(
                    origin_int,
                    search.beam_config.beam_width,
                )
            )

        if {"siRNA_list", "siRNA_passenger_list"}.issubset(group.columns):
            pair_key_series = group.apply(
                lambda row: "||".join(
                    make_pair_key(row["siRNA_list"], row["siRNA_passenger_list"])
                ),
                axis=1,
            )
        else:
            pair_key_series = group[HELM_COLUMN].astype(str)
        dedup_group = group.assign(_pair_key=pair_key_series).drop_duplicates(subset=["_pair_key"]).copy()

        token_choices = search.token_choices_by_origin[origin_int]
        seed_guide, seed_passenger = search.seed_pair_by_origin[origin_int]
        seed_guide_tokens = guide_helm_tokens(seed_guide, seed_passenger)
        seed_passenger_tokens = passenger_helm_tokens(seed_guide, seed_passenger)

        alternative_counts = _alternative_counts_for_mode(
            seed_guide_tokens,
            seed_passenger_tokens,
            token_choices,
        )
        reachable_space_est = _reachable_count_within_steps(alternative_counts, max_search_iteration)
        token_space_total_est = _token_space_total_est_for_mode(
            alternative_counts,
            token_choices,
        )

        observed_reachable_non_seed = 0
        observed_pool_unique = 0
        for guide_str, passenger_str in search.observed_pairs_by_origin.get(origin_int, set()):
            canonical_guide, canonical_passenger = make_pair_key(guide_str, passenger_str)
            guide_tokens = guide_helm_tokens(canonical_guide, canonical_passenger)
            passenger_tokens = passenger_helm_tokens(canonical_guide, canonical_passenger)
            normalized_guide_tokens = _normalize_tokens_to_seed_bases(guide_tokens, seed_guide_tokens)
            normalized_passenger_tokens = _normalize_tokens_to_seed_bases(passenger_tokens, seed_passenger_tokens)
            if normalized_guide_tokens is None or normalized_passenger_tokens is None:
                continue
            if not _is_token_space_compatible(guide_tokens, passenger_tokens, token_choices):
                if not _is_token_space_compatible(normalized_guide_tokens, normalized_passenger_tokens, token_choices):
                    continue
                guide_tokens = normalized_guide_tokens
                passenger_tokens = normalized_passenger_tokens
            else:
                guide_tokens = normalized_guide_tokens
                passenger_tokens = normalized_passenger_tokens
            distance = _mutation_operation_distance(
                seed_guide_tokens,
                seed_passenger_tokens,
                guide_tokens,
                passenger_tokens,
            )
            if distance is None:
                continue
            if distance <= max_search_iteration:
                observed_pool_unique += 1
                if distance > 0:
                    observed_reachable_non_seed += 1

        explored_unique_in_reachable = 0
        novel_explored_unique = 0
        recovered_unique = 0
        for row in dedup_group.itertuples(index=False):
            canonical_guide, canonical_passenger = make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
            guide_tokens = guide_helm_tokens(canonical_guide, canonical_passenger)
            passenger_tokens = passenger_helm_tokens(canonical_guide, canonical_passenger)
            normalized_guide_tokens = _normalize_tokens_to_seed_bases(guide_tokens, seed_guide_tokens)
            normalized_passenger_tokens = _normalize_tokens_to_seed_bases(passenger_tokens, seed_passenger_tokens)
            if normalized_guide_tokens is None or normalized_passenger_tokens is None:
                continue
            if not _is_token_space_compatible(guide_tokens, passenger_tokens, token_choices):
                if not _is_token_space_compatible(normalized_guide_tokens, normalized_passenger_tokens, token_choices):
                    continue
                guide_tokens = normalized_guide_tokens
                passenger_tokens = normalized_passenger_tokens
            else:
                guide_tokens = normalized_guide_tokens
                passenger_tokens = normalized_passenger_tokens
            distance = _mutation_operation_distance(
                seed_guide_tokens,
                seed_passenger_tokens,
                guide_tokens,
                passenger_tokens,
            )
            if distance is None or distance > max_search_iteration:
                continue
            explored_unique_in_reachable += 1
            if row.match_status == "Novel":
                novel_explored_unique += 1
            elif row.match_status == "RecoveredObserved":
                recovered_unique += 1

        novel_reachable_est = max(reachable_space_est - 1 - observed_reachable_non_seed, 0)
        novel_explored_fraction = (
            float(novel_explored_unique / novel_reachable_est)
            if novel_reachable_est > 0
            else np.nan
        )
        recovered_fraction_observed = (
            float(recovered_unique / observed_reachable_non_seed)
            if observed_reachable_non_seed > 0
            else np.nan
        )
        explored_fraction_reachable = (
            float(explored_unique_in_reachable / reachable_space_est)
            if reachable_space_est > 0
            else np.nan
        )
        explored_fraction_total_space = (
            float(explored_unique_in_reachable / token_space_total_est)
            if token_space_total_est > 0
            else np.nan
        )

        # Random baseline over the iteration-limited reachable neighborhood (excluding seed).
        reachable_non_seed = max(reachable_space_est - 1, 0)
        random_hit_observed_prob = (
            float(observed_reachable_non_seed / reachable_non_seed)
            if reachable_non_seed > 0
            else np.nan
        )
        explored_non_seed_unique = min(max(explored_unique_in_reachable - 1, 0), reachable_non_seed)
        expected_random_recovered = (
            float(explored_non_seed_unique * random_hit_observed_prob)
            if np.isfinite(random_hit_observed_prob)
            else np.nan
        )
        random_expected_recovery_fraction_observed = (
            float(expected_random_recovered / observed_reachable_non_seed)
            if observed_reachable_non_seed > 0 and np.isfinite(expected_random_recovered)
            else np.nan
        )
        recovered_enrichment_vs_random = (
            float(recovered_unique / expected_random_recovered)
            if np.isfinite(expected_random_recovered) and expected_random_recovered > 0
            else np.nan
        )

        # Exact finite-population null for output-size matched random selection.
        output_size_random_std_recovered = np.nan
        output_size_random_ci95_recovered_low = np.nan
        output_size_random_ci95_recovered_high = np.nan
        output_size_random_ci95_fraction_low = np.nan
        output_size_random_ci95_fraction_high = np.nan
        output_size_random_p_ge_observed = np.nan
        if reachable_non_seed > 0 and explored_non_seed_unique >= 0:
            N = int(reachable_non_seed)
            K = int(observed_reachable_non_seed)
            n = int(explored_non_seed_unique)
            c_obs = int(recovered_unique)
            if N >= 1 and 0 <= K <= N and 0 <= n <= N:
                if N > 1:
                    p = K / N
                    output_size_random_std_recovered = float(
                        math.sqrt(n * p * (1.0 - p) * ((N - n) / (N - 1)))
                    )
                ci_low, ci_high = hypergeom.interval(0.95, N, K, n)
                output_size_random_ci95_recovered_low = float(ci_low)
                output_size_random_ci95_recovered_high = float(ci_high)
                if observed_reachable_non_seed > 0:
                    output_size_random_ci95_fraction_low = float(ci_low / observed_reachable_non_seed)
                    output_size_random_ci95_fraction_high = float(ci_high / observed_reachable_non_seed)
                output_size_random_p_ge_observed = float(hypergeom.sf(c_obs - 1, N, K, n))

        rows.append(
            {
                "origin_id": origin_int,
                "max_search_iteration": max_search_iteration,
                "effective_beam_width": effective_beam_width,
                "reachable_space_est": reachable_space_est,
                "token_space_total_est": token_space_total_est,
                "novel_reachable_est": novel_reachable_est,
                "explored_unique_in_reachable": explored_unique_in_reachable,
                "novel_explored_unique": novel_explored_unique,
                "recovered_unique": recovered_unique,
                "observed_pool_unique": observed_pool_unique,
                "observed_reachable_non_seed": observed_reachable_non_seed,
                "novel_explored_fraction": novel_explored_fraction,
                "recovered_fraction_observed": recovered_fraction_observed,
                "explored_fraction_reachable": explored_fraction_reachable,
                "explored_fraction_total_space": explored_fraction_total_space,
                "random_hit_observed_prob": random_hit_observed_prob,
                "explored_non_seed_unique": explored_non_seed_unique,
                "expected_random_recovered": expected_random_recovered,
                "random_expected_recovery_fraction_observed": random_expected_recovery_fraction_observed,
                "recovered_enrichment_vs_random": recovered_enrichment_vs_random,
                "output_size_random_std_recovered": output_size_random_std_recovered,
                "output_size_random_ci95_recovered_low": output_size_random_ci95_recovered_low,
                "output_size_random_ci95_recovered_high": output_size_random_ci95_recovered_high,
                "output_size_random_ci95_fraction_low": output_size_random_ci95_fraction_low,
                "output_size_random_ci95_fraction_high": output_size_random_ci95_fraction_high,
                "output_size_random_p_ge_observed": output_size_random_p_ge_observed,
            }
        )

    return pd.DataFrame(rows)


def _expand_frontier_pairs_random(
    frontier_pairs: list[tuple[str, str]],
    token_choices: dict[str, list[list[str]]],
    seen_pairs: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    expanded_unique: list[tuple[str, str]] = []
    expanded_seen: set[tuple[str, str]] = set()

    for guide_str, passenger_str in frontier_pairs:
        canonical_guide, canonical_passenger = make_pair_key(guide_str, passenger_str)
        guide_tokens = guide_helm_tokens(canonical_guide, canonical_passenger)
        passenger_tokens = passenger_helm_tokens(canonical_guide, canonical_passenger)

        if len(guide_tokens) != len(token_choices["guide"]) or len(passenger_tokens) != len(token_choices["passenger"]):
            continue

        for position, allowed_tokens in enumerate(token_choices["guide"]):
            current_token = guide_tokens[position]
            for token in allowed_tokens:
                if token == current_token:
                    continue
                mutated = list(guide_tokens)
                mutated[position] = token
                _, next_guide, _ = helm_tokens_to_fennec_pair(mutated, passenger_tokens)
                pair = make_pair_key(next_guide, canonical_passenger)
                if pair in seen_pairs or pair in expanded_seen:
                    continue
                expanded_seen.add(pair)
                expanded_unique.append(pair)

        for position, allowed_tokens in enumerate(token_choices["passenger"]):
            current_token = passenger_tokens[position]
            for token in allowed_tokens:
                if token == current_token:
                    continue
                mutated = list(passenger_tokens)
                mutated[position] = token
                _, _, next_passenger = helm_tokens_to_fennec_pair(guide_tokens, mutated)
                pair = make_pair_key(canonical_guide, next_passenger)
                if pair in seen_pairs or pair in expanded_seen:
                    continue
                expanded_seen.add(pair)
                expanded_unique.append(pair)

    return expanded_unique


def _build_iteration_distance_profile(
    origin_group: pd.DataFrame,
    seed_pair: tuple[str, str],
) -> dict[int, dict[int, int]]:
    profile: dict[int, dict[int, int]] = {}
    seed_guide_tokens = guide_helm_tokens(seed_pair[0], seed_pair[1])
    seed_passenger_tokens = passenger_helm_tokens(seed_pair[0], seed_pair[1])

    seen_pairs: set[tuple[str, str]] = set()
    for row in origin_group.itertuples(index=False):
        iteration = int(row.search_iteration)
        if iteration <= 0:
            continue

        pair = make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
        if pair in seen_pairs or pair == seed_pair:
            continue
        seen_pairs.add(pair)

        guide_tokens = guide_helm_tokens(pair[0], pair[1])
        passenger_tokens = passenger_helm_tokens(pair[0], pair[1])
        distance = _mutation_operation_distance(
            seed_guide_tokens,
            seed_passenger_tokens,
            guide_tokens,
            passenger_tokens,
        )
        if distance is None:
            continue

        if iteration not in profile:
            profile[iteration] = {}
        profile[iteration][distance] = profile[iteration].get(distance, 0) + 1

    return profile


def _sample_indices_distance_stratified(
    candidate_distances: list[int | None],
    target_counts: dict[int, int],
    n_select: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_select <= 0 or not candidate_distances:
        return []

    by_distance: dict[int, list[int]] = {}
    for idx, distance in enumerate(candidate_distances):
        if distance is None:
            continue
        by_distance.setdefault(int(distance), []).append(idx)

    selected: list[int] = []
    used: set[int] = set()
    goal = min(int(n_select), len(candidate_distances))

    for distance, desired_count in sorted(target_counts.items(), key=lambda item: item[0]):
        if len(selected) >= goal:
            break
        needed = max(int(desired_count), 0)
        if needed == 0:
            continue
        pool = [idx for idx in by_distance.get(int(distance), []) if idx not in used]
        if not pool:
            continue
        take = min(needed, len(pool), goal - len(selected))
        if take <= 0:
            continue
        picks = rng.choice(pool, size=take, replace=False)
        for pick in np.atleast_1d(picks):
            pick_int = int(pick)
            if pick_int in used:
                continue
            used.add(pick_int)
            selected.append(pick_int)

    if len(selected) < goal:
        remaining = [idx for idx in range(len(candidate_distances)) if idx not in used]
        if remaining:
            take = min(goal - len(selected), len(remaining))
            picks = rng.choice(remaining, size=take, replace=False)
            for pick in np.atleast_1d(picks):
                pick_int = int(pick)
                if pick_int in used:
                    continue
                used.add(pick_int)
                selected.append(pick_int)

    return selected


def _simulate_single_origin_random_beam(
    search: ClosedWorldModEnhancerSearch,
    origin_id: int,
    max_search_iteration: int,
    effective_beam_width: int,
    rng: np.random.Generator,
    target_distance_counts_by_iteration: dict[int, dict[int, int]] | None = None,
) -> tuple[int, int]:
    seed_pair = search.seed_pair_by_origin[origin_id]
    observed_pairs = search.observed_pairs_by_origin[origin_id]
    token_choices = search.token_choices_by_origin[origin_id]
    seed_guide_tokens = guide_helm_tokens(seed_pair[0], seed_pair[1])
    seed_passenger_tokens = passenger_helm_tokens(seed_pair[0], seed_pair[1])

    seen_pairs: set[tuple[str, str]] = {seed_pair}
    frontier_pairs: list[tuple[str, str]] = [seed_pair]

    for iteration in range(1, max_search_iteration + 1):
        expanded_pairs = _expand_frontier_pairs_random(
            frontier_pairs=frontier_pairs,
            token_choices=token_choices,
            seen_pairs=seen_pairs,
        )
        if not expanded_pairs:
            break

        beam_limit = min(max(int(effective_beam_width), 1), len(expanded_pairs))
        selected_indices: list[int]
        if target_distance_counts_by_iteration is not None:
            target_counts = target_distance_counts_by_iteration.get(iteration, {})
            target_total = sum(max(int(value), 0) for value in target_counts.values())
            n_select = min(beam_limit, target_total) if target_total > 0 else beam_limit

            candidate_distances: list[int | None] = []
            for guide_str, passenger_str in expanded_pairs:
                candidate_distances.append(
                    _mutation_operation_distance(
                        seed_guide_tokens,
                        seed_passenger_tokens,
                        guide_helm_tokens(guide_str, passenger_str),
                        passenger_helm_tokens(guide_str, passenger_str),
                    )
                )
            selected_indices = _sample_indices_distance_stratified(
                candidate_distances=candidate_distances,
                target_counts=target_counts,
                n_select=n_select,
                rng=rng,
            )
        else:
            picks = rng.choice(len(expanded_pairs), size=beam_limit, replace=False)
            selected_indices = [int(idx) for idx in np.atleast_1d(picks)]

        if not selected_indices:
            break

        frontier_pairs = [expanded_pairs[int(idx)] for idx in selected_indices]
        seen_pairs.update(frontier_pairs)

    recovered = len((seen_pairs & observed_pairs) - {seed_pair})
    explored_non_seed_unique = max(len(seen_pairs) - 1, 0)
    return int(recovered), int(explored_non_seed_unique)


def simulate_compute_matched_random_control(
    search_results: pd.DataFrame,
    search: ClosedWorldModEnhancerSearch,
    origin_metrics: pd.DataFrame,
    n_simulations: int,
    random_seed: int,
    distance_stratified: bool = False,
) -> pd.DataFrame:
    if n_simulations <= 0:
        return pd.DataFrame(columns=["origin_id"])

    metrics_by_origin = origin_metrics.set_index("origin_id", drop=False)
    origin_rows: list[dict[str, float | int]] = []
    rng_master = np.random.default_rng(random_seed)

    grouped_results = search_results.groupby("origin_id", dropna=False)
    for group_index, (origin_id, group) in enumerate(grouped_results):
        origin_int = int(origin_id)
        if origin_int not in metrics_by_origin.index:
            continue

        seed_pair = search.seed_pair_by_origin[origin_int]
        target_distance_profile = (
            _build_iteration_distance_profile(group, seed_pair)
            if distance_stratified
            else None
        )

        max_search_iteration = int(group["search_iteration"].max())
        if "effective_beam_width" in group.columns and group["effective_beam_width"].notna().any():
            effective_beam_width = int(group["effective_beam_width"].dropna().iloc[0])
        else:
            effective_beam_width = int(
                search.effective_beam_width_by_origin.get(
                    origin_int,
                    search.beam_config.beam_width,
                )
            )

        origin_rng = np.random.default_rng(rng_master.integers(0, np.iinfo(np.uint32).max))

        recovered_samples = np.zeros(n_simulations, dtype=float)
        explored_non_seed_samples = np.zeros(n_simulations, dtype=float)
        for sim_idx in range(n_simulations):
            sim_rng = np.random.default_rng(origin_rng.integers(0, np.iinfo(np.uint32).max))
            recovered, explored_non_seed = _simulate_single_origin_random_beam(
                search=search,
                origin_id=origin_int,
                max_search_iteration=max_search_iteration,
                effective_beam_width=effective_beam_width,
                rng=sim_rng,
                target_distance_counts_by_iteration=target_distance_profile,
            )
            recovered_samples[sim_idx] = recovered
            explored_non_seed_samples[sim_idx] = explored_non_seed

        observed_recovered = float(metrics_by_origin.loc[origin_int, "recovered_unique"])
        observed_reachable_non_seed = float(metrics_by_origin.loc[origin_int, "observed_reachable_non_seed"])

        recovered_mean = float(np.mean(recovered_samples))
        recovered_std = float(np.std(recovered_samples, ddof=1)) if n_simulations > 1 else 0.0
        recovered_ci95_low = float(np.quantile(recovered_samples, 0.025))
        recovered_ci95_high = float(np.quantile(recovered_samples, 0.975))
        p_ge_observed = float(np.mean(recovered_samples >= observed_recovered))

        if observed_reachable_non_seed > 0:
            frac_mean = float(recovered_mean / observed_reachable_non_seed)
            frac_ci95_low = float(recovered_ci95_low / observed_reachable_non_seed)
            frac_ci95_high = float(recovered_ci95_high / observed_reachable_non_seed)
        else:
            frac_mean = np.nan
            frac_ci95_low = np.nan
            frac_ci95_high = np.nan

        enrichment_vs_random_mean = (
            float(observed_recovered / recovered_mean)
            if recovered_mean > 0
            else np.nan
        )

        origin_rows.append(
            {
                "origin_id": origin_int,
                "compute_random_n_simulations": int(n_simulations),
                "compute_random_control_mode": (
                    "compute-matched-distance-stratified"
                    if distance_stratified
                    else "compute-matched"
                ),
                "compute_random_expected_recovered": recovered_mean,
                "compute_random_std_recovered": recovered_std,
                "compute_random_ci95_recovered_low": recovered_ci95_low,
                "compute_random_ci95_recovered_high": recovered_ci95_high,
                "compute_random_p_ge_observed": p_ge_observed,
                "compute_random_expected_recovery_fraction_observed": frac_mean,
                "compute_random_ci95_fraction_low": frac_ci95_low,
                "compute_random_ci95_fraction_high": frac_ci95_high,
                "compute_recovered_enrichment_vs_random": enrichment_vs_random_mean,
                "compute_random_expected_explored_non_seed": float(np.mean(explored_non_seed_samples)),
            }
        )

    return pd.DataFrame(origin_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the notebook-style Mod_enhancer summary PNG from standalone search outputs."
    )
    parser.add_argument("--train-data", required=True, help="Train data CSV/CSV.GZ used to rebuild the selected condition pools")
    parser.add_argument("--selected-conditions", required=True, help="CSV written by --output-candidates from the redesign search")
    parser.add_argument("--search-results", required=True, help="CSV written by --output-results from the redesign search")
    parser.add_argument("--output-png", required=True, help="PNG path for the final figure")
    parser.add_argument(
        "--output-iteration-png",
        help="Optional PNG path for iteration trend plot (x=search_iteration, y=mean_pred, hue=origin_id)",
    )
    parser.add_argument("--output-plot-data", help="Optional CSV path for the plotted observed-design table")
    parser.add_argument("--rna-emb-path", default=".", help="RNA embedding directory passed to FENNEC preprocessing")
    parser.add_argument("--model-dir", help="Directory containing model checkpoints")
    parser.add_argument("--model-path", action="append", help="Specific model checkpoint path. Repeat as needed")
    parser.add_argument("--model-type", default="TCN_FM_multi_head")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fm-type", default="orthrus_4track_triple")
    parser.add_argument("--slice-length", type=int, default=100)
    parser.add_argument("--with-thermo", action="store_true")
    parser.add_argument("--only-cls-eos", action="store_true")
    parser.add_argument("--title", default="Per-siRNA performance (true vs. mean_pred)")
    parser.add_argument(
        "--iteration-title",
        default="Per-origin search trend across iterations",
        help="Title for optional iteration trend plot",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--distance-json",
        default=str(DEFAULT_DISTANCE_JSON),
        help="Per-siRNA token distance JSON from ablation/vizzes",
    )
    parser.add_argument(
        "--seed-origin",
        choices=["worst", "worst_predicted", "centroid", "centroid_lowest_neighbor"],
        default="worst",
        help="Seed origin selection used when rebuilding pools for plotting",
    )
    parser.add_argument(
        "--centroid-neighbor-radius",
        type=int,
        default=2,
        help="Edit distance radius around centroid for neighbor selection (used with centroid_lowest_neighbor)",
    )
    parser.add_argument(
        "--open-world-scope",
        choices=["cohort", "cohort-pool", "cohort-pool-all", "global"],
        default="cohort",
        help=(
            "Source position-specific mutation tokens from selected cohorts "
            "(per-origin, position-aware across all local shapes), pooled by shape across origins (cohort-pool), "
            "pooled across all selected origins regardless of shape with "
            "position-aware applicability (cohort-pool-all), or from all "
            "train-data designs by shape (global)"
        ),
    )
    parser.add_argument(
        "--annotation-style",
        choices=["full", "succinct", "none"],
        default="full",
        help="Panel annotation verbosity: full details, succinct summary, or none",
    )
    parser.add_argument(
        "--annotation-placement",
        choices=["overlay", "side", "bottom"],
        default="overlay",
        help="Where to place metrics text: overlay in each panel, in a right-side summary block (side), or below all plots (bottom)",
    )
    parser.add_argument(
        "--random-control-mode",
        choices=["none", "compute-matched", "compute-matched-distance-stratified"],
        default="none",
        help=(
            "Optional additional random baseline. "
            "'compute-matched' runs randomized beam simulations with the same iteration/beam constraints; "
            "'compute-matched-distance-stratified' additionally matches per-iteration edit-distance profile."
        ),
    )
    parser.add_argument(
        "--random-control-simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations per origin for --random-control-mode compute-matched",
    )
    parser.add_argument(
        "--random-control-seed",
        type=int,
        default=20260501,
        help="Random seed for compute-matched random control simulations",
    )
    parser.add_argument(
        "--parent-arrow-mode",
        choices=["none", "observed"],
        default="none",
        help=(
            "Arrow overlay mode for the final plot. "
            "'observed' draws parent->child arrows only when both nodes are real observed sequences present in the panel."
        ),
    )
    return parser


def calc_key_stats_notebook_style(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return float("nan"), float("nan")

    y_true_median = float(np.median(y_true))
    y_pred_median = float(np.median(y_pred))
    y_true_class = [0 if value <= y_true_median else 1 for value in y_true]
    y_pred_class = [0 if value <= y_pred_median else 1 for value in y_pred]

    spear = float(spearmanr(y_true, y_pred).correlation)
    try:
        auc = float(roc_auc_score(y_true_class, y_pred_class))
    except ValueError:
        auc = float("nan")
    return spear, auc


def _parse_seed_pair_key(value: object) -> tuple[str, str] | None:
    if isinstance(value, tuple) and len(value) >= 2:
        return make_pair_key(value[0], value[1])
    if isinstance(value, list) and len(value) >= 2:
        return make_pair_key(value[0], value[1])
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                return make_pair_key(parsed[0], parsed[1])
        except Exception:
            return None
    return None


def _resolve_search_result_origins(
    search_results: pd.DataFrame,
    search: ClosedWorldModEnhancerSearch,
) -> pd.DataFrame:
    if "seed_pair_key" not in search_results.columns:
        return search_results

    pair_to_origin = {
        make_pair_key(pair[0], pair[1]): int(origin_id)
        for origin_id, pair in search.seed_pair_by_origin.items()
    }

    resolved_origins: list[int | float] = []
    changed = False
    for row in search_results.itertuples(index=False):
        current_origin = int(row.origin_id)
        parsed_pair = _parse_seed_pair_key(getattr(row, "seed_pair_key", None))
        if parsed_pair is None:
            resolved_origins.append(current_origin)
            continue
        remapped_origin = pair_to_origin.get(parsed_pair, current_origin)
        changed = changed or (remapped_origin != current_origin)
        resolved_origins.append(remapped_origin)

    if not changed:
        return search_results

    aligned = search_results.copy()
    aligned["origin_id"] = resolved_origins
    return aligned


def build_observed_prediction_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, ClosedWorldModEnhancerSearch]:
    train_df = load_train_data(Path(args.train_data))
    selected_conditions = pd.read_csv(args.selected_conditions)
    model_paths = resolve_model_paths(Path(args.model_dir) if args.model_dir else None, args.model_path)

    distance_payload = {}
    if getattr(args, "seed_origin", "worst") in {"centroid", "centroid_lowest_neighbor"}:
        try:
            distance_payload = load_distance_payload(Path(args.distance_json))
        except Exception:
            distance_payload = {}

    search = ClosedWorldModEnhancerSearch(
        train_df=train_df,
        selected_conditions=selected_conditions,
        distance_payload=distance_payload,
        model_paths=model_paths,
        beam_config=BeamSearchConfig(
            model_type=args.model_type,
            device=args.device,
            fm_type=args.fm_type,
            slice_length=args.slice_length,
            with_thermo=args.with_thermo,
            only_cls_eos=args.only_cls_eos,
            open_world_scope=getattr(args, "open_world_scope", "cohort"),
            seed_origin=getattr(args, "seed_origin", "worst"),
            centroid_neighbor_radius=getattr(args, "centroid_neighbor_radius", 2),
        ),
        rna_emb_path=Path(args.rna_emb_path),
    )

    observed_rows: list[pd.DataFrame] = []
    for origin_id, pool_df in search.pool_by_origin.items():
        pool_df = ensure_helm_sequence_columns(pool_df)
        unique_pool = (
            pool_df.sort_values(by="mRNA_remaining_pct", ascending=False)
            .drop_duplicates(subset=[HELM_COLUMN], keep="first")
            .copy()
        )
        unique_pool["origin_id"] = origin_id
        unique_pool["candidate_rank"] = search.seed_df.loc[
            search.seed_df["origin_id"] == origin_id, "candidate_rank"
        ].iloc[0]
        unique_pool["seed_pair_key"] = [search.seed_pair_by_origin[origin_id]] * len(unique_pool)
        # record which seed-origin was used so plots can annotate correctly
        unique_pool["seed_origin"] = [search.beam_config.seed_origin] * len(unique_pool)
        observed_rows.append(unique_pool)

    observed_df = pd.concat(observed_rows, ignore_index=True)
    observed_predictions = search._predict_designs(observed_df)
    return observed_predictions, search


def annotate_plot_status(
    observed_predictions: pd.DataFrame,
    search_results: pd.DataFrame,
    search: ClosedWorldModEnhancerSearch,
    random_control_mode: str = "none",
    random_control_simulations: int = 1000,
    random_control_seed: int = 20260501,
    parent_arrow_mode: str = "none",
) -> pd.DataFrame:
    frame = observed_predictions.copy()
    aligned_search_results = _resolve_search_result_origins(search_results, search)

    recovered_pairs_by_origin: dict[int, set[tuple[str, str]]] = {}
    for origin_id, group in aligned_search_results.groupby("origin_id", dropna=False):
        origin_int = int(origin_id)
        observed_pairs = search.observed_pairs_by_origin.get(origin_int, set())
        recovered_pairs_by_origin[origin_int] = {
            make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
            for row in group.itertuples(index=False)
            if make_pair_key(row.siRNA_list, row.siRNA_passenger_list) in observed_pairs
        }

    statuses: list[str] = []
    for row in frame.itertuples(index=False):
        origin_id = int(row.origin_id)
        pair_key = make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
        if pair_key == search.seed_pair_by_origin[origin_id]:
            statuses.append("Unoptimized Least Effective siRNA Chemical Design")
        elif pair_key in recovered_pairs_by_origin.get(origin_id, set()):
            statuses.append("Redesigned Match Recovered")
        else:
            statuses.append("Experimental siRNA Match missing from AdaLead designs")

    frame["Design Status"] = pd.Categorical(statuses, categories=DESIGN_STATUS_ORDER, ordered=True)
    origin_metrics = build_origin_search_metrics(aligned_search_results, search)
    if random_control_mode in {"compute-matched", "compute-matched-distance-stratified"} and random_control_simulations > 0:
        compute_random = simulate_compute_matched_random_control(
            search_results=aligned_search_results,
            search=search,
            origin_metrics=origin_metrics,
            n_simulations=int(random_control_simulations),
            random_seed=int(random_control_seed),
            distance_stratified=(random_control_mode == "compute-matched-distance-stratified"),
        )
        if not compute_random.empty:
            origin_metrics = origin_metrics.merge(compute_random, on="origin_id", how="left")
    frame = frame.merge(origin_metrics, on="origin_id", how="left")
    frame = _attach_parent_plot_links(
        frame,
        aligned_search_results,
        parent_arrow_mode=parent_arrow_mode,
    )
    return frame


def _attach_parent_plot_links(
    plot_frame: pd.DataFrame,
    aligned_search_results: pd.DataFrame,
    parent_arrow_mode: str,
) -> pd.DataFrame:
    if parent_arrow_mode == "none":
        return plot_frame

    required_cols = {
        "origin_id",
        "design_id",
        "parent_design_id",
        "siRNA_list",
        "siRNA_passenger_list",
    }
    if not required_cols.issubset(aligned_search_results.columns):
        return plot_frame

    lineage = aligned_search_results.copy()
    lineage["design_id"] = lineage["design_id"].astype("string")
    lineage["parent_design_id"] = lineage["parent_design_id"].astype("string")
    lineage["pair_key"] = [
        make_pair_key(guide, passenger)
        for guide, passenger in zip(lineage["siRNA_list"], lineage["siRNA_passenger_list"])
    ]
    lineage = lineage.drop_duplicates(subset=["origin_id", "pair_key"], keep="first").copy()

    parent_lookup = lineage[["design_id", "origin_id", "pair_key", "search_iteration"]].rename(
        columns={
            "design_id": "parent_design_id",
            "origin_id": "parent_origin_id",
            "pair_key": "parent_pair_key",
            "search_iteration": "parent_search_iteration",
        }
    )
    lineage = lineage.merge(parent_lookup, on="parent_design_id", how="left")

    frame = plot_frame.copy()
    stale_lineage_columns = [
        "pair_key",
        "search_design_id",
        "search_parent_design_id",
        "parent_pair_key",
        "parent_search_iteration",
        "observed_parent_available",
        "traceback_parent_pair_key",
        "traceback_parent_design_id",
        "traceback_parent_search_iteration",
        "traceback_parent_hops",
    ]
    frame = frame.drop(columns=[col for col in stale_lineage_columns if col in frame.columns])
    frame["pair_key"] = [
        make_pair_key(guide, passenger)
        for guide, passenger in zip(frame["siRNA_list"], frame["siRNA_passenger_list"])
    ]

    observed_pairs = {
        (int(origin_id), pair_key)
        for origin_id, pair_key in zip(frame["origin_id"], frame["pair_key"])
    }
    lineage["observed_parent_available"] = [
        (int(origin_id), parent_pair_key) in observed_pairs if isinstance(parent_pair_key, tuple) else False
        for origin_id, parent_pair_key in zip(lineage["origin_id"], lineage["parent_pair_key"])
    ]

    design_lookup = lineage.set_index("design_id")[
        ["parent_design_id", "pair_key", "search_iteration", "match_status", "origin_id"]
    ].to_dict("index")

    traceback_parent_pair_keys: list[tuple[str, str] | None] = []
    traceback_parent_design_ids: list[str | None] = []
    traceback_parent_iterations: list[float] = []
    traceback_hops: list[float] = []
    for row in lineage.itertuples(index=False):
        traced_pair_key: tuple[str, str] | None = None
        traced_design_id: str | None = None
        traced_iteration = np.nan
        traced_hop_count = np.nan

        if str(getattr(row, "match_status", "")) == "RecoveredObserved" and not bool(row.observed_parent_available):
            current_parent_design_id = row.parent_design_id
            visited_design_ids: set[str] = set()
            hop_count = 0
            while isinstance(current_parent_design_id, str) and current_parent_design_id and current_parent_design_id not in visited_design_ids:
                visited_design_ids.add(current_parent_design_id)
                parent_row = design_lookup.get(current_parent_design_id)
                if parent_row is None:
                    break
                hop_count += 1
                parent_pair_key = parent_row["pair_key"]
                if isinstance(parent_pair_key, tuple) and (int(row.origin_id), parent_pair_key) in observed_pairs:
                    traced_pair_key = parent_pair_key
                    traced_design_id = current_parent_design_id
                    traced_iteration = float(parent_row["search_iteration"])
                    traced_hop_count = float(hop_count)
                    break
                current_parent_design_id = parent_row["parent_design_id"]

        traceback_parent_pair_keys.append(traced_pair_key)
        traceback_parent_design_ids.append(traced_design_id)
        traceback_parent_iterations.append(traced_iteration)
        traceback_hops.append(traced_hop_count)

    lineage["traceback_parent_pair_key"] = traceback_parent_pair_keys
    lineage["traceback_parent_design_id"] = traceback_parent_design_ids
    lineage["traceback_parent_search_iteration"] = traceback_parent_iterations
    lineage["traceback_parent_hops"] = traceback_hops

    if parent_arrow_mode == "observed":
        lineage.loc[~lineage["observed_parent_available"], "parent_pair_key"] = None

    lineage_subset = lineage[
        [
            "origin_id",
            "pair_key",
            "design_id",
            "parent_design_id",
            "parent_pair_key",
            "parent_search_iteration",
            "observed_parent_available",
            "traceback_parent_pair_key",
            "traceback_parent_design_id",
            "traceback_parent_search_iteration",
            "traceback_parent_hops",
        ]
    ].rename(
        columns={
            "design_id": "search_design_id",
            "parent_design_id": "search_parent_design_id",
        }
    )
    return frame.merge(lineage_subset, on=["origin_id", "pair_key"], how="left")


def _select_display_baseline(sub: pd.DataFrame) -> dict[str, float | str]:
    compute_available = (
        "compute_random_expected_recovery_fraction_observed" in sub.columns
        and sub["compute_random_expected_recovery_fraction_observed"].notna().any()
    )

    if compute_available:
        mode_value = "compute-matched"
        if "compute_random_control_mode" in sub.columns and sub["compute_random_control_mode"].notna().any():
            mode_value = str(sub["compute_random_control_mode"].iloc[0])
        label = "Compute-matched"
        if mode_value == "compute-matched-distance-stratified":
            # keep simple for plot
            label = "Compute-based"

        expected_count = np.nan
        if "compute_random_expected_recovered" in sub.columns:
            expected_count = float(sub["compute_random_expected_recovered"].iloc[0])

        p_value = np.nan
        if "compute_random_p_ge_observed" in sub.columns:
            p_value = float(sub["compute_random_p_ge_observed"].iloc[0])

        enrichment = np.nan
        if "compute_recovered_enrichment_vs_random" in sub.columns:
            enrichment = float(sub["compute_recovered_enrichment_vs_random"].iloc[0])

        return {
            "label": label,
            "expected_fraction": float(sub["compute_random_expected_recovery_fraction_observed"].iloc[0]),
            "expected_count": expected_count,
            "enrichment": enrichment,
            "p_value": p_value,
        }

    return {
        "label": "Random (output-size)",
        "expected_fraction": float(sub["random_expected_recovery_fraction_observed"].iloc[0]),
        "expected_count": float(sub["expected_random_recovered"].iloc[0]),
        "enrichment": float(sub["recovered_enrichment_vs_random"].iloc[0]),
        "p_value": np.nan,
    }


def _build_overlay_annotation_text(sub: pd.DataFrame, spear: float, auc: float, style: str) -> str:
    if style == "none":
        return ""

    baseline = _select_display_baseline(sub)
    baseline_label = str(baseline["label"])
    baseline_expected_frac = float(baseline["expected_fraction"])
    baseline_expected_count = float(baseline["expected_count"])
    baseline_enrichment = float(baseline["enrichment"])
    baseline_p_value = float(baseline["p_value"])
    observed_reachable = float(sub["observed_reachable_non_seed"].iloc[0])
    recovered_fraction = _format_recovery_fraction(sub["recovered_fraction_observed"].iloc[0], observed_reachable)
    baseline_fraction = _format_recovery_fraction(baseline_expected_frac, observed_reachable)

    if style == "succinct":
        return (
            f"N={len(sub)} | Spearman={spear:.3f}\n"
            f"Beam={int(sub['effective_beam_width'].iloc[0]) if not pd.isna(sub['effective_beam_width'].iloc[0]) else 'n/a'} | "
            f"Iter={int(sub['max_search_iteration'].iloc[0])} | AUC={auc:.3f}\n"
            #f"Space <=I/Total={_format_compact_int(sub['reachable_space_est'].iloc[0])}/"
            #f"{_format_compact_int(sub['token_space_total_est'].iloc[0])}\n"
            f"Explored {_format_compact_int(sub['explored_unique_in_reachable'].iloc[0])}/"
            f"{_format_compact_int(sub['reachable_space_est'].iloc[0])} ({_format_percent(sub['explored_fraction_reachable'].iloc[0])})\n"
            f"Recovered {int(sub['recovered_unique'].iloc[0])}/{int(sub['observed_reachable_non_seed'].iloc[0])} "
            f"({recovered_fraction}) \n "
            f"CBER*={baseline_fraction}\n"
            f"REF*={_format_multiplier_x(baseline_enrichment)}"
        )

    baseline_line_full = (
        f"\n{baseline_label} expected recovered (budget): "
        f"{0.0 if pd.isna(baseline_expected_count) else baseline_expected_count:.2f}/"
        f"{int(sub['observed_reachable_non_seed'].iloc[0])} "
        f"({baseline_fraction})"
    )
    if np.isfinite(baseline_p_value):
        baseline_line_full += f" | p(X>=obs)={baseline_p_value:.3f}"

    return (
        f"N={len(sub)} | Spearman={spear:.3f} | AUC={auc:.3f}\n"
        f"Beam width (effective): {int(sub['effective_beam_width'].iloc[0]) if not pd.isna(sub['effective_beam_width'].iloc[0]) else 'n/a'} | Iteration depth: {int(sub['max_search_iteration'].iloc[0])}\n"
        f"Search space (total / <=iter): {_format_compact_int(sub['token_space_total_est'].iloc[0])} / {_format_compact_int(sub['reachable_space_est'].iloc[0])}\n"
        f"Explored (count): {_format_compact_int(sub['explored_unique_in_reachable'].iloc[0])} | "
        f"Coverage (<=iter / total): "
        f"{_format_percent(sub['explored_fraction_reachable'].iloc[0])} / "
        f"{_format_percent(sub['explored_fraction_total_space'].iloc[0])}\n"
        f"Recovered: {int(sub['recovered_unique'].iloc[0])}/{int(sub['observed_reachable_non_seed'].iloc[0])} "
        f"({recovered_fraction})\n"
        f"Directed enrichment vs baseline: {_format_multiplier_x(baseline_enrichment)}"
        f"{baseline_line_full}"
    )


def _build_side_summary_line(sub: pd.DataFrame, spear: float, auc: float, style: str) -> str:
    origin_id = int(sub["origin_id"].iloc[0])
    gene = str(sub["gene_target_symbol_name"].iloc[0])
    beam = int(sub["effective_beam_width"].iloc[0]) if not pd.isna(sub["effective_beam_width"].iloc[0]) else "n/a"
    iteration_depth = int(sub["max_search_iteration"].iloc[0])
    explored = _format_compact_int(sub["explored_unique_in_reachable"].iloc[0])
    reachable = _format_compact_int(sub["reachable_space_est"].iloc[0])
    recovered = int(sub["recovered_unique"].iloc[0])
    observed = int(sub["observed_reachable_non_seed"].iloc[0])
    baseline = _select_display_baseline(sub)
    baseline_frac = _format_recovery_fraction(baseline["expected_fraction"], observed)
    recovered_frac = _format_recovery_fraction(sub["recovered_fraction_observed"].iloc[0], observed)
    baseline_enrichment = _format_multiplier_x(baseline["enrichment"])
    baseline_p_value = float(baseline["p_value"])

    if style == "full":
        baseline_tail = f"; BaseRec={baseline_frac}; AxBase={baseline_enrichment}"
        if np.isfinite(baseline_p_value):
            baseline_tail += f" (p={baseline_p_value:.3f})"
        return (
            f"O={origin_id} G={gene}: N={len(sub)}, Spear={spear:.3f}, AUC={auc:.3f}; "
            f"B={beam}, I={iteration_depth}; Exp={explored}/{reachable} ({_format_percent(sub['explored_fraction_reachable'].iloc[0])}); "
            f"Rec={recovered}/{observed} reachable-observed<=I ({recovered_frac})"
            f"{baseline_tail}"
        )

    return (
        f"O={origin_id} G={gene}: B={beam}, I={iteration_depth}, "
        f"Exp={explored}/{reachable} ({_format_percent(sub['explored_fraction_reachable'].iloc[0])}), "
        f"Rec={recovered}/{observed} reachable-observed<=I ({recovered_frac}), "
        f"BaseRec={baseline_frac}, "
        f"AxBase={baseline_enrichment}"
    )


def plot_final_graph(
    plot_df: pd.DataFrame,
    output_png: Path,
    title: str,
    dpi: int,
    annotation_style: str,
    annotation_placement: str,
) -> None:
    plot_df = plot_df.copy()
    if "effective_beam_width" not in plot_df.columns:
        plot_df["effective_beam_width"] = np.nan

    if "explored_fraction_total_space" not in plot_df.columns:
        explored_total = plot_df.get("explored_unique_in_reachable", pd.Series(np.nan, index=plot_df.index)).astype(float)
        token_space_total = plot_df.get("token_space_total_est", pd.Series(np.nan, index=plot_df.index)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_total = explored_total / token_space_total
        plot_df["explored_fraction_total_space"] = frac_total.where(token_space_total > 0, np.nan)

    # Backward compatibility: older exported plot tables may not include newer metrics.
    if "recovered_fraction_observed" not in plot_df.columns:
        denom = plot_df.get("observed_reachable_non_seed", pd.Series(np.nan, index=plot_df.index)).astype(float)
        numer = plot_df.get("recovered_unique", pd.Series(np.nan, index=plot_df.index)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = numer / denom
        plot_df["recovered_fraction_observed"] = frac.where(denom > 0, np.nan)

    if "random_hit_observed_prob" not in plot_df.columns:
        reachable_space = plot_df.get("reachable_space_est", pd.Series(np.nan, index=plot_df.index)).astype(float)
        observed_reachable = plot_df.get("observed_reachable_non_seed", pd.Series(np.nan, index=plot_df.index)).astype(float)
        reachable_non_seed = reachable_space - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            random_prob = observed_reachable / reachable_non_seed
        plot_df["random_hit_observed_prob"] = random_prob.where(reachable_non_seed > 0, np.nan)

    if "expected_random_recovered" not in plot_df.columns:
        explored_total = plot_df.get("explored_unique_in_reachable", pd.Series(np.nan, index=plot_df.index)).astype(float)
        explored_non_seed = np.maximum(explored_total - 1.0, 0.0)
        reachable_space = plot_df.get("reachable_space_est", pd.Series(np.nan, index=plot_df.index)).astype(float)
        reachable_non_seed = np.maximum(reachable_space - 1.0, 0.0)
        explored_non_seed = np.minimum(explored_non_seed, reachable_non_seed)
        random_prob = plot_df["random_hit_observed_prob"].astype(float)
        plot_df["expected_random_recovered"] = explored_non_seed * random_prob

    if "random_expected_recovery_fraction_observed" not in plot_df.columns:
        expected = plot_df["expected_random_recovered"].astype(float)
        observed = plot_df.get("observed_reachable_non_seed", pd.Series(np.nan, index=plot_df.index)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = expected / observed
        plot_df["random_expected_recovery_fraction_observed"] = frac.where(observed > 0, np.nan)

    if "recovered_enrichment_vs_random" not in plot_df.columns:
        recovered = plot_df.get("recovered_unique", pd.Series(np.nan, index=plot_df.index)).astype(float)
        expected = plot_df["expected_random_recovered"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            enrich = recovered / expected
        plot_df["recovered_enrichment_vs_random"] = enrich.where(expected > 0, np.nan)

    parent_arrow_present = "parent_pair_key" in plot_df.columns and plot_df["parent_pair_key"].notna().any()
    traceback_arrow_present = (
        "traceback_parent_pair_key" in plot_df.columns
        and plot_df["traceback_parent_pair_key"].notna().any()
    )

    sns.set_theme(style="whitegrid", context="paper")

    panel_order = (
        plot_df[["siRNA", "origin_id", "gene_target_symbol_name"]]
        .drop_duplicates(subset=["siRNA"])
        .sort_values(by=["origin_id", "gene_target_symbol_name", "siRNA"])
    )
    uniq = panel_order["siRNA"].to_numpy()
    n = len(uniq)
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))

    side_panel_enabled = annotation_placement == "side" and annotation_style != "none"
    bottom_panel_enabled = annotation_placement == "bottom" and annotation_style != "none"
    summary_panel_enabled = side_panel_enabled or bottom_panel_enabled
    side_panel_width = 3.5 if side_panel_enabled else 0.0
    bottom_panel_height = 2.0 if bottom_panel_enabled else 0.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols + side_panel_width, 4.15 * nrows + bottom_panel_height),
        dpi=dpi,
    )
    axes = np.atleast_1d(axes).ravel()
    side_summary_lines: list[str] = []

    y_true_col = "mRNA_remaining_pct"

    for idx, sirna in enumerate(uniq):
        ax = axes[idx]
        sub = plot_df[plot_df["siRNA"] == sirna].copy()
        y_true_sub = sub[y_true_col].to_numpy(dtype=float)
        y_pred_sub = sub["mean_pred"].to_numpy(dtype=float)
        spear, auc = calc_key_stats_notebook_style(y_true_sub, y_pred_sub)
        recovered = sub[sub["Design Status"] == "Redesigned Match Recovered"]
        origin_pt = sub[sub["Design Status"] == "Unoptimized Least Effective siRNA Chemical Design"]

        marker_map = {
            "Unoptimized Least Effective siRNA Chemical Design": "o",
            "Redesigned Match Recovered": "D",
            "Experimental siRNA Match missing from AdaLead designs": "X",
        }
        for status in DESIGN_STATUS_ORDER:
            status_sub = sub[sub["Design Status"] == status]
            if status_sub.empty:
                continue
            ax.scatter(
                status_sub[y_true_col],
                status_sub["mean_pred"],
                s=36,
                alpha=0.86,
                c=DESIGN_STATUS_PALETTE[status],
                marker=marker_map[status],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        if "pair_key" in sub.columns and "parent_pair_key" in sub.columns:
            point_coords = {
                row.pair_key: (float(row.mRNA_remaining_pct), float(row.mean_pred))
                for row in sub.itertuples(index=False)
                if isinstance(row.pair_key, tuple)
            }
            drawn_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
            for row in sub.itertuples(index=False):
                child_pair = getattr(row, "pair_key", None)
                parent_pair = getattr(row, "parent_pair_key", None)
                if not isinstance(child_pair, tuple) or not isinstance(parent_pair, tuple):
                    continue
                if child_pair == parent_pair:
                    continue
                if parent_pair not in point_coords or child_pair not in point_coords:
                    continue
                edge_key = (parent_pair, child_pair)
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                ax.annotate(
                    "",
                    xy=point_coords[child_pair],
                    xytext=point_coords[parent_pair],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#6B7280",
                        lw=1.0,
                        alpha=0.75,
                        shrinkA=6,
                        shrinkB=6,
                    ),
                    zorder=2.6,
                )

        if "pair_key" in sub.columns and "traceback_parent_pair_key" in sub.columns:
            drawn_traceback_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
            for row in sub.itertuples(index=False):
                child_pair = getattr(row, "pair_key", None)
                ancestor_pair = getattr(row, "traceback_parent_pair_key", None)
                if not isinstance(child_pair, tuple) or not isinstance(ancestor_pair, tuple):
                    continue
                if child_pair == ancestor_pair:
                    continue
                if ancestor_pair not in point_coords or child_pair not in point_coords:
                    continue
                edge_key = (ancestor_pair, child_pair)
                if edge_key in drawn_traceback_edges:
                    continue
                drawn_traceback_edges.add(edge_key)
                ax.annotate(
                    "",
                    xy=point_coords[child_pair],
                    xytext=point_coords[ancestor_pair],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#6B7280",
                        lw=1.1,
                        alpha=0.55,
                        linestyle=(0, (1.2, 2.2)),
                        connectionstyle="arc3,rad=0.08",
                        shrinkA=6,
                        shrinkB=6,
                    ),
                    zorder=2.45,
                )

        xmin = float(np.nanmin(y_true_sub))
        xmax = float(np.nanmax(y_true_sub))
        ymin = float(np.nanmin(y_pred_sub))
        ymax = float(np.nanmax(y_pred_sub))
        span_x = xmax - xmin
        span_y = ymax - ymin
        x_pad = 0.1 * span_x if span_x > 0 else 0.1
        y_pad = 0.1 * span_y if span_y > 0 else 0.1
        x_min_pad = xmin - x_pad
        x_max_pad = xmax + x_pad
        y_min_pad = ymin - y_pad
        y_max_pad = ymax + y_pad

        ax.set_xlim(x_max_pad, x_min_pad)
        ax.set_ylim(y_max_pad, y_min_pad)

        if not origin_pt.empty:
            ox_data = float(origin_pt.iloc[0][y_true_col])
            oy_data = float(origin_pt.iloc[0]["mean_pred"])

            # Add the legacy green gradient wash across the full panel, anchored at the start point.
            x_mesh = np.linspace(x_min_pad, x_max_pad, 100)
            y_mesh = np.linspace(y_min_pad, y_max_pad, 100)
            x_grid, y_grid = np.meshgrid(x_mesh, y_mesh)

            dx_frac = (ox_data - x_grid) / (ox_data - x_min_pad + 1e-9)
            dy_frac = (oy_data - y_grid) / (oy_data - y_min_pad + 1e-9)
            grad_val = np.maximum(0, dx_frac + dy_frac)
            if np.max(grad_val) > 0:
                grad_val = grad_val / np.max(grad_val)

            ax.imshow(
                grad_val,
                extent=[x_min_pad, x_max_pad, y_min_pad, y_max_pad],
                cmap="Greens",
                alpha=0.22,
                origin="lower",
                aspect="auto",
                zorder=-1,
            )

            # Add a translucent blue fan cone from the seed/start point to show search direction.
            # Build the cone in axes coordinates so it stays symmetric under axis scaling.
            origin_axes = ax.transAxes.inverted().transform(ax.transData.transform((ox_data, oy_data)))
            direction = np.array([1.0, 1.0], dtype=float)
            direction /= np.linalg.norm(direction)
            perpendicular = np.array([-direction[1], direction[0]], dtype=float)

            cone_length = 0.26
            cone_half_width = 0.095
            apex_axes = origin_axes + direction * cone_length
            left_axes = apex_axes + perpendicular * cone_half_width
            right_axes = apex_axes - perpendicular * cone_half_width

            fan_cone = Polygon(
                [tuple(origin_axes), tuple(left_axes), tuple(right_axes)],
                closed=True,
                transform=ax.transAxes,
                facecolor="#4C78A8",
                edgecolor="none",
                alpha=0.14,
                zorder=2,
            )
            ax.add_patch(fan_cone)

            if True: #idx == 0:
                dx_arrow = -span_x * 0.07 if span_x > 0 else -0.07
                dy_arrow = -span_y * 0.07 if span_y > 0 else -0.07
                ax.annotate(
                    "Optimization\n direction",
                    xy=(ox_data + dx_arrow, oy_data + dy_arrow),
                    xytext=(ox_data, oy_data),
                    fontsize=5,
                    ha="left",
                    va="bottom",
                    arrowprops=dict(arrowstyle="->", color="#4A4A4A", lw=1.1, alpha=0),
                    zorder=4,
                )

        panel_text = _build_overlay_annotation_text(sub, spear, auc, annotation_style)
        if annotation_placement == "overlay" and panel_text:
            ax.text(
                0.02,
                0.98,
                panel_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.4,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#C8C8C8", alpha=0.3),
                zorder=5,
            )
        elif summary_panel_enabled:
            # Keep key fit metrics in-panel for succinct publication mode.
            if annotation_style == "succinct":
                ax.text(
                    0.02,
                    0.98,
                    f"N={len(sub)} | Spearman={spear:.3f} | AUC={auc:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7.2,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#C8C8C8", alpha=0.3),
                    zorder=5,
                )
            side_summary_lines.append(_build_side_summary_line(sub, spear, auc, annotation_style))

        sirna_label = str(sirna)
        ax.set_title(
            f"siRNA {sirna_label} \nGene {sub.gene_target_symbol_name.iloc[0]} | Origin {int(sub.origin_id.iloc[0])}",
            size=10.5,
            pad=8,
        )
        ax.set_xlabel("True mRNA Remaining (%)", fontsize=10)
        ax.set_ylabel("Predicted mRNA Remaining (%)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, color="#D2D2D2", linewidth=0.8, alpha=0.85)
        for spine in ax.spines.values():
            spine.set_color("#BFBFBF")
            spine.set_linewidth(0.8)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, y=0.995)

    # Build a custom legend and replace the "Unoptimized" label with a seed-origin-aware label
    if "seed_origin" in plot_df.columns:
        seed_origin_global = plot_df["seed_origin"].mode().iloc[0]
    else:
        seed_origin_global = "worst"

    seed_label_map = {
        "worst": "Search start (worst observed)",
        "worst_predicted": "Search start (worst predicted)",
        "centroid": "Search start (centroid)",
        "centroid_lowest_neighbor": "Search start (centroid-lowest-neighbor)",
    }
    unopt_label = seed_label_map.get(seed_origin_global, "siRNA start to optimize")
    legend_label_map = {
        "Unoptimized Least Effective siRNA Chemical Design": unopt_label,
        "Redesigned Match Recovered": "Recovered experimental match",
        "Experimental siRNA Match missing from AdaLead designs": "Experimental match not recovered",
    }
    marker_map = {
        "Unoptimized Least Effective siRNA Chemical Design": "o",
        "Redesigned Match Recovered": "D",
        "Experimental siRNA Match missing from AdaLead designs": "X",
    }

    custom_handles: list[Line2D] = []
    custom_labels: list[str] = []
    for status in DESIGN_STATUS_ORDER:
        lbl = legend_label_map[status]
        handle = Line2D(
            [0],
            [0],
            marker=marker_map[status],
            color=DESIGN_STATUS_PALETTE[status],
            linestyle="None",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        custom_handles.append(handle)
        custom_labels.append(lbl)

    if parent_arrow_present:
        custom_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#6B7280",
                linewidth=1.0,
                alpha=0.75,
            )
        )
        custom_labels.append("Observed predecessor -> recovered child")

    if traceback_arrow_present:
        custom_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#6B7280",
                linewidth=1.1,
                alpha=0.55,
                linestyle=(0, (1.2, 2.2)),
            )
        )
        custom_labels.append("Observed ancestry via novel intermediates")

    if custom_handles and custom_labels:
        fig.legend(
            custom_handles,
            custom_labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.965),
            frameon=True,
            facecolor="white",
            edgecolor="#D0D0D0",
            fontsize=11,
        )

    right_bound = 0.98
    bottom_bound = 0.03
    summary_legend = (
        "Legend: O=Origin, G=Gene, B=effective beam, I=iteration depth, "
        "Exp=explored unique/reachable<=I, Rec=A/B where B is observed designs reachable within <=I steps."
    )

    if side_panel_enabled and side_summary_lines:
        right_bound = 0.80
        fig.text(
            0.81,
            0.92,
            "Per-Origin Summary",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        fig.text(
            0.81,
            0.895,
            summary_legend,
            ha="left",
            va="top",
            fontsize=7.7,
            linespacing=1.25,
        )
        fig.text(
            0.81,
            0.855,
            "\n".join(side_summary_lines),
            ha="left",
            va="top",
            fontsize=8,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C8C8C8", alpha=0.85),
        )
    elif bottom_panel_enabled and side_summary_lines:
        bottom_bound = 0.20
        fig.text(
            0.02,
            0.115,
            "Per-Origin Summary",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.095,
            summary_legend,
            ha="left",
            va="top",
            fontsize=7.7,
            linespacing=1.25,
        )
        fig.text(
            0.02,
            0.068,
            "\n".join(side_summary_lines),
            ha="left",
            va="top",
            fontsize=8,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C8C8C8", alpha=0.85),
        )

    footer_text = (
        "*REF (Recovery Enrichment Factor): observed recovery divided by the compute-matched random baseline's expected recovery.\n"
        "*CBER (Compute-Budget Expected Recovery): fraction of observed experimental designs a random search under the same "
        "beam/iteration budget is expected to recover."
    )
    footer_bottom = max(bottom_bound - 0.035, 0.005)
    fig.text(
        0.02,
        footer_bottom,
        footer_text,
        ha="left",
        va="bottom",
        fontsize=7.5,
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C8C8C8", alpha=0.85),
    )
    bottom_bound = max(bottom_bound, footer_bottom + 0.045)

    plt.tight_layout(rect=[0.02, bottom_bound, right_bound, 0.90])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_trend(
    search_results: pd.DataFrame,
    output_png: Path,
    title: str,
    dpi: int,
) -> None:
    if "search_iteration" not in search_results.columns:
        raise ValueError("search_results is missing required column 'search_iteration'")
    if "origin_id" not in search_results.columns:
        raise ValueError("search_results is missing required column 'origin_id'")
    if "mean_pred" not in search_results.columns:
        raise ValueError("search_results is missing required column 'mean_pred'")

    frame = search_results.copy()
    frame["search_iteration"] = pd.to_numeric(frame["search_iteration"], errors="coerce")
    frame["mean_pred"] = pd.to_numeric(frame["mean_pred"], errors="coerce")
    frame = frame[np.isfinite(frame["search_iteration"]) & np.isfinite(frame["mean_pred"])].copy()
    if frame.empty:
        return

    frame["origin_id"] = frame["origin_id"].astype(str)
    frame = frame.sort_values(by=["origin_id", "search_iteration", "mean_pred"]).reset_index(drop=True)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=dpi)
    ax.set_facecolor("whitesmoke")
    sns.lineplot(
        data=frame,
        x="search_iteration",
        y="mean_pred",
        hue="origin_id",
        estimator="mean",
        errorbar=None,
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlabel("ISMS Iteration Step N")
    ax.set_ylabel("Predicted mRNA Remaining (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.text(
        0.13,
        0.84,
        "Poor-performing\nstarting\nsiRNA",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.55),
    )
    ax.annotate(
        "",
        xy=(0.95, 0.28),
        xytext=(0.34, 0.82),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="->",
            color="#1F4FBF",
            lw=2.0,
            alpha=0.98,
        ),
        zorder=4,
    )
    ax.text(
        0.51,
        0.63,
        "Path of optimization",
        transform=ax.transAxes,
        rotation=-27,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#111111",
        zorder=5,
    )
    ax.text(
        1.00,
        0.18,
        "Chemically\noptimized\nsiRNA",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.55),
        clip_on=False,
    )
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Origin ID")
    # legend = ax.get_legend()
    # if legend is not None:
    #     legend.set_title("origin_id")

    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_argument_parser().parse_args()
    search_results = pd.read_csv(args.search_results)
    observed_predictions, search = build_observed_prediction_frame(args)
    plot_df = annotate_plot_status(
        observed_predictions,
        search_results,
        search,
        random_control_mode=args.random_control_mode,
        random_control_simulations=args.random_control_simulations,
        random_control_seed=args.random_control_seed,
        parent_arrow_mode=args.parent_arrow_mode,
    )

    if args.output_plot_data:
        output_plot_data = Path(args.output_plot_data)
        output_plot_data.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(output_plot_data, index=False)

    if args.output_iteration_png:
        plot_iteration_trend(
            search_results=search_results,
            output_png=Path(args.output_iteration_png),
            title=args.iteration_title,
            dpi=args.dpi,
        )

    plot_final_graph(
        plot_df,
        Path(args.output_png),
        args.title,
        args.dpi,
        annotation_style=args.annotation_style,
        annotation_placement=args.annotation_placement,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())