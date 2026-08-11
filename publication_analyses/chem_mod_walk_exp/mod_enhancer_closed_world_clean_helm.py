#!/usr/bin/env python3
"""Open-world chemical redesign search for Mod_enhancer experiments.

This module replaces the notebook-only redesign workflow with a
standalone workflow that:

1. Selects reliable siRNA condition candidates from vizzes artifacts.
2. Rebuilds the experimental pool from train data.
3. Builds an open mutational neighborhood from notebook-style positional alyn_str token pools.
4. Scores candidates with the newer seq_designer FENNEC prediction wrapper.

The distance JSON is not treated as an objective to maximize or minimize.
It is retained only as auxiliary context for seed selection and optional
debugging. The search itself explores unseen token combinations by mutating one
guide or passenger token at a time using a shared per-position chemistry pool.
By default that pool is built from the selected experimental cohorts, matching
the notebook's beam_all_new_mods-style behavior more closely than the
observed-neighbor graph. In `cohort` mode, each origin now pools tokens from
all observed designs in that origin (including different lengths) and applies
them position-wise only where the seed has that position. An optional global
mode can instead source position-specific tokens from all train-data designs
that share the same alyn_str token shape. A cross-shape pooled mode is also supported,
where tokens are pooled across all selected origins and applied position-wise
only when a position exists for the current design.

The condition metrics file name in ablation/vizzes currently contains the repo's
existing "chemicially" typo. This module keeps that default path for
compatibility.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from fennec.seq_designer.siRNA_design_pipeline import DesignParameters, siRNADesigner
from fennec.featurizer import BACKBONE_MOD_MAPPING, categorize_component  # type: ignore[import-not-found]
from fennec.helm_converter import (  # type: ignore[import-not-found]
    component_to_helm,
    create_helm_notation,
    helm_strand_to_fennec,
    helm_to_fennec,
    parse_helm_component,
    sequence_to_helm_strand,
)


DEFAULT_CONDITION_METRICS = (
    REPO_ROOT / "ablation" / "vizzes" / "ensemble_chemicially_numerous_siRNA_condition_spearman.csv"
)
DEFAULT_DISTANCE_JSON = (
    REPO_ROOT / "ablation" / "vizzes" / "siRNA_guide_passenger_token_edit_distance_matrices.json"
)

REQUIRED_TRAIN_COLUMNS = [
    "siRNA",
    "siRNA_list",
    "siRNA_passenger_list",
    "cell_line_donor",
    "siRNA_concentration",
    "gene_target_species",
    "Transfection_method",
    "gene_target_symbol_name",
    "mRNA_remaining_pct",
]

REQUIRED_ORIGIN_SEQUENCE_COLUMNS = [
    "siRNA_list",
    "siRNA_passenger_list",
]

CUSTOM_ORIGIN_METADATA_COLUMNS = [
    "cell_line_donor",
    "siRNA_concentration",
    "gene_target_species",
    "Transfection_method",
    "gene_target_symbol_name",
]


@dataclass(slots=True)
class CandidateSelectionConfig:
    min_group_spearman: float = 0.5
    min_group_n: int = 4
    top_k_conditions: int = 9
    diversity_penalty: float = 0.2
    model_name: str | None = None
    cell_line_donor: str | None = None
    gene_target_species: str | None = None
    transfection_method: str | None = None
    siRNA_concentration: float | None = None
    exclude_primary_cell_lines: bool = True
    require_shared_concentration: bool = True


@dataclass(slots=True)
class BeamSearchConfig:
    beam_width: int = 30
    num_iterations: int = 3
    confidence_multiplier: float = 1.96
    top_k_neighbors: int = 12
    max_neighbor_distance: float | None = None
    model_type: str = "TCN_FM_multi_head"
    device: str = "cuda"
    fm_type: str = "orthrus_4track"
    slice_length: int = 25
    with_thermo: bool = True
    only_cls_eos: bool = False
    open_world_scope: str = "cohort"
    seed_origin: str = "worst"
    centroid_neighbor_radius: int = 2
    adaptive_beam_cap: bool = True
    origin_beam_width_overrides: dict[int, int] | None = None
    prediction_chunk_size: int | None = None

    def score_predictions(self, frame: pd.DataFrame) -> tuple[pd.Series, bool]:
        # Lower predicted mRNA remaining is always better for redesign.
        score = frame["mean_pred"] + self.confidence_multiplier * frame["stdev"]
        return score, True


FENNEC_BACKBONE_TOKENS = frozenset(str(token) for token in BACKBONE_MOD_MAPPING.keys())
HELM_COLUMN = "HELM"
HELM_STRING_COLUMN = "HELM_String"
_RNA1_PATTERN = re.compile(r"RNA1\{([^}]*)\}")
_RNA2_PATTERN = re.compile(r"RNA2\{([^}]*)\}")


@dataclass(frozen=True, slots=True)
class HelmMonomerDescriptor:
    token: str
    base: str | None
    sugar_type: str | None
    sugar_mod: str | None
    backbone: str | None
    category: str


def explicit_helm_backbone(token: str) -> str:
    text = str(token).strip()
    if "[sP]" in text:
        return "s"
    if "[VP]" in text:
        return "VP"
    if "[L96]" in text:
        return "L96"
    if "[L10]" in text:
        return "L10"
    if "[2P]" in text:
        return "2p"
    if text.endswith("P"):
        return "p"
    return ""


@lru_cache(maxsize=None)
def describe_helm_monomer(token: str) -> HelmMonomerDescriptor:
    text = str(token).strip()
    if not text:
        return HelmMonomerDescriptor(
            token=text,
            base=None,
            sugar_type=None,
            sugar_mod=None,
            backbone=None,
            category="unsupported",
        )

    base, sugar_type, sugar_mod, _ = parse_helm_component(text)
    backbone = explicit_helm_backbone(text)

    if not base and backbone:
        category = "prefix_attachment"
    elif base in {"Y34", "Y44"}:
        category = "abasic"
    elif base:
        category = "nucleotide"
    else:
        category = "unsupported"

    return HelmMonomerDescriptor(
        token=text,
        base=base or None,
        sugar_type=sugar_type or None,
        sugar_mod=sugar_mod or None,
        backbone=backbone or None,
        category=category,
    )


def helm_monomer_to_chemistry_key(token: str) -> str:
    descriptor = describe_helm_monomer(token)
    return "|".join(
        [
            descriptor.category,
            descriptor.base or "",
            descriptor.sugar_type or "",
            descriptor.sugar_mod or "",
            descriptor.backbone or "",
        ]
    )


def parse_helm_chemistry_key(chemistry_key: str) -> tuple[str, str, str, str, str]:
    parts = str(chemistry_key).split("|", 4)
    if len(parts) != 5:
        raise ValueError(f"Invalid HELM chemistry key: {chemistry_key}")
    return tuple(parts)  # type: ignore[return-value]


def render_helm_chemistry_with_base(base: str, chemistry_key: str) -> str | None:
    category, chemistry_base, sugar_type, sugar_mod, backbone = parse_helm_chemistry_key(chemistry_key)
    if category not in {"nucleotide", "abasic"}:
        return None

    rendered_base = chemistry_base if category == "abasic" else str(base)
    return component_to_helm(
        rendered_base,
        sugar_type or "r",
        sugar_mod,
        backbone,
    )


def project_helm_chemistry_onto_seed_token(seed_token: str, chemistry_key: str) -> str | None:
    seed_descriptor = describe_helm_monomer(seed_token)
    category, chemistry_base, sugar_type, sugar_mod, backbone = parse_helm_chemistry_key(chemistry_key)

    if category == "nucleotide":
        if seed_descriptor.category != "nucleotide" or seed_descriptor.base is None:
            return None
        return render_helm_chemistry_with_base(seed_descriptor.base, chemistry_key)

    if category == "prefix_attachment":
        if seed_descriptor.category != "prefix_attachment":
            return None
        return component_to_helm("", "", "", backbone)

    if category == "abasic":
        if seed_descriptor.category != "abasic":
            return None
        return component_to_helm(
            chemistry_base or (seed_descriptor.base or ""),
            sugar_type or (seed_descriptor.sugar_type or "r"),
            sugar_mod,
            backbone,
        )

    if category == "literal":
        return chemistry_base or None

    return None


def build_helm_from_pair(guide: str, passenger: str) -> str:
    return create_helm_notation(str(guide), str(passenger))


def split_helm_notation(helm_notation: str) -> tuple[str, str]:
    text = str(helm_notation).strip()
    if "$" in text:
        text = text.split("$", 1)[0]

    rna1_match = _RNA1_PATTERN.search(text)
    rna2_match = _RNA2_PATTERN.search(text)
    if rna1_match is None or rna2_match is None:
        raise ValueError("Invalid HELM notation: expected RNA1{...}|RNA2{...}")
    return rna1_match.group(1), rna2_match.group(1)


def tokenize_helm_strand(helm_strand: str) -> list[str]:
    return [token.strip() for token in str(helm_strand).split(".") if token.strip()]


def tokenize_helm_pair(helm_notation: str) -> tuple[list[str], list[str]]:
    guide_strand, passenger_strand = split_helm_notation(helm_notation)
    return tokenize_helm_strand(guide_strand), tokenize_helm_strand(passenger_strand)


def guide_helm_tokens(guide: str, passenger: str) -> list[str]:
    helm = build_helm_from_pair(guide, passenger)
    guide_tokens, _ = tokenize_helm_pair(helm)
    return guide_tokens


def passenger_helm_tokens(guide: str, passenger: str) -> list[str]:
    helm = build_helm_from_pair(guide, passenger)
    _, passenger_tokens = tokenize_helm_pair(helm)
    return passenger_tokens


def detokenize_helm_strand(tokens: Iterable[str]) -> str:
    return ".".join(str(token).strip() for token in tokens if str(token).strip())


def detokenize_helm_pair(guide_tokens: Iterable[str], passenger_tokens: Iterable[str]) -> str:
    return f"RNA1{{{detokenize_helm_strand(guide_tokens)}}}|RNA2{{{detokenize_helm_strand(passenger_tokens)}}}"


def helm_tokens_to_fennec_pair(
    guide_tokens: Iterable[str],
    passenger_tokens: Iterable[str],
) -> tuple[str, str, str]:
    helm = detokenize_helm_pair(guide_tokens, passenger_tokens)
    guide, passenger = helm_to_fennec(helm)
    return helm, guide, passenger


def helm_token_to_alyn_token(token: str) -> str:
    return helm_strand_to_fennec(str(token))


def alyn_token_to_helm_token(token: str) -> str:
    helm_strand = sequence_to_helm_strand(str(token))
    tokens = tokenize_helm_strand(helm_strand)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single HELM token, received: {helm_strand}")
    return tokens[0]


def describe_helm_position(token: str) -> "AlynPositionDescriptor":
    return describe_alyn_position(helm_token_to_alyn_token(token))


def render_helm_position_from_template(template_token: str, target_base: str) -> str | None:
    return render_helm_chemistry_with_base(target_base, helm_monomer_to_chemistry_key(template_token))


def ensure_helm_sequence_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()

    has_legacy_pair = all(column in normalized.columns for column in REQUIRED_ORIGIN_SEQUENCE_COLUMNS)
    if has_legacy_pair:
        normalized[HELM_COLUMN] = [
            build_helm_from_pair(guide, passenger)
            for guide, passenger in zip(normalized["siRNA_list"], normalized["siRNA_passenger_list"])
        ]
    elif HELM_COLUMN not in normalized.columns and HELM_STRING_COLUMN in normalized.columns:
        normalized[HELM_COLUMN] = normalized[HELM_STRING_COLUMN]

    if HELM_COLUMN not in normalized.columns:
        if not has_legacy_pair:
            raise ValueError(
                f"Expected either {REQUIRED_ORIGIN_SEQUENCE_COLUMNS} or '{HELM_COLUMN}' columns"
            )
        normalized[HELM_COLUMN] = [
            build_helm_from_pair(guide, passenger)
            for guide, passenger in zip(normalized["siRNA_list"], normalized["siRNA_passenger_list"])
        ]

    if not has_legacy_pair:
        guides: list[str] = []
        passengers: list[str] = []
        for helm in normalized[HELM_COLUMN]:
            guide, passenger = helm_to_fennec(str(helm))
            guides.append(guide)
            passengers.append(passenger)
        normalized["siRNA_list"] = guides
        normalized["siRNA_passenger_list"] = passengers

    normalized[HELM_STRING_COLUMN] = normalized[HELM_COLUMN]

    return normalized


def _raw_alyn_tokens(sequence: str) -> list[str]:
    return [token for token in str(sequence).strip().split() if token]


def tokenize_alyn_str(sequence: str) -> list[str]:
    raw_tokens = _raw_alyn_tokens(sequence)
    if not raw_tokens:
        return []

    grouped_tokens: list[str] = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token in FENNEC_BACKBONE_TOKENS:
            grouped_tokens.append(token)
            index += 1
            continue

        if index + 1 < len(raw_tokens) and raw_tokens[index + 1] in FENNEC_BACKBONE_TOKENS:
            grouped_tokens.append(f"{token} {raw_tokens[index + 1]}")
            index += 2
            continue

        grouped_tokens.append(token)
        index += 1

    return grouped_tokens


def detokenize_alyn_str(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def align_missing_leading_backbone_to_reference(
    tokens: Iterable[str],
    reference_tokens: Iterable[str],
) -> list[str]:
    aligned_tokens = list(tokens)
    reference_list = list(reference_tokens)

    if len(aligned_tokens) == len(reference_list):
        return aligned_tokens

    if len(aligned_tokens) + 1 != len(reference_list) or not aligned_tokens or not reference_list:
        return aligned_tokens

    reference_first = describe_alyn_position(reference_list[0])
    aligned_first = describe_alyn_position(aligned_tokens[0])
    if reference_first.category != "prefix_attachment":
        return aligned_tokens
    if aligned_first.category != "nucleotide":
        return aligned_tokens

    return ["p", *aligned_tokens]


def promote_seed_tokens_to_pool_reference(
    seed_tokens: Iterable[str],
    pool_token_lists: Iterable[Iterable[str]],
) -> list[str]:
    promoted_tokens = list(seed_tokens)
    for pool_tokens in pool_token_lists:
        aligned_tokens = align_missing_leading_backbone_to_reference(
            promoted_tokens,
            pool_tokens,
        )
        if len(aligned_tokens) > len(promoted_tokens):
            promoted_tokens = aligned_tokens
    return promoted_tokens


def normalize_design_pair_to_reference(
    guide: str,
    passenger: str,
    reference_guide_tokens: Iterable[str],
    reference_passenger_tokens: Iterable[str],
) -> tuple[str, str]:
    aligned_guide_tokens = align_missing_leading_backbone_to_reference(
        tokenize_alyn_str(guide),
        reference_guide_tokens,
    )
    aligned_passenger_tokens = align_missing_leading_backbone_to_reference(
        tokenize_alyn_str(passenger),
        reference_passenger_tokens,
    )
    return detokenize_alyn_str(aligned_guide_tokens), detokenize_alyn_str(aligned_passenger_tokens)


def normalize_origin_seed_and_pool(
    seed_row: pd.Series,
    pool_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    normalized_seed = ensure_helm_sequence_columns(pd.DataFrame([seed_row])).iloc[0].copy()
    normalized_pool = ensure_helm_sequence_columns(pool_df)

    if "seed_pair_key" in normalized_seed.index:
        normalized_seed["seed_pair_key"] = make_pair_key(
            str(normalized_seed["siRNA_list"]),
            str(normalized_seed["siRNA_passenger_list"]),
        )

    return normalized_seed, normalized_pool


@dataclass(frozen=True, slots=True)
class AlynTokenDescriptor:
    token: str
    base: str | None
    sugar_type: str | None
    sugar_mod: str | None
    backbone: str | None
    wrapped: bool
    category: str


@dataclass(frozen=True, slots=True)
class AlynPositionDescriptor:
    token: str
    anchor_token: str | None
    anchor_descriptor: AlynTokenDescriptor | None
    explicit_backbone_token: str | None
    effective_backbone: str | None
    category: str


@lru_cache(maxsize=None)
def describe_alyn_token(token: str) -> AlynTokenDescriptor:
    text = str(token).strip()
    wrapped = text.startswith("(") and text.endswith(")")
    try:
        base, sugar_type, sugar_mod, backbone = categorize_component(text)
    except Exception:
        return AlynTokenDescriptor(
            token=text,
            base=None,
            sugar_type=None,
            sugar_mod=None,
            backbone=None,
            wrapped=wrapped,
            category="unsupported",
        )

    if base in {"Y34", "Y44"}:
        category = "abasic"
    elif base:
        category = "nucleotide"
    elif backbone:
        category = "backbone"
    else:
        category = "unsupported"

    return AlynTokenDescriptor(
        token=text,
        base=base or None,
        sugar_type=sugar_type or None,
        sugar_mod=sugar_mod or None,
        backbone=backbone or None,
        wrapped=wrapped,
        category=category,
    )


@lru_cache(maxsize=None)
def describe_alyn_position(token: str) -> AlynPositionDescriptor:
    parts = _raw_alyn_tokens(token)
    if not parts:
        return AlynPositionDescriptor(
            token="",
            anchor_token=None,
            anchor_descriptor=None,
            explicit_backbone_token=None,
            effective_backbone=None,
            category="unsupported",
        )

    if parts[0] in FENNEC_BACKBONE_TOKENS:
        return AlynPositionDescriptor(
            token=str(token).strip(),
            anchor_token=None,
            anchor_descriptor=None,
            explicit_backbone_token=parts[0],
            effective_backbone=parts[0],
            category="prefix_attachment",
        )

    if len(parts) == 1:
        anchor_descriptor = describe_alyn_token(parts[0])
        if anchor_descriptor.category == "unsupported":
            return AlynPositionDescriptor(
                token=str(token).strip(),
                anchor_token=parts[0],
                anchor_descriptor=anchor_descriptor,
                explicit_backbone_token=None,
                effective_backbone=None,
                category="prefix_attachment",
            )

    if len(parts) > 2:
        return AlynPositionDescriptor(
            token=str(token).strip(),
            anchor_token=parts[0],
            anchor_descriptor=describe_alyn_token(parts[0]),
            explicit_backbone_token=None,
            effective_backbone=None,
            category="unsupported",
        )

    anchor_token = parts[0]
    anchor_descriptor = describe_alyn_token(anchor_token)
    explicit_backbone_token = None
    if len(parts) == 2:
        if parts[1] not in FENNEC_BACKBONE_TOKENS:
            return AlynPositionDescriptor(
                token=str(token).strip(),
                anchor_token=anchor_token,
                anchor_descriptor=anchor_descriptor,
                explicit_backbone_token=None,
                effective_backbone=None,
                category="unsupported",
            )
        explicit_backbone_token = parts[1]

    effective_backbone = explicit_backbone_token or anchor_descriptor.backbone
    return AlynPositionDescriptor(
        token=str(token).strip(),
        anchor_token=anchor_token,
        anchor_descriptor=anchor_descriptor,
        explicit_backbone_token=explicit_backbone_token,
        effective_backbone=effective_backbone,
        category=anchor_descriptor.category,
    )


def render_alyn_token_from_template(template_token: str, target_base: str) -> str | None:
    descriptor = describe_alyn_token(template_token)
    if descriptor.category != "nucleotide" or descriptor.base is None:
        return None

    base = "I" if target_base == "N" else target_base
    rendered: str | None = None

    if descriptor.backbone == "2p":
        rendered = f"{base}2p"
    elif descriptor.sugar_mod == "2OMe":
        if base == "I":
            rendered = "i"
        elif base == "T":
            rendered = "t"
        elif base in {"A", "C", "G", "U"}:
            rendered = base.lower()
    elif descriptor.sugar_type == "d" and not descriptor.sugar_mod:
        rendered = f"d{base}"
    elif descriptor.sugar_type == "gn" and not descriptor.sugar_mod:
        rendered = f"{base}gn"
    elif descriptor.sugar_type == "x" and not descriptor.sugar_mod:
        rendered = f"{base}x"
    elif descriptor.sugar_mod in {"f", "s", "hd", "da", "am"} and descriptor.sugar_type == "r":
        rendered = f"{base}{descriptor.sugar_mod}"
    elif descriptor.sugar_mod in {"+", "-"} and descriptor.sugar_type == "r":
        rendered = f"{descriptor.sugar_mod}{base}"
    elif descriptor.sugar_type == "r" and not descriptor.sugar_mod:
        rendered = base

    if rendered is None:
        return None

    candidate = f"({rendered})" if descriptor.wrapped else rendered
    return candidate if describe_alyn_token(candidate).category == "nucleotide" else None


def render_alyn_position_from_template(template_token: str, target_base: str) -> str | None:
    descriptor = describe_alyn_position(template_token)
    if descriptor.category != "nucleotide" or descriptor.anchor_token is None:
        return None

    rendered_anchor = render_alyn_token_from_template(descriptor.anchor_token, target_base)
    if rendered_anchor is None:
        return None

    candidate_parts = [rendered_anchor]
    if descriptor.explicit_backbone_token not in {None, "P", "p"}:
        candidate_parts.append(descriptor.explicit_backbone_token)
    candidate = detokenize_alyn_str(candidate_parts)
    return candidate if describe_alyn_position(candidate).category == "nucleotide" else None


# Backward-compatible aliases for older notebook and script imports.
_raw_helm_tokens = _raw_alyn_tokens
tokenize_helm = tokenize_helm_strand
detokenize_helm = detokenize_helm_strand
HelmTokenDescriptor = AlynTokenDescriptor
HelmPositionDescriptor = AlynPositionDescriptor
describe_helm_token = describe_alyn_token
render_helm_token_from_template = render_alyn_token_from_template


def canonicalize_sirna_key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().upper().replace("T", "U")
    return re.sub(r"\s+", "", text)


def _tokens_preserve_reference_bases(
    reference_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
) -> bool:
    candidate_list = list(candidate_tokens)
    aligned_reference = align_missing_leading_backbone_to_reference(
        reference_tokens,
        candidate_list,
    )

    if len(aligned_reference) != len(candidate_list):
        return False

    for reference_token, candidate_token in zip(aligned_reference, candidate_list):
        reference_descriptor = describe_alyn_position(reference_token)
        candidate_descriptor = describe_alyn_position(candidate_token)

        if reference_descriptor.category == "nucleotide":
            if candidate_descriptor.category != "nucleotide":
                return False
            reference_base = (
                reference_descriptor.anchor_descriptor.base
                if reference_descriptor.anchor_descriptor is not None
                else None
            )
            candidate_base = (
                candidate_descriptor.anchor_descriptor.base
                if candidate_descriptor.anchor_descriptor is not None
                else None
            )
            if reference_base != candidate_base:
                return False
            continue

        if reference_descriptor.category == "abasic":
            if candidate_descriptor.category != "abasic":
                return False
            continue

        if reference_descriptor.category == "prefix_attachment":
            if candidate_descriptor.category != "prefix_attachment":
                return False
            continue

        if reference_descriptor.token != candidate_descriptor.token:
            return False

    return True


def pair_preserves_reference_bases(
    reference_guide: str,
    reference_passenger: str,
    candidate_guide: str,
    candidate_passenger: str,
) -> bool:
    return _tokens_preserve_reference_bases(
        tokenize_alyn_str(reference_guide),
        tokenize_alyn_str(candidate_guide),
    ) and _tokens_preserve_reference_bases(
        tokenize_alyn_str(reference_passenger),
        tokenize_alyn_str(candidate_passenger),
    )


def lookup_distance_entry(
    distance_payload: dict[str, dict[str, Any]] | None,
    sirna_value: Any,
) -> dict[str, Any] | None:
    if not distance_payload:
        return None

    raw_key = "" if sirna_value is None else str(sirna_value).strip()
    canonical_key = canonicalize_sirna_key(raw_key)
    candidate_keys = [
        raw_key,
        raw_key.upper(),
        raw_key.replace("T", "U"),
        raw_key.replace("U", "T"),
        raw_key.upper().replace("T", "U"),
        raw_key.upper().replace("U", "T"),
        canonical_key,
        canonical_key.replace("U", "T"),
    ]

    for candidate in candidate_keys:
        if candidate in distance_payload:
            return distance_payload[candidate]

    # Last-resort fallback for legacy files with inconsistent whitespace/casing.
    for key, entry in distance_payload.items():
        if canonicalize_sirna_key(key) == canonical_key:
            return entry
    return None


def reachable_count_within_steps(alternative_counts: list[int], max_steps: int) -> int:
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


def require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_condition_metrics(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    numeric_cols = ["siRNA_concentration", "true_y_stdev", "n", "spearman", "p_value"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_distance_payload(json_path: Path) -> dict[str, dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def summarize_distance_json(json_path: Path) -> pd.DataFrame:
    payload = load_distance_payload(json_path)

    rows: list[dict[str, Any]] = []
    for sirna, entry in payload.items():
        dist = np.asarray(entry.get("dist", []), dtype=float)
        list_full = entry.get("list_full", [])
        if dist.size and dist.ndim == 2 and dist.shape[0] == dist.shape[1]:
            upper = dist[np.triu_indices(dist.shape[0], k=1)]
            non_zero = upper[upper > 0]
            positive_dist = np.where(dist > 0, dist, np.nan)
            nearest_neighbor = np.nanmin(positive_dist, axis=1) if dist.shape[0] else np.asarray([])
            mean_distance = float(np.nanmean(upper)) if upper.size else np.nan
            max_distance = float(np.nanmax(upper)) if upper.size else np.nan
            min_nonzero = float(np.nanmin(non_zero)) if non_zero.size else np.nan
            mean_nearest_neighbor = float(np.nanmean(nearest_neighbor)) if nearest_neighbor.size else np.nan
        else:
            mean_distance = np.nan
            max_distance = np.nan
            min_nonzero = np.nan
            mean_nearest_neighbor = np.nan
        rows.append(
            {
                "siRNA": sirna,
                "n_distance_designs": len(list_full),
                "mean_edit_distance": mean_distance,
                "max_edit_distance": max_distance,
                "min_nonzero_edit_distance": min_nonzero,
                "mean_nearest_neighbor_distance": mean_nearest_neighbor,
            }
        )
    return pd.DataFrame(rows)


def add_rank_fraction(frame: pd.DataFrame, source_column: str, output_column: str, ascending: bool = True) -> None:
    values = pd.to_numeric(frame[source_column], errors="coerce")
    if values.notna().sum() == 0:
        frame[output_column] = 0.0
        return
    frame[output_column] = values.rank(method="average", pct=True, ascending=ascending).fillna(0.0)


def filter_condition_subset(
    condition_df: pd.DataFrame,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    subset = condition_df[condition_df["model_name"] == choose_best_model(condition_df, config.model_name, config.min_group_n)].copy()

    if config.cell_line_donor is not None:
        subset = subset[subset["cell_line_donor"] == config.cell_line_donor]
    elif config.exclude_primary_cell_lines:
        subset = subset[~subset["cell_line_donor"].fillna("").str.contains("Primary", case=False, na=False)]

    if config.gene_target_species is not None:
        subset = subset[subset["gene_target_species"] == config.gene_target_species]
    if config.transfection_method is not None:
        subset = subset[subset["Transfection_method"] == config.transfection_method]
    if config.siRNA_concentration is not None:
        subset = subset[np.isclose(subset["siRNA_concentration"], config.siRNA_concentration)]

    return subset


def choose_shared_concentration(subset: pd.DataFrame, config: CandidateSelectionConfig) -> float | None:
    if config.siRNA_concentration is not None or not config.require_shared_concentration:
        return config.siRNA_concentration

    reliable_subset = subset[
        subset["spearman"].notna()
        & (subset["spearman"] >= config.min_group_spearman)
        & (subset["n"] >= config.min_group_n)
    ].copy()
    if reliable_subset.empty:
        return None

    concentration_scores = (
        reliable_subset.groupby("siRNA_concentration", dropna=False)
        .agg(
            unique_sirnas=("siRNA", "nunique"),
            mean_spearman=("spearman", "mean"),
            mean_n=("n", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["unique_sirnas", "mean_spearman", "mean_n", "siRNA_concentration"],
            ascending=[False, False, False, True],
        )
    )
    if concentration_scores.empty:
        return None
    return float(concentration_scores.iloc[0]["siRNA_concentration"])


def prepare_condition_subset(
    condition_df: pd.DataFrame,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    subset = filter_condition_subset(condition_df, config)
    shared_concentration = choose_shared_concentration(subset, config)
    if shared_concentration is not None and config.siRNA_concentration is None and config.require_shared_concentration:
        subset = subset[np.isclose(subset["siRNA_concentration"], shared_concentration)]
    return subset


def summarize_sirna_candidates(
    condition_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    subset = prepare_condition_subset(condition_df, config)

    subset["reliable_condition"] = (
        subset["spearman"].notna()
        & (subset["spearman"] >= config.min_group_spearman)
        & (subset["n"] >= config.min_group_n)
    )
    subset["condition_key"] = (
        subset["Gene"].astype(str)
        + "|"
        + subset["cell_line_donor"].astype(str)
        + "|"
        + subset["gene_target_species"].astype(str)
        + "|"
        + subset["Transfection_method"].astype(str)
        + "|"
        + subset["siRNA_concentration"].astype(str)
    )

    reliable_subset = subset[subset["reliable_condition"]].copy()
    if reliable_subset.empty:
        raise ValueError("No siRNA conditions passed the current reliability filters")

    grouped = reliable_subset.groupby("siRNA", dropna=False)
    summary = grouped.agg(
        representative_gene=("Gene", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        representative_cell_line=("cell_line_donor", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        representative_species=("gene_target_species", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        representative_transfection=("Transfection_method", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        reliable_condition_count=("condition_key", "nunique"),
        median_spearman=("spearman", "median"),
        best_spearman=("spearman", "max"),
        median_n=("n", "median"),
        best_true_y_stdev=("true_y_stdev", "max"),
        median_true_y_stdev=("true_y_stdev", "median"),
        concentration_count=("siRNA_concentration", "nunique"),
    ).reset_index()
    summary = summary.merge(distance_df, on="siRNA", how="left")

    add_rank_fraction(summary, "median_spearman", "spearman_rank", ascending=True)
    add_rank_fraction(summary, "reliable_condition_count", "condition_count_rank", ascending=True)
    add_rank_fraction(summary, "median_n", "support_rank", ascending=True)
    add_rank_fraction(summary, "best_true_y_stdev", "dynamic_range_rank", ascending=True)
    add_rank_fraction(summary, "n_distance_designs", "design_count_rank", ascending=True)
    add_rank_fraction(summary, "mean_edit_distance", "distance_span_rank", ascending=True)
    add_rank_fraction(summary, "mean_nearest_neighbor_distance", "local_jump_rank", ascending=True)

    summary["interesting_score"] = (
        0.28 * summary["spearman_rank"]
        + 0.18 * summary["condition_count_rank"]
        + 0.12 * summary["support_rank"]
        + 0.14 * summary["dynamic_range_rank"]
        + 0.14 * summary["design_count_rank"]
        + 0.09 * summary["distance_span_rank"]
        + 0.05 * summary["local_jump_rank"]
    )
    return summary.sort_values(by=["interesting_score", "median_spearman", "reliable_condition_count"], ascending=[False, False, False]).reset_index(drop=True)


def select_interesting_sirnas(
    sirna_summary: pd.DataFrame,
    top_k: int,
    diversity_penalty: float,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    used_genes: set[str] = set()
    used_contexts: set[tuple[str, str, str]] = set()
    available = sirna_summary.reset_index(drop=True).copy()

    while len(selected_indices) < min(top_k, len(available)):
        if not selected_indices:
            next_idx = int(available["interesting_score"].idxmax())
            selected_indices.append(next_idx)
            row = available.loc[next_idx]
            used_genes.add(str(row["representative_gene"]))
            used_contexts.add(
                (
                    str(row["representative_cell_line"]),
                    str(row["representative_species"]),
                    str(row["representative_transfection"]),
                )
            )
            continue

        scores = available["interesting_score"].copy()
        gene_penalty = available["representative_gene"].astype(str).isin(used_genes).astype(float) * diversity_penalty
        context_penalty = available.apply(
            lambda row: (
                (str(row["representative_cell_line"]), str(row["representative_species"]), str(row["representative_transfection"]))
                in used_contexts
            ),
            axis=1,
        ).astype(float) * diversity_penalty
        scores = scores - gene_penalty - context_penalty
        if selected_indices:
            scores.loc[selected_indices] = -np.inf
        next_idx = int(scores.idxmax())
        if not np.isfinite(scores.loc[next_idx]):
            break
        selected_indices.append(next_idx)
        row = available.loc[next_idx]
        used_genes.add(str(row["representative_gene"]))
        used_contexts.add(
            (
                str(row["representative_cell_line"]),
                str(row["representative_species"]),
                str(row["representative_transfection"]),
            )
        )

    selected = available.loc[selected_indices].copy()
    selected["interesting_rank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def choose_representative_conditions(
    condition_df: pd.DataFrame,
    interesting_sirnas: pd.DataFrame,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    subset = prepare_condition_subset(condition_df, config)

    subset = subset[
        subset["spearman"].notna()
        & (subset["spearman"] >= config.min_group_spearman)
        & (subset["n"] >= config.min_group_n)
    ].copy()
    subset["condition_score"] = (
        subset["spearman"]
        + 0.03 * subset["n"]
        + 0.002 * subset["true_y_stdev"].fillna(0.0)
    )

    selected_rows: list[pd.Series] = []
    for summary_row in interesting_sirnas.itertuples(index=False):
        candidates = subset[subset["siRNA"] == summary_row.siRNA].copy()
        if candidates.empty:
            continue
        best = candidates.sort_values(by=["condition_score", "spearman", "n", "true_y_stdev"], ascending=[False, False, False, False]).iloc[0].copy()
        best["interesting_rank"] = summary_row.interesting_rank
        best["interesting_score"] = summary_row.interesting_score
        best["reliable_condition_count"] = summary_row.reliable_condition_count
        best["n_distance_designs"] = summary_row.n_distance_designs
        best["mean_edit_distance"] = summary_row.mean_edit_distance
        best["max_edit_distance"] = summary_row.max_edit_distance
        best["mean_nearest_neighbor_distance"] = summary_row.mean_nearest_neighbor_distance
        selected_rows.append(best)

    if not selected_rows:
        raise ValueError("Could not resolve representative condition rows for selected siRNAs")
    selected_df = pd.DataFrame(selected_rows)
    selected_df["candidate_rank"] = selected_df["interesting_rank"]
    return selected_df.sort_values(by=["interesting_rank", "condition_score"], ascending=[True, False]).reset_index(drop=True)


def choose_best_model(condition_df: pd.DataFrame, explicit_model_name: str | None, min_group_n: int) -> str:
    if explicit_model_name:
        available = set(condition_df["model_name"].dropna().unique())
        if explicit_model_name not in available:
            raise ValueError(f"Requested model_name '{explicit_model_name}' not present in condition metrics")
        return explicit_model_name

    grouped = (
        condition_df[condition_df["n"] >= min_group_n]
        .groupby("model_name", dropna=True)["spearman"]
        .mean()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        raise ValueError("Could not infer best model_name from condition metrics")
    return str(grouped.index[0])


def select_design_conditions(
    condition_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    config: CandidateSelectionConfig,
) -> pd.DataFrame:
    require_columns(
        condition_df,
        [
            "model_name",
            "siRNA",
            "cell_line_donor",
            "siRNA_concentration",
            "gene_target_species",
            "Transfection_method",
            "Gene",
            "n",
            "spearman",
            "true_y_stdev",
        ],
        "condition metrics",
    )

    sirna_summary = summarize_sirna_candidates(condition_df, distance_df, config)
    interesting_sirnas = select_interesting_sirnas(
        sirna_summary=sirna_summary,
        top_k=config.top_k_conditions,
        diversity_penalty=config.diversity_penalty,
    )
    return choose_representative_conditions(condition_df, interesting_sirnas, config)


def load_train_data(train_data_path: Path) -> pd.DataFrame:
    compression = "gzip" if train_data_path.suffix == ".gz" else None
    frame = pd.read_csv(train_data_path, compression=compression)
    frame.loc[:, "siRNA"] = frame["siRNA"].map(canonicalize_sirna_key)
    require_columns(frame, REQUIRED_TRAIN_COLUMNS, "train data")
    return ensure_helm_sequence_columns(frame)


def load_origin_sequences(origin_sequences_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(origin_sequences_path)
    if HELM_COLUMN not in frame.columns:
        require_columns(frame, REQUIRED_ORIGIN_SEQUENCE_COLUMNS, "origin sequences")
    return ensure_helm_sequence_columns(frame)


def _resolve_custom_origin_seed_row(train_df: pd.DataFrame, origin_row: pd.Series) -> pd.Series:
    guide = str(origin_row["siRNA_list"])
    passenger = str(origin_row["siRNA_passenger_list"])
    exact_matches = train_df[
        train_df["siRNA_list"].astype(str).eq(guide)
        & train_df["siRNA_passenger_list"].astype(str).eq(passenger)
    ].copy()

    if not exact_matches.empty:
        seed_row = exact_matches.sort_values(by="mRNA_remaining_pct", ascending=False, na_position="last").iloc[0].copy()
    else:
        missing_metadata = [
            column for column in CUSTOM_ORIGIN_METADATA_COLUMNS
            if pd.isna(origin_row.get(column))
        ]
        if missing_metadata:
            raise ValueError(
                "Custom origin sequence was not found in train data and is missing required metadata columns: "
                f"{missing_metadata}"
            )

        seed_payload = {column: np.nan for column in REQUIRED_TRAIN_COLUMNS}
        seed_payload.update({column: origin_row.get(column, np.nan) for column in origin_row.index})
        seed_payload["siRNA_list"] = guide
        seed_payload["siRNA_passenger_list"] = passenger
        seed_payload["siRNA"] = canonicalize_sirna_key(origin_row.get("siRNA", guide))
        seed_row = pd.Series(seed_payload)

    for column in origin_row.index:
        value = origin_row.get(column)
        if pd.isna(value):
            continue
        seed_row[column] = value

    if pd.isna(seed_row.get("siRNA")):
        seed_row["siRNA"] = canonicalize_sirna_key(origin_row.get("siRNA", guide))

    return seed_row


def build_custom_origin_seed_table(
    train_df: pd.DataFrame,
    origin_sequences: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    seeds: list[dict[str, Any]] = []
    pools: dict[int, pd.DataFrame] = {}

    for origin_id, (_, origin_row) in enumerate(origin_sequences.iterrows()):
        seed = _resolve_custom_origin_seed_row(train_df, origin_row)
        seed["origin_id"] = origin_id
        seed["candidate_rank"] = int(origin_row.get("candidate_rank", origin_id + 1))
        seed["group_spearman"] = origin_row.get("spearman", np.nan)
        seed["group_n"] = origin_row.get("n", np.nan)
        seed["true_y_stdev"] = origin_row.get("true_y_stdev", np.nan)
        seed["distance_designs"] = origin_row.get("n_distance_designs", np.nan)
        seed["mean_edit_distance"] = origin_row.get("mean_edit_distance", np.nan)
        seed["max_edit_distance"] = origin_row.get("max_edit_distance", np.nan)
        seed["seed_pair_key"] = make_pair_key(seed["siRNA_list"], seed["siRNA_passenger_list"])
        seeds.append(seed.to_dict())
        pools[origin_id] = train_df.copy()

    if not seeds:
        raise ValueError("No origin sequences were provided")

    return pd.DataFrame(seeds), pools


def match_condition_pool(train_df: pd.DataFrame, condition_row: pd.Series) -> pd.DataFrame:
    condition_sirna = canonicalize_sirna_key(condition_row.get("siRNA"))
    train_sirna = train_df["siRNA"].map(canonicalize_sirna_key)
    mask = (
        train_sirna.eq(condition_sirna)
        & train_df["cell_line_donor"].eq(condition_row["cell_line_donor"])
        & train_df["gene_target_species"].eq(condition_row["gene_target_species"])
        & train_df["Transfection_method"].eq(condition_row["Transfection_method"])
        & train_df["gene_target_symbol_name"].eq(condition_row["Gene"])
    )
    mask &= np.isclose(train_df["siRNA_concentration"], condition_row["siRNA_concentration"])
    return train_df[mask].copy()


def build_seed_table(
    train_df: pd.DataFrame,
    ranked_conditions: pd.DataFrame,
    distance_payload: dict[str, dict[str, Any]] | None = None,
    seed_origin: str = "worst",
    centroid_radius: int = 2,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if seed_origin == "worst_predicted" and predict_fn is None:
        raise ValueError("seed_origin='worst_predicted' requires predict_fn")

    seeds: list[dict[str, Any]] = []
    pools: dict[int, pd.DataFrame] = {}

    for _, row in ranked_conditions.iterrows():
        pool = match_condition_pool(train_df, row)
        if pool.empty:
            continue

        origin_id = len(seeds)

        seed_row: pd.Series | None = None
        if seed_origin in {"centroid", "centroid_lowest_neighbor"} and distance_payload is not None:
            entry = lookup_distance_entry(distance_payload, row.get("siRNA"))
            if entry:
                list_full = entry.get("list_full") or []
                dist = np.asarray(entry.get("dist", []), dtype=float)
                if list_full and dist.ndim == 2 and dist.shape[0] == dist.shape[1] and len(list_full) == dist.shape[0]:
                    with np.errstate(invalid="ignore"):
                        row_means = np.nanmean(np.where(dist > 0, dist, np.nan), axis=1)
                    if np.all(np.isnan(row_means)):
                        row_means = np.nanmean(dist, axis=1)
                    centroid_idx = int(np.nanargmin(row_means))
                    centroid_design = list_full[centroid_idx]

                    # ensure combined_design exists for matching
                    if "combined_design" not in pool.columns:
                        pool = pool.copy()
                        pool["combined_design"] = [
                            make_combined_design(r.siRNA_list, r.siRNA_passenger_list)
                            for r in pool.itertuples(index=False)
                        ]
                    # normalized helper: remove whitespace and non-alphanumerics, lowercase
                    def norm(s: str) -> str:
                        if s is None:
                            return ""
                        s2 = "".join(str(s).split()).lower()
                        return re.sub(r"[^a-z0-9]", "", s2)

                    if seed_origin == "centroid":
                        # try exact match on siRNA_list
                        try:
                            mask = pool["siRNA_list"].astype(str).fillna("").str.strip() == str(centroid_design).strip()
                        except Exception:
                            mask = np.zeros(len(pool), dtype=bool)

                        # try normalized fallback
                        if not mask.any():
                            centroid_norm = norm(centroid_design)
                            mask = pool["siRNA_list"].fillna("").map(norm) == centroid_norm

                        # try combined_design match
                        if not mask.any() and "combined_design" in pool.columns:
                            mask = pool["combined_design"].fillna("").map(norm) == centroid_norm

                        if mask.any():
                            seed_row = pool[mask].iloc[0].copy()

                    else:  # centroid_lowest_neighbor
                        centroid_norm = norm(centroid_design)
                        # find all JSON indices within centroid_radius
                        try:
                            near_idx = [int(j) for j in range(dist.shape[0]) if np.isfinite(dist[centroid_idx, j]) and dist[centroid_idx, j] <= float(centroid_radius)]
                        except Exception:
                            near_idx = []

                        matched_rows: list[pd.Series] = []
                        if near_idx:
                            # map combined_designs normalized to pool rows
                            combined_norms = pool.get("combined_design")
                            if combined_norms is None:
                                pool = pool.copy()
                                pool["combined_design"] = [
                                    make_combined_design(r.siRNA_list, r.siRNA_passenger_list)
                                    for r in pool.itertuples(index=False)
                                ]
                                combined_norms = pool["combined_design"]

                            # map normalized design strings to positional indices (0..n-1)
                            combined_vals = list(combined_norms.fillna("").tolist())
                            combined_map = {norm(val): pos for pos, val in enumerate(combined_vals)}
                            siRNA_vals = list(pool["siRNA_list"].fillna("").tolist())
                            siRNA_map = {norm(val): pos for pos, val in enumerate(siRNA_vals)}

                            # token-aware maps: detokenize(tokenize(candidate)) -> positional index
                            token_combined_vals = [detokenize_alyn_str(tokenize_alyn_str(str(v))) for v in combined_vals]
                            token_combined_map = {val.strip(): pos for pos, val in enumerate(token_combined_vals)}
                            token_siRNA_vals = [detokenize_alyn_str(tokenize_alyn_str(str(v))) for v in siRNA_vals]
                            token_siRNA_map = {val.strip(): pos for pos, val in enumerate(token_siRNA_vals)}

                            # try to map centroid explicitly
                            centroid_row_idx = None
                            try:
                                centroid_row_idx = combined_map.get(centroid_norm)
                                if centroid_row_idx is None:
                                    centroid_row_idx = siRNA_map.get(centroid_norm)
                            except Exception:
                                centroid_row_idx = None

                            # attempt token-aware centroid mapping if needed
                            if centroid_row_idx is None:
                                try:
                                    centroid_detok = detokenize_alyn_str(tokenize_alyn_str(str(centroid_design)))
                                    centroid_row_idx = token_combined_map.get(centroid_detok)
                                    if centroid_row_idx is None:
                                        centroid_row_idx = token_siRNA_map.get(centroid_detok)
                                except Exception:
                                    centroid_row_idx = None

                            for j in near_idx:
                                candidate = list_full[j]
                                cand_norm = norm(candidate)
                                row_idx = combined_map.get(cand_norm)
                                if row_idx is None:
                                    row_idx = siRNA_map.get(cand_norm)
                                # fallback to token-aware detokenized mapping
                                if row_idx is None:
                                    try:
                                        cand_detok = detokenize_alyn_str(tokenize_alyn_str(str(candidate)))
                                        row_idx = token_combined_map.get(cand_detok)
                                        if row_idx is None:
                                            row_idx = token_siRNA_map.get(cand_detok)
                                    except Exception:
                                        row_idx = None
                                if row_idx is not None:
                                    try:
                                        matched_rows.append(pool.iloc[int(row_idx)].copy())
                                    except Exception:
                                        continue

                            # include centroid row explicitly if we found a mapping and it's not already present
                            if centroid_row_idx is not None:
                                try:
                                    centroid_row = pool.iloc[int(centroid_row_idx)].copy()
                                    centroid_pair = make_pair_key(centroid_row.siRNA_list, centroid_row.siRNA_passenger_list)
                                    if not any(make_pair_key(r.siRNA_list, r.siRNA_passenger_list) == centroid_pair for r in matched_rows):
                                        matched_rows.append(centroid_row)
                                except Exception:
                                    pass

                        if matched_rows:
                            neighbors_df = pd.DataFrame(matched_rows)
                            # pick the worst performing design among centroid+neighbors (highest mRNA_remaining_pct)
                            neighbors_df = neighbors_df.sort_values(by="mRNA_remaining_pct", ascending=False, na_position="last")
                            seed_row = neighbors_df.iloc[0].copy()

        if seed_row is None and seed_origin == "worst_predicted" and predict_fn is not None:
            unique_pool = (
                pool.sort_values(by="mRNA_remaining_pct", ascending=False)
                .drop_duplicates(subset=["siRNA_list", "siRNA_passenger_list"], keep="first")
                .copy()
            )
            if not unique_pool.empty:
                try:
                    predicted_pool = predict_fn(unique_pool)
                    if "mean_pred" in predicted_pool.columns and not predicted_pool.empty:
                        seed_row = (
                            predicted_pool
                            .sort_values(by="mean_pred", ascending=False, na_position="last")
                            .iloc[0]
                            .copy()
                        )
                except Exception:
                    seed_row = None

        if seed_row is None:
            seed_row = pool.sort_values(by="mRNA_remaining_pct", ascending=False).iloc[0].copy()

        seed = seed_row
        seed["origin_id"] = origin_id
        seed["candidate_rank"] = row["candidate_rank"]
        seed["group_spearman"] = row["spearman"]
        seed["group_n"] = row["n"]
        seed["true_y_stdev"] = row["true_y_stdev"]
        seed["distance_designs"] = row.get("n_distance_designs", np.nan)
        seed["mean_edit_distance"] = row.get("mean_edit_distance", np.nan)
        seed["max_edit_distance"] = row.get("max_edit_distance", np.nan)
        seed["seed_pair_key"] = make_pair_key(seed["siRNA_list"], seed["siRNA_passenger_list"])
        seeds.append(seed.to_dict())
        pools[origin_id] = pool.copy()

    if not seeds:
        raise ValueError("No candidate pools matched the selected condition rows")
    return pd.DataFrame(seeds), pools


def infer_pair_shape(guide: str, passenger: str) -> tuple[int, int] | None:
    guide_tokens = guide_helm_tokens(guide, passenger)
    passenger_tokens = passenger_helm_tokens(guide, passenger)
    if not guide_tokens or not passenger_tokens:
        return None
    return len(guide_tokens), len(passenger_tokens)


def build_design_pool_by_shape(frame: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    frame = ensure_helm_sequence_columns(frame)
    unique_designs = (
        frame.dropna(subset=[HELM_COLUMN])
        .sort_values(by="mRNA_remaining_pct", ascending=False, na_position="last")
        .drop_duplicates(subset=[HELM_COLUMN], keep="first")
        .copy()
    )
    if unique_designs.empty:
        return {}

    pair_shapes = [
        infer_pair_shape(row.siRNA_list, row.siRNA_passenger_list)
        for row in unique_designs.itertuples(index=False)
    ]
    valid_mask = [pair_shape is not None for pair_shape in pair_shapes]
    if not any(valid_mask):
        return {}

    unique_designs = unique_designs.loc[valid_mask].copy()
    unique_designs["pool_shape"] = [pair_shape for pair_shape in pair_shapes if pair_shape is not None]
    return {
        pool_shape: group.drop(columns=["pool_shape"]).reset_index(drop=True).copy()
        for pool_shape, group in unique_designs.groupby("pool_shape", dropna=False)
    }


def make_pair_key(guide: str, passenger: str) -> tuple[str, str]:
    helm = build_helm_from_pair(guide, passenger)
    normalized_guide, normalized_passenger = helm_to_fennec(helm)
    return normalized_guide, normalized_passenger


def make_combined_design(guide: str, passenger: str) -> str:
    guide_tokens = tokenize_alyn_str(guide)
    passenger_tokens = tokenize_alyn_str(passenger)
    return detokenize_alyn_str([*guide_tokens, *passenger_tokens])


def infer_guide_token_length(pool_df: pd.DataFrame) -> int:
    normalized_pool = ensure_helm_sequence_columns(pool_df)
    token_lengths = normalized_pool[HELM_COLUMN].dropna().map(lambda helm: len(tokenize_helm_pair(helm)[0]))
    if token_lengths.empty:
        raise ValueError("Cannot infer guide token length from an empty pool")
    return int(token_lengths.mode().iloc[0])


def split_combined_design(combined_design: str, guide_token_length: int) -> tuple[str, str]:
    text = str(combined_design)
    if "RNA1{" in text and "|RNA2{" in text:
        return helm_to_fennec(text)

    tokens = tokenize_alyn_str(text)
    if guide_token_length <= 0 or guide_token_length >= len(tokens):
        raise ValueError("Combined design does not match the inferred guide token length")
    guide = detokenize_alyn_str(tokens[:guide_token_length])
    passenger = detokenize_alyn_str(tokens[guide_token_length:])
    return guide, passenger


def infer_passenger_token_length(pool_df: pd.DataFrame) -> int:
    normalized_pool = ensure_helm_sequence_columns(pool_df)
    token_lengths = normalized_pool[HELM_COLUMN].dropna().map(lambda helm: len(tokenize_helm_pair(helm)[1]))
    if token_lengths.empty:
        raise ValueError("Cannot infer passenger token length from an empty pool")
    return int(token_lengths.mode().iloc[0])


class ClosedWorldModEnhancerSearch:
    def __init__(
        self,
        train_df: pd.DataFrame,
        selected_conditions: pd.DataFrame | None,
        distance_payload: dict[str, dict[str, Any]] | None,
        model_paths: list[Path],
        beam_config: BeamSearchConfig,
        rna_emb_path: Path,
        precomputed_seed_df: pd.DataFrame | None = None,
        precomputed_pool_by_origin: dict[int, pd.DataFrame] | None = None,
    ) -> None:
        if not model_paths:
            raise ValueError("At least one model path is required")
        if beam_config.open_world_scope not in {"cohort", "cohort-pool", "cohort-pool-all", "global"}:
            raise ValueError(
                "open_world_scope must be one of "
                "'cohort', 'cohort-pool', 'cohort-pool-all' or 'global'"
            )
        if beam_config.prediction_chunk_size is not None and beam_config.prediction_chunk_size < 1:
            raise ValueError("prediction_chunk_size must be >= 1 when provided")

        if (precomputed_seed_df is None) != (precomputed_pool_by_origin is None):
            raise ValueError("precomputed_seed_df and precomputed_pool_by_origin must be provided together")

        self.train_df = train_df.copy()
        self.train_df = ensure_helm_sequence_columns(self.train_df)
        self.selected_conditions = selected_conditions.copy() if selected_conditions is not None else pd.DataFrame()
        self.distance_payload = distance_payload or {}
        self.model_paths = [str(path) for path in model_paths]
        self.beam_config = beam_config
        self.rna_emb_path = str(rna_emb_path)
        params = DesignParameters(
            model_type=beam_config.model_type,
            device=beam_config.device,
            fm_type=beam_config.fm_type,
            slice_length=beam_config.slice_length,
            with_thermo=beam_config.with_thermo,
            only_cls_eos=beam_config.only_cls_eos,
        )
        self.designer = siRNADesigner(params)

        if precomputed_seed_df is not None and precomputed_pool_by_origin is not None:
            self.seed_df = precomputed_seed_df.copy()
            self.pool_by_origin = {
                int(origin_id): pool_df.copy()
                for origin_id, pool_df in precomputed_pool_by_origin.items()
            }
        else:
            self.seed_df, self.pool_by_origin = build_seed_table(
                self.train_df,
                self.selected_conditions,
                distance_payload=self.distance_payload,
                seed_origin=self.beam_config.seed_origin,
                centroid_radius=self.beam_config.centroid_neighbor_radius,
                predict_fn=self._predict_designs,
            )

        normalized_seed_rows: list[pd.Series] = []
        normalized_pools_by_origin: dict[int, pd.DataFrame] = {}
        for _, seed_row in self.seed_df.iterrows():
            origin_id = int(seed_row["origin_id"])
            normalized_seed, normalized_pool = normalize_origin_seed_and_pool(
                seed_row,
                self.pool_by_origin[origin_id],
            )
            normalized_seed_rows.append(normalized_seed)
            normalized_pools_by_origin[origin_id] = normalized_pool

        self.seed_df = pd.DataFrame(normalized_seed_rows).reset_index(drop=True)
        self.pool_by_origin = normalized_pools_by_origin

        self.observed_pairs_by_origin = {
            origin_id: {
                make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
                for row in pool_df.itertuples(index=False)
            }
            for origin_id, pool_df in self.pool_by_origin.items()
        }
        self.seed_pair_by_origin = {
            int(row.origin_id): make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
            for row in self.seed_df.itertuples(index=False)
        }
        self.pool_designs_by_origin: dict[int, pd.DataFrame] = {}
        self.design_lookup_by_origin: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
        self.pool_shape_by_origin: dict[int, tuple[int, int]] = {}
        self.token_choices_by_origin: dict[int, dict[str, list[list[str]]]] = {}
        cohort_designs_by_shape: dict[tuple[int, int], list[pd.DataFrame]] = defaultdict(list)

        for origin_id, pool_df in self.pool_by_origin.items():
            unique_pool = (
                pool_df.sort_values(by="mRNA_remaining_pct", ascending=False)
                .drop_duplicates(subset=["siRNA_list", "siRNA_passenger_list"], keep="first")
                .copy()
            )
            unique_pool["pair_key"] = [
                make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
                for row in unique_pool.itertuples(index=False)
            ]
            unique_pool["combined_design"] = [
                make_combined_design(row.siRNA_list, row.siRNA_passenger_list)
                for row in unique_pool.itertuples(index=False)
            ]
            self.pool_designs_by_origin[origin_id] = unique_pool
            self.design_lookup_by_origin[origin_id] = {
                row.pair_key: row._asdict() for row in unique_pool.itertuples(index=False)
            }
            pool_shape = (
                infer_guide_token_length(unique_pool),
                infer_passenger_token_length(unique_pool),
            )
            self.pool_shape_by_origin[origin_id] = pool_shape
            cohort_designs_by_shape[pool_shape].append(unique_pool)

        cohort_token_pools_by_shape = {
            pool_shape: pd.concat(pools, ignore_index=True)
            for pool_shape, pools in cohort_designs_by_shape.items()
        }
        # Build token choice pools depending on open_world_scope
        # - 'cohort': use each origin's full observed pool (all local shapes),
        #   with position-aware applicability based on the origin seed shape
        # - 'cohort-pool': use a shared pool per alyn_str token shape across selected origins
        # - 'cohort-pool-all': use a single shared pool across all selected origins,
        #   but only apply tokens at positions that exist for the current design
        # - 'global': use tokens sourced from all train-data designs with the same alyn_str token shape
        if self.beam_config.open_world_scope == "global":
            token_pools_by_shape = build_design_pool_by_shape(self.train_df)
            shared_token_choices_by_shape = {
                pool_shape: self._build_token_choices(token_pool)
                for pool_shape, token_pool in token_pools_by_shape.items()
            }
            self.token_choices_by_origin = {
                origin_id: shared_token_choices_by_shape[self.pool_shape_by_origin[origin_id]]
                for origin_id in self.pool_by_origin
            }
        elif self.beam_config.open_world_scope == "cohort-pool-all":
            all_selected_pool = pd.concat(list(self.pool_designs_by_origin.values()), ignore_index=True)
            self.token_choices_by_origin = {}
            for origin_id, origin_pool in self.pool_designs_by_origin.items():
                origin_choices = self._build_token_choices(origin_pool)
                seed_guide, seed_passenger = self.seed_pair_by_origin[origin_id]
                pooled_choices = self._build_token_choices_variable_lengths(
                    all_selected_pool,
                    guide_length=len(origin_choices["guide"]),
                    passenger_length=len(origin_choices["passenger"]),
                    seed_guide_tokens=guide_helm_tokens(seed_guide, seed_passenger),
                    seed_passenger_tokens=passenger_helm_tokens(seed_guide, seed_passenger),
                )
                self.token_choices_by_origin[origin_id] = self._merge_token_choices(
                    origin_choices,
                    pooled_choices,
                )
        elif self.beam_config.open_world_scope == "cohort-pool":
            # use a shared token pool made by concatenating all origin pools for each shape
            shared_token_choices_by_shape = {
                pool_shape: self._build_token_choices(cohort_pool)
                for pool_shape, cohort_pool in cohort_token_pools_by_shape.items()
            }
            self.token_choices_by_origin = {
                origin_id: shared_token_choices_by_shape[self.pool_shape_by_origin[origin_id]]
                for origin_id in self.pool_by_origin
            }
        else:  # 'cohort' - per-origin, position-aware across all local shapes
            self.token_choices_by_origin = {}
            for origin_id, unique_pool in self.pool_designs_by_origin.items():
                seed_guide, seed_passenger = self.seed_pair_by_origin[origin_id]
                seed_guide_tokens = guide_helm_tokens(seed_guide, seed_passenger)
                seed_passenger_tokens = passenger_helm_tokens(seed_guide, seed_passenger)
                choices = self._build_token_choices_variable_lengths(
                    unique_pool,
                    guide_length=len(seed_guide_tokens),
                    passenger_length=len(seed_passenger_tokens),
                    seed_guide_tokens=seed_guide_tokens,
                    seed_passenger_tokens=seed_passenger_tokens,
                )
                # Always keep current seed tokens admissible at each position.
                for position, token in enumerate(seed_guide_tokens):
                    chemistry_key = helm_monomer_to_chemistry_key(token)
                    if chemistry_key not in choices["guide"][position]:
                        choices["guide"][position].append(chemistry_key)
                        choices["guide"][position] = sorted(set(choices["guide"][position]))
                for position, token in enumerate(seed_passenger_tokens):
                    chemistry_key = helm_monomer_to_chemistry_key(token)
                    if chemistry_key not in choices["passenger"][position]:
                        choices["passenger"][position].append(chemistry_key)
                        choices["passenger"][position] = sorted(set(choices["passenger"][position]))
                self.token_choices_by_origin[origin_id] = choices

        self.reachable_space_est_by_origin: dict[int, int] = {}
        self.effective_beam_width_by_origin: dict[int, int] = {}
        for origin_id, token_choices in self.token_choices_by_origin.items():
            seed_guide, seed_passenger = self.seed_pair_by_origin[origin_id]
            seed_guide_tokens = guide_helm_tokens(seed_guide, seed_passenger)
            seed_passenger_tokens = passenger_helm_tokens(seed_guide, seed_passenger)

            # Normalize per-position token choices to match the concrete seed token lengths.
            # This avoids shape mismatches when pools contain mixed tokenization styles.
            if len(token_choices["guide"]) != len(seed_guide_tokens):
                normalized_guide_choices: list[list[str]] = []
                for position, current_token in enumerate(seed_guide_tokens):
                    if position < len(token_choices["guide"]):
                        choices = list(token_choices["guide"][position])
                    else:
                        choices = []
                    current_chemistry = helm_monomer_to_chemistry_key(current_token)
                    if current_chemistry not in choices:
                        choices.append(current_chemistry)
                    normalized_guide_choices.append(sorted(set(choices)))
                token_choices["guide"] = normalized_guide_choices

            if len(token_choices["passenger"]) != len(seed_passenger_tokens):
                normalized_passenger_choices: list[list[str]] = []
                for position, current_token in enumerate(seed_passenger_tokens):
                    if position < len(token_choices["passenger"]):
                        choices = list(token_choices["passenger"][position])
                    else:
                        choices = []
                    current_chemistry = helm_monomer_to_chemistry_key(current_token)
                    if current_chemistry not in choices:
                        choices.append(current_chemistry)
                    normalized_passenger_choices.append(sorted(set(choices)))
                token_choices["passenger"] = normalized_passenger_choices

            token_choices = self._lock_token_choices_to_seed(int(origin_id), token_choices)
            self.token_choices_by_origin[int(origin_id)] = token_choices

            guide_alternatives = []
            for position, choices in enumerate(token_choices["guide"]):
                current_token = seed_guide_tokens[position]
                guide_alternatives.append(sum(token != current_token for token in choices))

            passenger_alternatives = []
            for position, choices in enumerate(token_choices["passenger"]):
                current_token = seed_passenger_tokens[position]
                passenger_alternatives.append(sum(token != current_token for token in choices))

            alternative_counts = guide_alternatives + passenger_alternatives
            reachable_space_est = reachable_count_within_steps(
                alternative_counts,
                self.beam_config.num_iterations,
            )
            self.reachable_space_est_by_origin[origin_id] = reachable_space_est

            effective_beam_width = int(self.beam_config.beam_width)
            if self.beam_config.adaptive_beam_cap:
                reachable_non_seed = max(reachable_space_est - 1, 0)
                if reachable_non_seed > 0:
                    effective_beam_width = min(effective_beam_width, reachable_non_seed)

            origin_overrides = self.beam_config.origin_beam_width_overrides or {}
            if int(origin_id) in origin_overrides:
                override_beam = max(1, int(origin_overrides[int(origin_id)]))
                if self.beam_config.adaptive_beam_cap:
                    reachable_non_seed = max(reachable_space_est - 1, 0)
                    if reachable_non_seed > 0:
                        override_beam = min(override_beam, reachable_non_seed)
                effective_beam_width = override_beam

            self.effective_beam_width_by_origin[origin_id] = max(1, effective_beam_width)

            if self.effective_beam_width_by_origin[origin_id] != self.beam_config.beam_width:
                print(
                    f"Origin {origin_id}: capping beam width "
                    f"{self.beam_config.beam_width} -> {self.effective_beam_width_by_origin[origin_id]} "
                    f"(reachable<=iter={reachable_space_est})"
                )

        # print pool size for each origin total across all positions:
        for origin_id, token_choices in self.token_choices_by_origin.items():
            total_tokens = sum(len(choices) for choices in token_choices["guide"]) + sum(len(choices) for choices in token_choices["passenger"])
            print(f"Origin {origin_id} has a total of {total_tokens} unique tokens across guide and passenger positions")

    def _lock_position_choices_to_seed(self, seed_token: str, pooled_tokens: Iterable[str]) -> list[str]:
        locked_tokens = {seed_token}

        for pooled_token in pooled_tokens:
            remapped = project_helm_chemistry_onto_seed_token(seed_token, str(pooled_token))
            if remapped is not None:
                locked_tokens.add(remapped)

        return sorted(locked_tokens)

    def _lock_token_choices_to_seed(
        self,
        origin_id: int,
        token_choices: dict[str, list[list[str]]],
    ) -> dict[str, list[list[str]]]:
        seed_guide, seed_passenger = self.seed_pair_by_origin[int(origin_id)]
        seed_guide_tokens = guide_helm_tokens(seed_guide, seed_passenger)
        seed_passenger_tokens = passenger_helm_tokens(seed_guide, seed_passenger)

        locked_guide = [
            self._lock_position_choices_to_seed(seed_token, token_choices["guide"][position])
            for position, seed_token in enumerate(seed_guide_tokens)
        ]
        locked_passenger = [
            self._lock_position_choices_to_seed(seed_token, token_choices["passenger"][position])
            for position, seed_token in enumerate(seed_passenger_tokens)
        ]
        return {"guide": locked_guide, "passenger": locked_passenger}

    def _predict_designs(self, design_df: pd.DataFrame) -> pd.DataFrame:
        internal_cols = [
            "beam_score",
            "sort_ascending",
            "match_status",
            "pair_key",
            "combined_design",
            "search_iteration",
            "design_id",
            "parent_design_id",
            "mutation_distance",
            "mutation_rank",
            "mutation_side",
            "mutation_position",
            "mutation_from",
            "mutation_to",
            "mutation_track",
            "mutations_from_seed",
            "mutation_step_size",
        ]
        scored_design_df = ensure_helm_sequence_columns(design_df.copy()).reset_index(drop=True)
        prediction_input = scored_design_df.drop(
            columns=[col for col in internal_cols if col in scored_design_df.columns]
        ).copy()
        prediction_input = prediction_input.drop(
            columns=[column for column in ["siRNA_list", "siRNA_passenger_list"] if column in prediction_input.columns]
        )
        prediction_input = prediction_input.reset_index(drop=True)
        chunk_size = self.beam_config.prediction_chunk_size
        if chunk_size is None or len(prediction_input) <= chunk_size:
            predictions = self.designer.predict_efficacy(prediction_input, self.model_paths, self.rna_emb_path)
        else:
            prediction_chunks: list[pd.DataFrame] = []
            total_rows = len(prediction_input)
            for start_idx in range(0, total_rows, chunk_size):
                stop_idx = min(start_idx + chunk_size, total_rows)
                chunk = prediction_input.iloc[start_idx:stop_idx].reset_index(drop=True)
                print(f"Scoring chunk {start_idx + 1}-{stop_idx} of {total_rows} designs")
                prediction_chunks.append(
                    self.designer.predict_efficacy(chunk, self.model_paths, self.rna_emb_path)
                )
            predictions = pd.concat(prediction_chunks, ignore_index=True)
        scored = pd.concat([scored_design_df, predictions.reset_index(drop=True)], axis=1)
        if not scored.columns.is_unique:
            scored = scored.loc[:, ~scored.columns.duplicated()].copy()
        scored["beam_score"], sort_ascending = self.beam_config.score_predictions(scored)
        scored["sort_ascending"] = sort_ascending
        return scored

    def _build_token_choices(self, unique_pool: pd.DataFrame) -> dict[str, list[list[str]]]:
        unique_pool = ensure_helm_sequence_columns(unique_pool)
        guide_length = infer_guide_token_length(unique_pool)
        passenger_length = infer_passenger_token_length(unique_pool)

        valid_rows: list[tuple[list[str], list[str]]] = []
        for row in unique_pool.itertuples(index=False):
            guide_tokens, passenger_tokens = tokenize_helm_pair(row.HELM)
            if len(guide_tokens) != guide_length or len(passenger_tokens) != passenger_length:
                continue
            valid_rows.append((guide_tokens, passenger_tokens))

        if not valid_rows:
            raise ValueError("Could not build token choices from the matched pool")

        guide_choices: list[list[str]] = []
        for position in range(guide_length):
            tokens = sorted({helm_monomer_to_chemistry_key(guide_tokens[position]) for guide_tokens, _ in valid_rows})
            guide_choices.append(tokens)

        passenger_choices: list[list[str]] = []
        for position in range(passenger_length):
            tokens = sorted(
                {helm_monomer_to_chemistry_key(passenger_tokens[position]) for _, passenger_tokens in valid_rows}
            )
            passenger_choices.append(tokens)

        return {"guide": guide_choices, "passenger": passenger_choices}

    def _build_token_choices_variable_lengths(
        self,
        pool_df: pd.DataFrame,
        guide_length: int,
        passenger_length: int,
        seed_guide_tokens: Iterable[str] | None = None,
        seed_passenger_tokens: Iterable[str] | None = None,
    ) -> dict[str, list[list[str]]]:
        pool_df = ensure_helm_sequence_columns(pool_df)
        guide_choices: list[set[str]] = [set() for _ in range(guide_length)]
        passenger_choices: list[set[str]] = [set() for _ in range(passenger_length)]

        for row in pool_df.itertuples(index=False):
            guide_tokens, passenger_tokens = tokenize_helm_pair(row.HELM)

            for position in range(min(len(guide_tokens), guide_length)):
                guide_choices[position].add(helm_monomer_to_chemistry_key(guide_tokens[position]))

            for position in range(min(len(passenger_tokens), passenger_length)):
                passenger_choices[position].add(helm_monomer_to_chemistry_key(passenger_tokens[position]))

        return {
            "guide": [sorted(tokens) for tokens in guide_choices],
            "passenger": [sorted(tokens) for tokens in passenger_choices],
        }

    def _merge_token_choices(
        self,
        base_choices: dict[str, list[list[str]]],
        pooled_choices: dict[str, list[list[str]]],
    ) -> dict[str, list[list[str]]]:
        merged_guide: list[list[str]] = []
        merged_passenger: list[list[str]] = []

        for index, base_tokens in enumerate(base_choices["guide"]):
            pooled_tokens = pooled_choices["guide"][index] if index < len(pooled_choices["guide"]) else []
            merged_guide.append(sorted(set(base_tokens) | set(pooled_tokens)))

        for index, base_tokens in enumerate(base_choices["passenger"]):
            pooled_tokens = pooled_choices["passenger"][index] if index < len(pooled_choices["passenger"]) else []
            merged_passenger.append(sorted(set(base_tokens) | set(pooled_tokens)))

        return {"guide": merged_guide, "passenger": merged_passenger}

    def _annotate_match_status(self, frame: pd.DataFrame) -> pd.DataFrame:
        statuses = []
        for row in frame.itertuples(index=False):
            pair_key = make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
            if pair_key == self.seed_pair_by_origin[int(row.origin_id)]:
                statuses.append("Origin")
            elif pair_key in self.observed_pairs_by_origin[int(row.origin_id)]:
                statuses.append("RecoveredObserved")
            else:
                statuses.append("Novel")
        annotated = frame.copy()
        annotated["match_status"] = statuses
        return annotated

    def _seed_diff_sides(self, origin_id: int, guide: str, passenger: str) -> set[str]:
        seed_guide, seed_passenger = self.seed_pair_by_origin[int(origin_id)]
        diff_sides: set[str] = set()
        if guide_helm_tokens(seed_guide, seed_passenger) != guide_helm_tokens(guide, passenger):
            diff_sides.add("guide")
        if passenger_helm_tokens(seed_guide, seed_passenger) != passenger_helm_tokens(guide, passenger):
            diff_sides.add("passenger")
        return diff_sides

    def _mutation_track_for_pair(self, origin_id: int, guide: str, passenger: str) -> str | None:
        diff_sides = self._seed_diff_sides(origin_id, guide, passenger)
        if not diff_sides:
            return None
        if len(diff_sides) == 1:
            return next(iter(diff_sides))
        return "mixed"

    def _preserves_seed_base_identity(self, origin_id: int, guide: str, passenger: str) -> bool:
        seed_guide, seed_passenger = self.seed_pair_by_origin[int(origin_id)]
        return pair_preserves_reference_bases(
            seed_guide,
            seed_passenger,
            guide,
            passenger,
        )

    def _mutation_branches_for_row(self, row: Any) -> list[tuple[str, ...]]:
        return [("guide", "passenger")]

    def _mutable_sites_for_branch(
        self,
        guide_tokens: list[str],
        passenger_tokens: list[str],
        token_choices: dict[str, list[list[str]]],
        branch_sides: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        mutable_sites: list[dict[str, Any]] = []

        if "guide" in branch_sides:
            for position, allowed_tokens in enumerate(token_choices["guide"]):
                current_token = guide_tokens[position]
                alternatives = [token for token in allowed_tokens if token != current_token]
                if alternatives:
                    mutable_sites.append(
                        {
                            "side": "guide",
                            "position": position,
                            "current": current_token,
                            "alternatives": alternatives,
                        }
                    )

        if "passenger" in branch_sides:
            for position, allowed_tokens in enumerate(token_choices["passenger"]):
                current_token = passenger_tokens[position]
                alternatives = [token for token in allowed_tokens if token != current_token]
                if alternatives:
                    mutable_sites.append(
                        {
                            "side": "passenger",
                            "position": position,
                            "current": current_token,
                            "alternatives": alternatives,
                        }
                    )

        return mutable_sites

    def _format_mutation_field(self, values: list[Any]) -> Any:
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return "|".join(str(value) for value in values)

    def _make_mutated_row(
        self,
        row: Any,
        mutated_guide: str,
        mutated_passenger: str,
        mutation_side: str,
        mutation_position: Any,
        mutation_from: Any,
        mutation_to: Any,
        mutation_distance: float = 1.0,
        mutation_step_size: int = 1,
    ) -> dict[str, Any]:
        base = row._asdict()
        drop_columns = {
            "index",
            "mean_pred",
            "stdev",
            "beam_score",
            "sort_ascending",
            "match_status",
            "search_iteration",
            "design_id",
            "parent_design_id",
            "mutation_distance",
            "mutation_rank",
            "mutation_side",
            "mutation_position",
            "mutation_from",
            "mutation_to",
        }
        for column in drop_columns:
            base.pop(column, None)

        base["siRNA_list"] = mutated_guide
        base["siRNA_passenger_list"] = mutated_passenger
        base["mRNA_remaining_pct"] = np.nan
        base["origin_id"] = int(row.origin_id)
        base["parent_design_id"] = row.design_id
        base["mutation_rank"] = None
        base["mutation_distance"] = float(mutation_distance)
        base["mutation_side"] = mutation_side
        base["mutation_position"] = mutation_position
        base["mutation_from"] = mutation_from
        base["mutation_to"] = mutation_to
        base["mutation_track"] = self._mutation_track_for_pair(
            int(row.origin_id),
            mutated_guide,
            mutated_passenger,
        )
        base["mutations_from_seed"] = self._pair_edit_distance(
            self.seed_pair_by_origin[int(row.origin_id)],
            make_pair_key(mutated_guide, mutated_passenger),
        )
        base["mutation_step_size"] = int(mutation_step_size)
        return base

    def _enumerate_mutations(self, frontier: pd.DataFrame) -> pd.DataFrame:
        candidate_rows: list[dict[str, Any]] = []
        for row in frontier.itertuples(index=False):
            origin_id = int(row.origin_id)
            token_choices = self.token_choices_by_origin[origin_id]
            guide_tokens = guide_helm_tokens(row.siRNA_list, row.siRNA_passenger_list)
            passenger_tokens = passenger_helm_tokens(row.siRNA_list, row.siRNA_passenger_list)

            if len(guide_tokens) != len(token_choices["guide"]) or len(passenger_tokens) != len(token_choices["passenger"]):
                continue

            rank = 0
            for branch_sides in self._mutation_branches_for_row(row):
                if "guide" in branch_sides:
                    for position, allowed_tokens in enumerate(token_choices["guide"]):
                        current_token = guide_tokens[position]
                        for token in allowed_tokens:
                            if token == current_token:
                                continue
                            mutated_tokens = list(guide_tokens)
                            mutated_tokens[position] = token
                            _, mutated_guide, _ = helm_tokens_to_fennec_pair(
                                mutated_tokens,
                                passenger_tokens,
                            )
                            if not self._preserves_seed_base_identity(origin_id, mutated_guide, row.siRNA_passenger_list):
                                continue
                            rank += 1
                            candidate_rows.append(
                                self._make_mutated_row(
                                    row=row,
                                    mutated_guide=mutated_guide,
                                    mutated_passenger=row.siRNA_passenger_list,
                                    mutation_side="guide",
                                    mutation_position=position,
                                    mutation_from=current_token,
                                    mutation_to=token,
                                )
                            )

                if "passenger" in branch_sides:
                    for position, allowed_tokens in enumerate(token_choices["passenger"]):
                        current_token = passenger_tokens[position]
                        for token in allowed_tokens:
                            if token == current_token:
                                continue
                            mutated_tokens = list(passenger_tokens)
                            mutated_tokens[position] = token
                            _, _, mutated_passenger = helm_tokens_to_fennec_pair(
                                guide_tokens,
                                mutated_tokens,
                            )
                            if not self._preserves_seed_base_identity(origin_id, row.siRNA_list, mutated_passenger):
                                continue
                            rank += 1
                            candidate_rows.append(
                                self._make_mutated_row(
                                    row=row,
                                    mutated_guide=row.siRNA_list,
                                    mutated_passenger=mutated_passenger,
                                    mutation_side="passenger",
                                    mutation_position=position,
                                    mutation_from=current_token,
                                    mutation_to=token,
                                )
                            )

            if rank:
                start_idx = len(candidate_rows) - rank
                for local_rank, idx in enumerate(range(start_idx, len(candidate_rows)), start=1):
                    candidate_rows[idx]["mutation_rank"] = local_rank

        if not candidate_rows:
            return pd.DataFrame(columns=list(frontier.columns) + [
                "mutation_side",
                "mutation_position",
                "mutation_from",
                "mutation_to",
                "parent_design_id",
            ])

        mutated_df = ensure_helm_sequence_columns(pd.DataFrame(candidate_rows))
        mutated_df = mutated_df.drop_duplicates(subset=["origin_id", "siRNA_list", "siRNA_passenger_list"])
        return mutated_df.reset_index(drop=True)

    def _pair_edit_distance(self, left_pair: tuple[str, str], right_pair: tuple[str, str]) -> float:
        left_tokens = guide_helm_tokens(left_pair[0], left_pair[1]) + passenger_helm_tokens(left_pair[0], left_pair[1])
        right_tokens = guide_helm_tokens(right_pair[0], right_pair[1]) + passenger_helm_tokens(right_pair[0], right_pair[1])
        max_len = max(len(left_tokens), len(right_tokens))
        left_pad = left_tokens + [""] * (max_len - len(left_tokens))
        right_pad = right_tokens + [""] * (max_len - len(right_tokens))
        return float(sum(left != right for left, right in zip(left_pad, right_pad)))

    def run(self) -> pd.DataFrame:
        seeds = self.seed_df.copy().reset_index(drop=True)
        seeds["design_id"] = [f"origin-{origin_id}" for origin_id in seeds["origin_id"]]
        seeds["search_iteration"] = 0
        seeds["selected_for_frontier"] = True
        seeds["mutation_side"] = None
        seeds["mutation_position"] = None
        seeds["mutation_from"] = None
        seeds["mutation_to"] = None
        seeds["parent_design_id"] = None
        seeds["mutation_track"] = None
        seeds["mutations_from_seed"] = 0.0
        seeds["mutation_step_size"] = 0

        all_results = self._annotate_match_status(self._predict_designs(seeds))
        sort_ascending = bool(all_results["sort_ascending"].iloc[0])
        frontier = all_results.copy()

        seen_pairs: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for row in all_results.itertuples(index=False):
            seen_pairs[int(row.origin_id)].add(make_pair_key(row.siRNA_list, row.siRNA_passenger_list))

        for iteration in range(1, self.beam_config.num_iterations + 1):
            expanded = self._enumerate_mutations(frontier)
            if expanded.empty:
                break

            keep_mask = []
            for row in expanded.itertuples(index=False):
                pair_key = make_pair_key(row.siRNA_list, row.siRNA_passenger_list)
                keep_mask.append(pair_key not in seen_pairs[int(row.origin_id)])
            expanded = expanded[np.asarray(keep_mask, dtype=bool)].copy()
            if expanded.empty:
                break

            expanded["design_id"] = [f"iter-{iteration}-{idx}" for idx in range(len(expanded))]
            expanded["search_iteration"] = iteration

            scored = self._annotate_match_status(self._predict_designs(expanded))
            scored = scored.sort_values(by=["origin_id", "beam_score"], ascending=[True, sort_ascending])
            scored["selected_for_frontier"] = False

            frontier_chunks: list[pd.DataFrame] = []
            for origin_id, group in scored.groupby("origin_id", sort=False):
                beam_limit = self.effective_beam_width_by_origin.get(int(origin_id), self.beam_config.beam_width)
                frontier_chunks.append(group.head(int(beam_limit)))
            frontier = pd.concat(frontier_chunks, ignore_index=False).copy() if frontier_chunks else scored.iloc[0:0].copy()
            if not frontier.empty:
                scored.loc[frontier.index, "selected_for_frontier"] = True
                frontier = scored.loc[frontier.index].copy()

            all_results = pd.concat([all_results, scored], ignore_index=True)

            for row in frontier.itertuples(index=False):
                seen_pairs[int(row.origin_id)].add(make_pair_key(row.siRNA_list, row.siRNA_passenger_list))

        all_results = all_results.sort_values(by=["origin_id", "search_iteration", "beam_score"], ascending=[True, True, sort_ascending]).reset_index(drop=True)
        all_results["open_world_scope"] = self.beam_config.open_world_scope
        all_results["effective_beam_width"] = all_results["origin_id"].map(self.effective_beam_width_by_origin)
        return all_results


def extract_final_iteration_pool(results: pd.DataFrame, final_pool_size: int | None = None) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    frontier_results = results
    if "selected_for_frontier" in frontier_results.columns:
        frontier_results = frontier_results[frontier_results["selected_for_frontier"].fillna(False)].copy()
        if frontier_results.empty:
            frontier_results = results.copy()

    latest_iteration_by_origin = frontier_results.groupby("origin_id")["search_iteration"].transform("max")
    final_pool = frontier_results[frontier_results["search_iteration"] == latest_iteration_by_origin].copy()

    sort_ascending = bool(final_pool["sort_ascending"].iloc[0]) if "sort_ascending" in final_pool.columns else False
    sort_columns = ["beam_score", "origin_id", "design_id"] if "beam_score" in final_pool.columns else ["origin_id", "design_id"]
    sort_orders = [sort_ascending, True, True] if "beam_score" in final_pool.columns else [True, True]
    final_pool = final_pool.sort_values(by=sort_columns, ascending=sort_orders).reset_index(drop=True)

    if final_pool_size is not None:
        if final_pool_size < 1:
            raise ValueError("final_pool_size must be >= 1 when provided")
        final_pool = final_pool.head(int(final_pool_size)).reset_index(drop=True)

    return final_pool


def resolve_model_paths(model_dir: Path | None, model_paths: list[str] | None) -> list[Path]:
    resolved: list[Path] = []
    if model_dir is not None:
        resolved.extend(sorted(path for path in model_dir.glob("*.pt") if path.is_file()))
    if model_paths:
        resolved.extend(Path(path) for path in model_paths)

    deduped: list[Path] = []
    seen = set()
    for path in resolved:
        real = path.resolve()
        if real in seen:
            continue
        if not real.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        seen.add(real)
        deduped.append(real)
    return deduped


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-world Mod_enhancer beam search with cohort, pooled, or global token pools")
    parser.add_argument("--train-data", required=True, help="Path to train_data CSV/CSV.GZ used to rebuild candidate pools")
    parser.add_argument(
        "--origin-sequences",
        help=(
            "Optional CSV of explicit origin sequences. Requires siRNA_list and siRNA_passenger_list; "
            "if a sequence is absent from train-data, also provide the metadata columns needed for prediction"
        ),
    )
    parser.add_argument("--rna-emb-path", default=".", help="RNA embedding directory passed to FENNEC preprocessing")
    parser.add_argument("--condition-metrics", default=str(DEFAULT_CONDITION_METRICS), help="Condition Spearman CSV from ablation/vizzes")
    parser.add_argument("--distance-json", default=str(DEFAULT_DISTANCE_JSON), help="Per-siRNA token distance JSON from ablation/vizzes")
    parser.add_argument("--model-dir", help="Directory containing .pt model files for prediction")
    parser.add_argument("--model-path", action="append", help="Specific .pt model path. Repeat as needed")
    parser.add_argument("--model-name", help="Optional explicit model_name from condition metrics")
    parser.add_argument("--min-group-spearman", type=float, default=0.5)
    parser.add_argument("--min-group-n", type=int, default=4)
    parser.add_argument("--top-k-conditions", type=int, default=9, help="Number of candidate conditions to select for beam search seeding")
    parser.add_argument("--diversity-penalty", type=float, default=0.2, help="Penalty applied when selected siRNAs reuse the same gene or context")
    parser.add_argument("--cell-line-donor")
    parser.add_argument("--gene-target-species")
    parser.add_argument("--transfection-method")
    parser.add_argument("--sirna-concentration", type=float)
    parser.add_argument("--allow-primary-cell-lines", action="store_true", help="Allow condition selection from Primary cell lines")
    parser.add_argument("--allow-mixed-concentrations", action="store_true", help="Allow selected conditions to span multiple concentrations")
    parser.add_argument("--beam-width", type=int, default=30)
    parser.add_argument(
        "--origin-beam-width-overrides",
        default="",
        help="Comma-separated origin beam overrides, e.g. '1:3,2:3'",
    )
    parser.add_argument(
        "--disable-adaptive-beam-cap",
        action="store_true",
        help="Use fixed beam width for every origin even if reachable space is tiny",
    )
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--confidence-multiplier", type=float, default=1.96)
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        help=(
            "Optional maximum number of designs to score in one prediction batch. "
            "Use this to bound memory for large expanded iterations."
        ),
    )
    parser.add_argument("--top-k-neighbors", type=int, default=12, help="Max observed JSON neighbors to expand per design")
    parser.add_argument("--max-neighbor-distance", type=float, help="Optional maximum token edit distance for JSON graph expansion")
    parser.add_argument("--model-type", default="TCN_FM_multi_head", help="FENNEC model registry key used to instantiate checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fm-type", default="orthrus_4track")
    parser.add_argument("--slice-length", type=int, default=100)
    parser.add_argument("--with-thermo", action="store_true", help="Enable thermo features during preprocessing")
    parser.add_argument("--only-cls-eos", action="store_true", help="Restrict embeddings to CLS/EOS tokens when supported")
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
        "--seed-origin",
        choices=["worst", "worst_predicted", "centroid", "centroid_lowest_neighbor"],
        default="worst",
        help="Select seed origin per condition: worst, worst_predicted, centroid, or centroid_lowest_neighbor",
    )
    parser.add_argument(
        "--centroid-neighbor-radius",
        type=int,
        default=2,
        help="Edit distance radius around centroid for neighbor selection (used with centroid_lowest_neighbor)",
    )
    parser.add_argument("--output-candidates", help="Optional CSV path for selected candidate conditions")
    parser.add_argument("--output-results", help="Optional CSV path for beam search results")
    parser.add_argument(
        "--output-final-pool",
        help="Optional CSV path for the final frontier pool after combining the last available iteration from each origin",
    )
    parser.add_argument(
        "--final-pool-size",
        type=int,
        help="Optional cap on the exported final frontier pool, e.g. 100",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origin_beam_overrides: dict[int, int] = {}
    raw_origin_overrides = str(getattr(args, "origin_beam_width_overrides", "") or "").strip()
    if raw_origin_overrides:
        for token in raw_origin_overrides.split(","):
            piece = token.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise ValueError(
                    "Invalid --origin-beam-width-overrides token "
                    f"'{piece}'. Expected format origin:beam, e.g. 1:3"
                )
            origin_str, beam_str = piece.split(":", 1)
            try:
                origin_id = int(origin_str.strip())
                beam_width = int(beam_str.strip())
            except ValueError as exc:
                raise ValueError(
                    "Invalid --origin-beam-width-overrides token "
                    f"'{piece}'. Expected integers like 1:3"
                ) from exc
            if beam_width < 1:
                raise ValueError(
                    "Invalid --origin-beam-width-overrides token "
                    f"'{piece}'. Beam width must be >= 1"
                )
            origin_beam_overrides[origin_id] = beam_width

    beam_config = BeamSearchConfig(
        beam_width=args.beam_width,
        num_iterations=args.num_iterations,
        confidence_multiplier=args.confidence_multiplier,
        top_k_neighbors=args.top_k_neighbors,
        max_neighbor_distance=args.max_neighbor_distance,
        model_type=args.model_type,
        device=args.device,
        fm_type=args.fm_type,
        slice_length=args.slice_length,
        with_thermo=args.with_thermo,
        only_cls_eos=args.only_cls_eos,
        open_world_scope=args.open_world_scope,
        seed_origin=args.seed_origin,
        centroid_neighbor_radius=args.centroid_neighbor_radius,
        adaptive_beam_cap=not args.disable_adaptive_beam_cap,
        origin_beam_width_overrides=origin_beam_overrides,
        prediction_chunk_size=args.prediction_chunk_size,
    )
    train_df = load_train_data(Path(args.train_data))
    model_paths = resolve_model_paths(Path(args.model_dir) if args.model_dir else None, args.model_path)

    precomputed_seed_df: pd.DataFrame | None = None
    precomputed_pool_by_origin: dict[int, pd.DataFrame] | None = None
    distance_payload: dict[str, dict[str, Any]] | None = None

    if args.origin_sequences:
        selected_conditions = load_origin_sequences(Path(args.origin_sequences))
        precomputed_seed_df, precomputed_pool_by_origin = build_custom_origin_seed_table(
            train_df,
            selected_conditions,
        )
    else:
        selection_config = CandidateSelectionConfig(
            min_group_spearman=args.min_group_spearman,
            min_group_n=args.min_group_n,
            top_k_conditions=args.top_k_conditions,
            diversity_penalty=args.diversity_penalty,
            model_name=args.model_name,
            cell_line_donor=args.cell_line_donor,
            gene_target_species=args.gene_target_species,
            transfection_method=args.transfection_method,
            siRNA_concentration=args.sirna_concentration,
            exclude_primary_cell_lines=not args.allow_primary_cell_lines,
            require_shared_concentration=not args.allow_mixed_concentrations,
        )
        condition_df = load_condition_metrics(Path(args.condition_metrics))
        distance_payload = load_distance_payload(Path(args.distance_json))
        distance_df = summarize_distance_json(Path(args.distance_json))
        selected_conditions = select_design_conditions(condition_df, distance_df, selection_config)

    search = ClosedWorldModEnhancerSearch(
        train_df=train_df,
        selected_conditions=selected_conditions,
        distance_payload=distance_payload,
        model_paths=model_paths,
        beam_config=beam_config,
        rna_emb_path=Path(args.rna_emb_path),
        precomputed_seed_df=precomputed_seed_df,
        precomputed_pool_by_origin=precomputed_pool_by_origin,
    )
    results = search.run()
    final_pool = extract_final_iteration_pool(results, args.final_pool_size)

    if args.output_candidates:
        selected_conditions.to_csv(args.output_candidates, index=False)
    if args.output_results:
        results.to_csv(args.output_results, index=False)
    if args.output_final_pool:
        final_pool.to_csv(args.output_final_pool, index=False)

    return selected_conditions, results, final_pool


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    selected_conditions, results, final_pool = run_from_args(args)

    if args.origin_sequences:
        print("Loaded custom origin sequences:")
        display_columns = [
            column for column in ["candidate_rank", "siRNA_list", "siRNA_passenger_list", "cell_line_donor", "gene_target_symbol_name"]
            if column in selected_conditions.columns
        ]
        print(selected_conditions[display_columns].to_string(index=False))
    else:
        print("Selected candidate conditions:")
        print(
            selected_conditions[
                [
                    "candidate_rank",
                    "siRNA",
                    "Gene",
                    "cell_line_donor",
                    "siRNA_concentration",
                    "Transfection_method",
                    "spearman",
                    "n",
                    "mean_edit_distance",
                ]
            ].to_string(index=False)
        )
    print()
    print(f"Open-world scope: {args.open_world_scope}")
    print()
    print("Beam search summary:")
    summary = (
        results.groupby(["origin_id", "search_iteration", "match_status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()
    print(f"Final pool size: {len(final_pool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())