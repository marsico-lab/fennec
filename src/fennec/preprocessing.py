from .fm_utils import (
    run_mrna_through_rna_fm,
    run_mrna_through_rinalmo,
    run_mrna_through_one_hot,
    run_mrna_through_orthrus_4track,
    run_mrna_through_gene_one_hot,
    run_mrna_through_rna_fm_triple,
    run_mrna_through_orthrus_4track_triple,
    save_rna_fm_tokens,
    save_orthrus_one_hot,
    reload_fm_outputs,
)
from .featurizer import (helm_one_hot_encode,
                        generate_feature_vector_encoding,
                        apply_internal_features_condensed,
                        seq_list_to_seq,
                        BASE_MAPPING,
                        SUGAR_TYPE_MAP,
                        SUGAR_MODS,
                        BACKBONE_MOD_MAPPING)

from .helm_converter import helm_to_fennec
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any


ONE_HOT_FEATURE_LEN = (
    len(next(iter(BASE_MAPPING.values())))
    + len(next(iter(SUGAR_TYPE_MAP.values())))
    + len(next(iter(SUGAR_MODS.values())))
    + len(next(iter(BACKBONE_MOD_MAPPING.values())))
)


def prepare_meta_features_and_one_hot(train_data,
                                      with_passenger=True):
    feature_df = generate_feature_vector_encoding(train_data).fillna(0)

    if 'siRNA_list' not in train_data.columns and 'HELM' in train_data.columns:
        guides, passengers, unmod_guides = [], [], []
        for helm in tqdm(train_data['HELM'], desc="Converting HELM to Fennec format"):
            guide, passenger = helm_to_fennec(helm)
            guides.append(guide)
            passengers.append(passenger)
            unmod_guides.append(seq_list_to_seq(guide)) 
        train_data['siRNA_list'] = guides
        train_data['siRNA_passenger_list'] = passengers
        train_data['siRNA'] = unmod_guides

    X_hot = train_data.siRNA_list.map(lambda x: helm_one_hot_encode(x, one_hot_len=ONE_HOT_FEATURE_LEN)).values.tolist()


    X = torch.tensor(X_hot).type(torch.float32)
    if with_passenger:
        X_hot2 = train_data.siRNA_passenger_list.map(lambda x: helm_one_hot_encode(x, one_hot_len=ONE_HOT_FEATURE_LEN)).values.tolist()

        X2 = torch.tensor(X_hot2).type(torch.float32)
        X = torch.cat([X, X2], dim=2)
                    
    X = X.permute(0, 2, 1)
    y = torch.tensor(train_data['mRNA_remaining_pct'].values).type(torch.float32).clamp(0,200)
    X_meta = torch.tensor(feature_df.fillna(0).values)

    return X, y, X_meta

def prepare_meta_features_and_one_hot_chem_ablate(train_data):
    X, y, X_meta = prepare_meta_features_and_one_hot(train_data, with_passenger=False)
    # Keep nucleotide identity channels (A, C, G, U, T, I); ablate chemistry-related channels.
    X[:, 6:, :] = 0

    return X, y, X_meta


def calculate_fm_embeddings(
    data: pd.DataFrame,
    outdir: str,
    device: str = "cuda",
    slice_length: int = 100,
    slice_from: str = 'center',
    fm_type: str = 'rna_fm',
    only_cls_eos: bool = False,
    ):
    """
    Calculate RNA-FM or RiNALMo embeddings for the given data.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing siRNA and mRNA sequences.
    outdir: str
        Directory to save the output files.
    device: str
        Device to run the model on (e.g., 'cuda' or 'cpu').
    slice_length: int
        Length of the slices for processing.
    slice_from: str
        Where to slice mRNA sequence from. One of: 'center' (default),
        'five_prime', 'three_prime', 'five_prime_end', or 'three_prime_start'.
    fm_type: str
        Type of FM to use ('rna_fm' or 'rinalmo' or 'one_hot').
    only_cls_eos: bool
        Whether to only use the CLS and EOS tokens for the embeddings.

    Returns
    -------
    fm_embeddings: torch.Tensor
        The calculated FM embeddings.
    embed_map: pd.DataFrame
        DataFrame mapping siRNA to mRNA.
    """

    # 1. Prepare the FM dictionary (siRNA -> mRNA or UTR column depending on slice_from)
    if slice_from == 'five_prime_end':
        mrna_col = 'mRNA_five_prime'
    elif slice_from == 'three_prime_start':
        mrna_col = 'mRNA_three_prime'
    else:
        mrna_col = 'mRNA'
    fm_dict = data.set_index('siRNA')[mrna_col].fillna('').to_dict()

    if fm_type == 'rna_fm':
        # 2. Run the RNA-FM pipeline (this might be time-consuming)
        run_mrna_through_rna_fm(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    elif fm_type == 'rinalmo':
        # 2. Run the RiNALMo pipeline (this might be time-consuming)
        run_mrna_through_rinalmo(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
            only_cls_eos=only_cls_eos,
        )
    elif fm_type == 'one_hot':
        run_mrna_through_one_hot(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    elif fm_type == 'orthrus_4track':
        run_mrna_through_orthrus_4track(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    elif fm_type == 'gene_one_hot':
        gene_dict = (
            data.drop_duplicates('siRNA')
            .set_index('siRNA')['gene_target_symbol_name']
            .fillna('Unknown')
            .to_dict()
        )
        run_mrna_through_gene_one_hot(
            fm_dict,
            gene_dict=gene_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    # ── Triple-slice: center + five_prime + three_prime_start ───────────
    elif fm_type in ('rna_fm_triple', 'orthrus_4track_triple'):
        _col = lambda col: data.set_index('siRNA')[col].fillna('').to_dict() \
            if col in data.columns else fm_dict
        fm_dict_center = data.set_index('siRNA')['mRNA'].fillna('').to_dict()
        fm_dict_five   = _col('mRNA_five_prime')
        fm_dict_three  = _col('mRNA_three_prime')
        if fm_type == 'rna_fm_triple':
            run_mrna_through_rna_fm_triple(
                fm_dict_center, fm_dict_five, fm_dict_three,
                outdir=outdir, device=device, slice_length=slice_length,
            )
        else:
            run_mrna_through_orthrus_4track_triple(
                fm_dict_center, fm_dict_five, fm_dict_three,
                outdir=outdir, device=device, slice_length=slice_length,
            )
    # ── LoRA / adapter: save raw tokens or one-hot for online FM during training
    elif fm_type == 'rna_fm_lora':
        save_rna_fm_tokens(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    elif fm_type == 'orthrus_4track_lora':
        save_orthrus_one_hot(
            fm_dict,
            outdir=outdir,
            device=device,
            slice_length=slice_length,
            slice_from=slice_from,
        )
    else:
        raise ValueError("Invalid fm_type. Choose one of: 'rna_fm', 'rinalmo', 'orthrus_4track', "
                         "'one_hot', 'gene_one_hot', 'rna_fm_triple', 'orthrus_4track_triple', "
                         "'rna_fm_lora', 'orthrus_4track_lora'.")

    # 3. Load RNA-FM embeddings and the embedding map
    fm_embeddings = reload_fm_outputs(outdir)
    embed_map = pd.read_csv(outdir + "index_siRNA_mRNA.csv", index_col=0)

    return fm_embeddings, embed_map

def expand_embed_map(embed_map: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the embedding map to include all siRNA sequences in the data.
    
    Parameters
    ----------
    embed_map: pd.DataFrame
        DataFrame mapping siRNA to mRNA.
    data: pd.DataFrame
        DataFrame containing siRNA and mRNA sequences.

    Returns
    -------
    train_indices: np.ndarray
        Array of indices for the training data.

    """
    embed_index_map = embed_map.reset_index().set_index('siRNA')['index'].to_dict()

    train_indices = data['siRNA'].map(lambda x: x).map(lambda x: embed_index_map[x]).values

    return train_indices


def preprocess_input_tracks(
    data: pd.DataFrame,
    outdir: str,
    device: str = "cuda",
    slice_length: int = 100,
    slice_from: str = 'center',
    fm_type: str = 'rna_fm',
    only_cls_eos: bool = False,
    with_passenger: bool = True,
    with_chem_ablate: bool = False,
    with_thermo: bool = False,
    with_log1p_mrna_remaining_pct: bool = False) -> Dict[str, Any]:

    """
    Prepare all data for training, including siRNA, mRNA, and meta features.
    This function also calculates RNA-FM or RiNALMo embeddings.
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing siRNA and mRNA sequences.
    outdir: str
        Directory to save the output files.
    device: str
        Device to run the model on (e.g., 'cuda' or 'cpu').
    slice_length: int
        Length of the slices for processing.
    slice_from: str
        Where to slice mRNA sequence from. One of: 'center' (default),
        'five_prime', 'three_prime', 'five_prime_end', or 'three_prime_start'.
    fm_type: str
        Type of FM to use ('rna_fm', 'rinalmo', 'orthrus_4track', or 'one_hot').
    only_cls_eos: bool
        Whether to only use the CLS and EOS tokens for the embeddings.
    with_passenger: bool
        Whether to include passenger siRNA sequences.
    with_chem_ablate: bool
        Whether to include chemical ablation features.
    with_thermo: bool
        Whether to include thermo features.
    Returns
    -------
    X: torch.Tensor
        The prepared siRNA and mRNA sequences.
    y: torch.Tensor
        The target variable (mRNA remaining percentage).
    X_meta: torch.Tensor
        The prepared meta features.
    fm_embeddings: torch.Tensor
        The calculated FM embeddings.
    train_indices: np.ndarray
        Array of indices for the training data.

    """
    # 1. Validate the input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    data = data.copy()

    # Standardize common-name synonyms to canonical scientific names for gene_target_species.
    if 'gene_target_species' in data.columns:
        _species_synonyms = {
            "Human": "Homo sapiens",
            "Cynomolgus monkey": "Macaca fascicularis",
            "Mouse": "Mus musculus",
        }
        data['gene_target_species'] = data['gene_target_species'].replace(_species_synonyms)

    # Always-on transform: concentration is log-normalized after filling missing values with 1.
    if 'siRNA_concentration' in data.columns:
        conc = pd.to_numeric(data['siRNA_concentration'], errors='coerce').fillna(1.0)
    else:
        conc = pd.Series(1.0, index=data.index)
    data['siRNA_concentration'] = np.log(np.clip(conc.astype(float), a_min=1e-8, a_max=None))

    if 'mRNA_remaining_pct' not in data.columns:
        raise ValueError("DataFrame must contain 'mRNA_remaining_pct' column.")

    # Optional transform for target variable.
    if with_log1p_mrna_remaining_pct:
        mrna_pct = pd.to_numeric(data['mRNA_remaining_pct'], errors='coerce').fillna(0.0)
        data['mRNA_remaining_pct'] = np.log1p(np.clip(mrna_pct.astype(float), a_min=0.0, a_max=None))
    if 'siRNA_list' not in data.columns and 'HELM' not in data.columns:
        raise ValueError("DataFrame must contain 'siRNA_list' or 'HELM' column.")
    if with_passenger and 'siRNA_passenger_list' not in data.columns and 'HELM' not in data.columns:
        raise ValueError("DataFrame must contain 'siRNA_passenger_list' column if with_passenger is True.")
    # 2. Prepare the meta features and one-hot encoding
    if with_chem_ablate:
        X, y, X_meta = prepare_meta_features_and_one_hot_chem_ablate(data)
    else:
        X, y, X_meta = prepare_meta_features_and_one_hot(data, with_passenger=with_passenger)
    # 3. Run the RNA-FM or RiNALMo pipeline
    fm_embeddings, embed_map = calculate_fm_embeddings(
        data,
        outdir=outdir,
        device=device,
        only_cls_eos=only_cls_eos,
        slice_length=slice_length,
        slice_from=slice_from,
        fm_type=fm_type
    )
    mrna_map_index = expand_embed_map(embed_map, data)
    # 4. Apply internal features if specified
    if with_thermo:
        features2 = apply_internal_features_condensed(embed_map)
        features2 = features2.iloc[mrna_map_index].drop(columns=['siRNA', 'mRNA']).reset_index(drop=True)
        features2 = torch.tensor(features2.values).type(torch.float32)
        X_meta = torch.cat([X_meta, features2], dim=1)
        
    # 5. Return the prepared data
    return X, y, X_meta, fm_embeddings, mrna_map_index



def build_meta_dataframe_with_thermo(
    data: pd.DataFrame,
    outdir: str,
    device: str = "cuda",
    slice_length: int = 100,
    slice_from: str = 'center',
    fm_type: str = "rna_fm",
    only_cls_eos: bool = False,
) -> pd.DataFrame:
    """
    Construct the meta feature dataframe (X_meta) augmented with thermo features.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe containing siRNA and mRNA information.
    outdir: str
        Output directory used for feature model artifacts.
    device: str, optional
        Compute device for feature model inference.
    slice_length: int, optional
        Slice length forwarded to the feature model runners.
    slice_from: str, optional
        Where to slice mRNA sequence from. One of: 'center' (default),
        'five_prime', 'three_prime', 'five_prime_end', or 'three_prime_start'.
    fm_type: str, optional
        Feature model type ('rna_fm', 'rinalmo', 'orthrus_4track', or 'one_hot').
    only_cls_eos: bool, optional
        Whether to restrict RiNALMo outputs to CLS/EOS tokens.

    Returns
    -------
    pd.DataFrame
        Meta features concatenated with thermo features.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    required_cols = {"siRNA_list", "siRNA", "mRNA"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    _, embed_map = calculate_fm_embeddings(
        data,
        outdir=outdir,
        device=device,
        slice_length=slice_length,
        slice_from=slice_from,
        fm_type=fm_type,
        only_cls_eos=only_cls_eos,
    )

    base_features = generate_feature_vector_encoding(data).fillna(0).reset_index(drop=True)

    mrna_map_index = expand_embed_map(embed_map, data)
    thermo_features = (
        apply_internal_features_condensed(embed_map)
        .iloc[mrna_map_index]
        .drop(columns=[col for col in ["siRNA", "mRNA"] if col in embed_map.columns], errors="ignore")
        .reset_index(drop=True)
        .fillna(0)
    )

    return pd.concat([base_features, thermo_features], axis=1)

