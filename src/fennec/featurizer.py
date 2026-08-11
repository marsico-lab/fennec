import pandas as pd
import numpy as np
from Bio.SeqUtils import gc_fraction

import json
import os
import logging
from functools import lru_cache
from typing import Tuple
from collections import Counter

from .mod_seq_utils import tokenize_seq_to_list

logger = logging.getLogger()

# ALL_VOCAB for token-based embeddings NOT USED for any models except maybe 1!
ALL_VOCAB = {'PAD': 0, 'UNK': 1, 'A': 2, 'C': 3, 'G': 4, 'U': 5, 'N': 6, 'T': 7, 's': 8, 'P': 9, 'VP': 10, 'L96': 11, 'Ahd': 12, 'A2p': 13,
              'Agn': 14, 'Ax': 15, 'Af': 16, 'As': 17, 'Chd': 18, 'C2p': 19, 'Cgn': 20, 'Cx': 21, 'Cf': 22, 'Cs': 23, 'Ghd': 24, 'G2p': 25,
                'Ggn': 26, 'Gx': 27, 'Gf': 28, 'Gs': 29, 'Uhd': 30, 'U2p': 31, 'Ugn': 32, 'Ux': 33, 'Uf': 34, 'Us': 35, '+A': 36, '+C': 37,
                  '+G': 38, '+U': 39, '-A': 40, '-C': 41, '-G': 42, '-U': 43, 'dA': 44, 'dC': 45, 'dG': 46, 'dT': 47, 'a': 48, 'c': 49, 'g': 50,
                    'u': 51, 't': 52, 
              # Parenthesized versions (for HELM notation)
              '(Agn)': 53, '(A2p)': 54, '(Ahd)': 55, '(Cgn)': 56, '(C2p)': 57, '(Chd)': 58, 
              '(Ggn)': 59, '(G2p)': 60, '(Ghd)': 61, '(Tgn)': 62, '(U2p)': 63, '(Uhd)': 64,
              # Additional tokens
              'p': 65,      # Standalone phosphate
              'S': 66,      # IUPAC ambiguity code: G or C
              # Special/proprietary modifications
              'Ada': 67, 'L10': 68, 'Uda': 69, 'Y34': 70, 'Y44': 71,
              # 2'-OMe inosine (custom notation)
              'i': 72}

try:
    from primer3.bindings import calc_hairpin
    from bioinutils.seqfeatures.utils import (
        calc_homo,
        calc_hetero,
        calc_tm,
        calculate_lc,
        homopolymer_counter,
        calc_3_prime_stability,
    )
    from bioinutils.vienna import rfold
except ImportError:
    # Optional: log or print a warning
    logger.error("Some bioinutils modules not available, skipping related features.")


def calc_hairpin_dg(seq: str):
    """calculate the homodimerization delta G

    :param seq: primer
    :type seq: str
    :return: change in gibbs free energy kcal/mol
    :rtype: int
    """
    dg = (
        calc_hairpin(
            seq, mv_conc=200, dv_conc=50, dntp_conc=50, dna_conc=50, temp_c=37
        ).dg
        / 1000
    )
    return dg


hydro_nuc_paired = {"A": 0.393, "G": 0.342, "U": 0.350, "C": 0.384, "N": 0.367} # N is the average

hydro_nuc_unpaired = {"A": 0.798, "G": 0.827, "U": 0.967, "C": 0.962, "N": 0.888}


base_dir = os.path.dirname(os.path.abspath(__file__))

with open(base_dir + "/support_files/Human_seed_toxicity.json", "r") as f:
    via_json = f.read()
    viability = json.loads(via_json)


def add_cell_viability(sirna):
    sirna = sirna.replace("T", "U")
    sirna_seed = sirna[1:7]
    tox = viability.get(sirna_seed)
    return tox


def calculate_polarity(sequence):
    """
    Calculate the polarity of a nucleotide sequence based on B-factors for paired nucleotides
    from table 5 of Kirillova et al 2011.

    Parameters:
    - sequence (str): The nucleotide sequence.

    Returns:
    - polarity (float): A calculated value representing the polarity based on the sum of B-factors.
    """
    # Initialize the total B-factor sum
    total_b_factor = 0.0

    # Iterate through the sequence and sum the corresponding B-factors from the dictionary
    for nucleotide in sequence:
        if nucleotide in hydro_nuc_paired:
            total_b_factor += hydro_nuc_paired[nucleotide]

    # Calculate polarity as an arbitrary function of the B-factor sum
    polarity = total_b_factor / len(sequence)  # Normalizing by sequence length

    return polarity


def calculate_tm_asymmetry(siRNA, num_positions=4):
    """
    Calculate the asymmetry between the 5' and 3' ends of an siRNA based on Gibbs free energy (ΔG).

    Parameters:
    - siRNA (str): The siRNA sequence.
    - num_positions (int): Number of positions to consider at the 5' and 3' ends (default is 4).

    Returns:
    - asymmetry (float): The difference in ΔG between the 5' and 3' ends.
    """
    if len(siRNA) < num_positions * 2:
        raise ValueError(
            "siRNA sequence is too short for the given number of positions."
        )

    # Calculate TM for the 5' end
    five_prime_end = siRNA[:num_positions]
    dg_five_prime = calc_tm(five_prime_end)

    # Calculate TM for the 3' end
    three_prime_end = siRNA[-num_positions:]
    dg_three_prime = calc_tm(three_prime_end)

    # Calculate asymmetry as the difference between 5' and 3' ΔTM
    asymmetry = dg_five_prime - dg_three_prime

    return asymmetry


def calc_i_weight(i, slen):
    middle = slen / 2 + 1
    scale_factor = 3.5 / (
        middle**2
    )  # Adjust to get the values to around 4 at the ends and 1 in the middle
    diff = middle - i
    v = scale_factor * (diff) ** 2
    v *= diff / (abs(diff) + 1)

    return v


def DG_distribution_bias(seq: str):
    seq = seq.replace("T", "U")
    slen = len(seq)
    dg_score = 0
    for i in range(slen - 1):
        nu = seq[i: i + 2]
        if 'N' in nu:
            continue
        dg = DeltaG[nu]
        modifier = calc_i_weight(i, slen)
        dg_score += dg * modifier

    return dg_score


def apply_internal_features(dataframe):

    assert "siRNA" in dataframe.columns

    # Backported from fix_simple_thermo (a6c2089): use standardize_sirna so
    # inosine (I → G), U → T, and any other non-ACGT characters are handled
    # consistently with apply_internal_features_condensed.
    dataframe["siRNA"] = dataframe["siRNA"].map(standardize_sirna)
    siRNA_series = dataframe["siRNA"]
    dataframe["mRNA"] = dataframe["mRNA"].map(
        lambda x: x.replace("U", "T").replace("I", "G").replace("X", "")
    )

    # Calculate the homodimer delta G
    dataframe["homodimer_dG"] = siRNA_series.map(calc_homo)

    # Calculate the heterodimer delta G
    dataframe["heterodimer_dG"] = dataframe.apply(
        lambda x: calc_hetero(x["siRNA"], x["mRNA"]), axis=1
    )

    # Calculate the seed toxicity
    dataframe["seed_toxicity"] = siRNA_series.map(add_cell_viability)

    # Calculate the melting temperature
    dataframe["tm"] = siRNA_series.map(calc_tm)

    # Calculate the GC content
    dataframe["GC"] = siRNA_series.map(gc_fraction)

    # Position-specific nucleotide features
    dataframe["U_pos_1"] = (siRNA_series.str[0] == "T").astype(int)
    dataframe["G_pos_1"] = (siRNA_series.str[0] == "G").astype(int)
    dataframe["C_pos_1"] = (siRNA_series.str[0] == "C").astype(int)
    dataframe["A_pos_1"] = (siRNA_series.str[0] == "A").astype(int)
    dataframe["U_pos_2"] = (siRNA_series.str[1] == "T").astype(int)
    dataframe["A_pos_2"] = (siRNA_series.str[1] == "A").astype(int)

    dataframe["A_end_pos"] = (siRNA_series.str[-1] == "A").astype(int)
    dataframe["U_end_pos"] = (siRNA_series.str[-1] == "T").astype(int)
    dataframe["G_end_pos"] = (siRNA_series.str[-1] == "G").astype(int)
    dataframe["C_end_pos"] = (siRNA_series.str[-1] == "C").astype(int)

    # Dinucleotide features at position 1
    dataframe["UU_pos_1"] = (siRNA_series.str[0:2] == "TT").astype(int)
    dataframe["AA_pos_1"] = (siRNA_series.str[0:2] == "AA").astype(int)
    dataframe["GG_pos_1"] = (siRNA_series.str[0:2] == "GG").astype(int)
    dataframe["GC_pos_1"] = (siRNA_series.str[0:2] == "GC").astype(int)
    dataframe["CC_pos_1"] = (siRNA_series.str[0:2] == "CC").astype(int)
    dataframe["CG_pos_1"] = (siRNA_series.str[0:2] == "CG").astype(int)

    # Overall nucleotide frequency features
    dataframe["U_all"] = siRNA_series.apply(lambda x: x.count("T") / len(x))
    dataframe["G_all"] = siRNA_series.apply(lambda x: x.count("G") / len(x))

    # Dinucleotide frequency features across the sequence
    dataframe["GG_all"] = siRNA_series.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i: i + 2] == "GG")
        / (len(x) - 1)
    )
    dataframe["UU_all"] = siRNA_series.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i: i + 2] == "TT")
        / (len(x) - 1)
    )
    dataframe["UA_all"] = siRNA_series.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i: i + 2] == "TA")
        / (len(x) - 1)
    )
    dataframe["CC_all"] = siRNA_series.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i: i + 2] == "CC")
        / (len(x) - 1)
    )
    dataframe["GC_all"] = siRNA_series.apply(
        lambda x: sum(1 for i in range(len(x) - 1) if x[i: i + 2] == "GC")
        / (len(x) - 1)
    )

    # Trinucleotide homopolymer run
    dataframe["UUU_count"] = siRNA_series.map(lambda x: x.count("TTT"))
    dataframe["AAA_count"] = siRNA_series.map(lambda x: x.count("AAA"))
    dataframe["GGG_count"] = siRNA_series.map(lambda x: x.count("GGG"))
    dataframe["CCC_count"] = siRNA_series.map(lambda x: x.count("CCC"))

    dataframe["5p_4nt_GC"] = siRNA_series.map(lambda x: gc_fraction(x[:4]))
    dataframe["3p_4nt_GC"] = siRNA_series.map(lambda x: gc_fraction(x[-4:]))
    dataframe["ends_4nt_tm_asymetry"] = siRNA_series.map(calculate_tm_asymmetry)
    dataframe["ends_3nt_tm_asymetry"] = siRNA_series.map(
        lambda x: calculate_tm_asymmetry(x, num_positions=3)
    )
    dataframe["ends_2nt_tm_asymetry"] = siRNA_series.map(
        lambda x: calculate_tm_asymmetry(x, num_positions=2)
    )
    dataframe["dg_dist_bias"] = siRNA_series.map(DG_distribution_bias)

    dataframe["polarity"] = siRNA_series.map(calculate_polarity)
    dataframe["GC_skew"] = siRNA_series.apply(
        lambda x: (x.count("G") - x.count("C")) / (x.count("G") + x.count("C"))
    )

    siRNA_r_dict = rfold(siRNA_series.to_dict())
    dataframe["siRNA_structured_percentage"] = [
        siRNA_r_dict[i].unpaired_seq_perc for i in siRNA_r_dict
    ]
    dataframe["siRNA_vienna_mfe"] = [float(siRNA_r_dict[i].mfe) for i in siRNA_r_dict]

    if "mRNA" in dataframe.columns:
        mRNA_r_dict = rfold(dataframe["mRNA"].to_dict())
        dataframe["mRNA_structured_percentage"] = [
            mRNA_r_dict[i].unpaired_seq_perc for i in mRNA_r_dict
        ]
        dataframe["mRNA_vienna_mfe"] = [float(mRNA_r_dict[i].mfe) for i in mRNA_r_dict]

    # Calculate the linguistic complexity
    dataframe["lc"] = siRNA_series.map(calculate_lc)

    dataframe["hairpin_dg"] = siRNA_series.map(calc_hairpin_dg)

    # Count homopolymers
    dataframe["homopolymers_3long"] = siRNA_series.map(
        lambda x: homopolymer_counter(x, limit=3)
    )

    # Calculate 3' prime stability
    dataframe["3_prime_stability"] = dataframe.apply(
        lambda x: calc_3_prime_stability(x["siRNA"], x["mRNA"]), axis=1
    )

    return dataframe

def standardize_sirna(sirna):
    """Remove everything that is not a standard nucleotide (A, C, G, U/T) from the siRNA sequence for feature calculations.
    
    however, account for irregulars such as inosine -> G and U -> T
    """
    sirna = sirna.replace("U", "T").replace("X", "")
    sirna = sirna.replace("I", "G")  # Inosine treated as G for feature calculations

    filtered_bases = []
    for char in sirna:
        if char in {"A", "C", "G", "T"}:
            filtered_bases.append(char)
        else:
            logger.warning(
                f"Non-standard character '{char}' found in siRNA sequence, ignoring it for feature calculations."
            )

    sirna = ''.join(filtered_bases)

    return sirna

def apply_internal_features_condensed(dataframe):

    assert "siRNA" in dataframe.columns

    dataframe["siRNA"] = dataframe["siRNA"].map(standardize_sirna)
    siRNA_series = dataframe["siRNA"]
    dataframe["mRNA"] = dataframe["mRNA"].map(
        lambda x: x.replace("U", "T").replace("I", "G").replace("X", "")
    )

    # Calculate the homodimer delta G
    dataframe["homodimer_dG"] = siRNA_series.map(calc_homo)

    # Calculate the heterodimer delta G
    dataframe["heterodimer_dG"] = dataframe.apply(
        lambda x: calc_hetero(x["siRNA"], x["mRNA"]), axis=1
    )

    # Calculate the seed toxicity
    dataframe["seed_toxicity"] = siRNA_series.map(add_cell_viability)

    # Calculate the melting temperature
    dataframe["tm"] = siRNA_series.map(calc_tm)

    # Calculate the GC content
    dataframe["GC"] = siRNA_series.map(gc_fraction)

    dataframe["ends_3nt_tm_asymetry"] = siRNA_series.map(
        lambda x: calculate_tm_asymmetry(x, num_positions=3)
    )
    dataframe["ends_2nt_tm_asymetry"] = siRNA_series.map(
        lambda x: calculate_tm_asymmetry(x, num_positions=2)
    )
    dataframe["dg_dist_bias"] = siRNA_series.map(DG_distribution_bias)

    dataframe["polarity"] = siRNA_series.map(calculate_polarity)
    dataframe["GC_skew"] = siRNA_series.apply(
        lambda x: (x.count("G") - x.count("C")) / (x.count("G") + x.count("C") + 0.1)
    )

    if "mRNA" in dataframe.columns:
        mRNA_r_dict = rfold(dataframe["mRNA"].to_dict())
        dataframe["mRNA_structured_percentage"] = [
            mRNA_r_dict[i].unpaired_seq_perc for i in mRNA_r_dict
        ]
        dataframe["mRNA_vienna_mfe"] = [float(mRNA_r_dict[i].mfe) for i in mRNA_r_dict]

    # Calculate the linguistic complexity
    dataframe["lc"] = siRNA_series.map(calculate_lc)

    return dataframe


DeltaG = {
    "AA": -0.93,
    "UU": -0.93,
    "AU": -1.10,
    "UA": -1.33,
    "CU": -2.08,
    "AG": -2.08,
    "CA": -2.11,
    "UG": -2.11,
    "GU": -2.24,
    "AC": -2.24,
    "GA": -2.35,
    "UC": -2.35,
    "CG": -2.36,
    "GG": -3.26,
    "CC": -3.26,
    "GC": -3.42,
    "init": 4.09,
    "endAU": 0.45,
    "sym": 0.43,
}
DeltaH = {
    "AA": -6.82,
    "UU": -6.82,
    "AU": -9.38,
    "UA": -7.69,
    "CU": -10.48,
    "AG": -10.48,
    "CA": -10.44,
    "UG": -10.44,
    "GU": -11.40,
    "AC": -11.40,
    "GA": -12.44,
    "UC": -12.44,
    "CG": -10.64,
    "GG": -13.39,
    "CC": -13.39,
    "GC": -14.88,
    "init": 3.61,
    "endAU": 3.72,
    "sym": 0,
}


def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == "A" or i == "a":
            antiRNA.append("U")
        elif i == "U" or i == "u" or i == "T" or i == "t":
            antiRNA.append("A")
        elif i == "C" or i == "c":
            antiRNA.append("G")
        elif i == "G" or i == "g":
            antiRNA.append("C")
        elif i == "X" or i == "x":
            antiRNA.append("X")
    return "".join(antiRNA[::-1])


def Calculate_DGH(seq):
    DG_all = 0
    DG_all += DeltaG["init"]
    DG_all += (
        (seq[0] + seq[len(seq) - 1]).count("A")
        + (seq[0] + seq[len(seq) - 1]).count("U")
    ) * DeltaG["endAU"]
    DG_all += DeltaG["sym"] if antiRNA(seq).replace("T", "U") == seq else 0
    for i in range(len(seq) - 1):
        if 'N' in seq[i:i+2]:
            continue
        DG_all += DeltaG[seq[i] + seq[i + 1]]
    DH_all = 0
    DH_all += DeltaH["init"]
    DH_all += (
        (seq[0] + seq[len(seq) - 1]).count("A")
        + (seq[0] + seq[len(seq) - 1]).count("U")
    ) * DeltaH["endAU"]
    DH_all += DeltaH["sym"] if antiRNA(seq).replace("T", "U") == seq else 0
    for i in range(len(seq) - 1):
        if 'N' in seq[i:i+2]:
            continue
        DH_all += DeltaH[seq[i] + seq[i + 1]]
    return DG_all, DH_all


def Calculate_end_diff(siRNA):
    count = 0
    _5 = siRNA[:2]  # 5'end
    _3 = siRNA[-2:]  # 3' end
    if _5 in ["AC", "AG", "UC", "UG"]:
        count += 1
    elif _5 in ["GA", "GU", "CA", "CU"]:
        count -= 1
    if _3 in ["AC", "AG", "UC", "UG"]:
        count += 1
    elif _3 in ["GA", "GU", "CA", "CU"]:
        count -= 1
    return float("{:.2f}".format(DeltaG[_5] - DeltaG[_3] + count * 0.45))

FEATURE_MAP = {
    "gene_target_species": [
        "Homo sapiens", 
        "Macaca fascicularis", 
        "Mus musculus", 
        "Firefly",  # Luciferase controls
        "Other_species"
    ],
    "cell_line_donor": [
        # Abbreviated forms (as they appear in dataset)
        "A549",
        "BE(2)C",
        "Cos-7",
        "DU145",
        "HeLa",
        "Hep3B",
        "HepG2",
        "Hepa1-6",
        "Neuro2A",
        "Panc-1",
        # Full names (for backwards compatibility)
        "A549 Cells",
        "BE(2)-C Cells",
        "COS-7 Cells",
        "DU145 Cells",
        "HeLa Cells",
        "Hep3B Cells",
        "HepG2 Cells",
        "Hepa1-6 Cells",
        "Human Trabecular Meshwork Cells",
        "Neuro2a Cells",
        "Panc-1 cells",
        # Hepatocyte abbreviations
        "PHH",  # Primary Human Hepatocytes
        "PMH",  # Primary Mouse Hepatocytes
        "PCH",  # Primary Cynomolgus (Monkey) Hepatocytes
        # Hepatocyte full names
        "Primary Human Hepatocytes",
        "Primary Monkey Hepatocytes",
        "Primary Mouse Hepatocytes",
        # Additional cell lines
        "HEK293 Cells",
        "T24 Cells",
        "H1299 Cells",
        "U87MG",  # Glioblastoma - 2,132 samples
        "AXO-U-251",  # Glioblastoma - 676 samples
        "ARPE-19",  # Retinal pigment epithelium
        "hTERT RPE-1",  # Retinal pigment epithelium
        "SK-MEL-28",  # Melanoma
        "Unknown_cell_line",
    ],
    "siRNA_concentration": 0.0,  # Numeric continuous variable (not one-hot encoded)
    "Transfection_method": [
        "Free uptake", 
        "Lipofectamine",  # Includes Lipofectamine2000 variants
        "RNAiMAX",  # 50,227 samples - most common method
        "Not specified",  # Maps to unknown
        "Unknown_transfection"
    ],
    "Duration_after_transfection_h": 0.0,  # Numeric continuous variable (not one-hot encoded)
}


def generate_feature_vector_encoding(df, feature_map=FEATURE_MAP):
    """
    Generates a concatenated feature vector for each row in the dataframe based on the specified keys.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    keys (dict): A dictionary with keys representing column names and values containing unique categories
                 for categorical columns or a numeric placeholder (e.g., 0).

    Returns:
    pd.DataFrame: DataFrame with each row as a concatenated feature vector.
    """

    # Initialize a list to hold the feature vectors
    feature_vectors = []

    target_cols = [k for k in feature_map.keys() if k in df.columns]
    for k in feature_map.keys():
        if k not in target_cols:
            df[k] = None
            logging.error(f'Missing df feature key, {k}')


    def normalize_categorical_value(col, value, unique_values):
        if col == "Transfection_method":
            if pd.isna(value):
                return "Not specified"

            value_str = str(value)
            if value_str == "Unknown_transfection":
                return "Not specified"
            if value_str not in unique_values:
                return "Not specified"

        return value

    # Loop through each row in the dataframe
    for row in df[feature_map.keys()].values:
        vector = []

        # Process each key and its expected values
        for i, (col, unique_values) in enumerate(feature_map.items()):
            if isinstance(unique_values, list):  # Categorical column
                # One-hot encode the value if it's present in the column, else use zeros
                one_hot = np.zeros(len(unique_values), dtype=int)
                normalized_value = normalize_categorical_value(col, row[i], unique_values)
                if normalized_value in unique_values:
                    one_hot[unique_values.index(normalized_value)] = 1
                vector.extend(one_hot)
            elif unique_values == 0:  # Numeric column
                # Append the numeric value as-is
                vector.append(row[i])

        # Add the row's vector to the list of feature vectors
        feature_vectors.append(vector)

    # Convert the list of feature vectors into a DataFrame
    columns = []
    for key, unique_values in feature_map.items():
        if isinstance(unique_values, list):
            columns.extend(unique_values)
        else:
            columns.append(key)
    feature_df = pd.DataFrame(feature_vectors)
    feature_df.columns = columns

    return feature_df


# One-hot mappings
BASE_MAPPING = {
    "":  [0, 0, 0, 0, 0, 0, 0, 0],  # In case there are multiple phosphate chains.
    "A": [1, 0, 0, 0, 0, 0, 0, 0],
    "C": [0, 1, 0, 0, 0, 0, 0, 0],
    "G": [0, 0, 1, 0, 0, 0, 0, 0],
    "U": [0, 0, 0, 1, 0, 0, 0, 0],
    "T": [0, 0, 0, 1, 1, 0, 0, 0],
    "I": [0, 0, 0, 0, 0, 1, 0, 0],  # Inosine
    "N": [1, 1, 1, 1, 1, 1, 0, 0],  # Represents any nucleotide, including inosine.
    "R": [1, 0, 1, 0, 0, 0, 0, 0],  # A or G (purines).
    "Y": [0, 1, 0, 1, 0, 0, 0, 0],  # C or T/U (pyrimidines).
    "S": [0, 1, 1, 0, 0, 0, 0, 0],  # G or C.
    "W": [1, 0, 0, 1, 0, 0, 0, 0],  # A or T/U.
    "K": [0, 0, 1, 1, 0, 0, 0, 0],  # G or T/U.
    "M": [1, 1, 0, 0, 0, 0, 0, 0],  # A or C.
    "B": [0, 1, 1, 1, 0, 0, 0, 0],  # C or G or T/U.
    "D": [1, 0, 1, 1, 0, 0, 0, 0],  # A or G or T/U.
    "H": [1, 1, 0, 1, 0, 0, 0, 0],  # A or C or T/U.
    "V": [1, 1, 1, 0, 0, 0, 0, 0],  # A or C or G.
    "Y34": [0, 0, 0, 0, 0, 0, 1, 0],  # Abasic spacer (2'-OMe furanose)
    "Y44": [0, 0, 0, 0, 0, 0, 0, 1],  # Inverted abasic cap
}

REVERSE_BASE_MAPPING = {tuple(v): k for k, v in BASE_MAPPING.items()}

SUGAR_TYPE_MAP = {
    "": [0, 0, 0, 0],  # in case there are multiple phosphate chains
    "r": [1, 0, 0, 0],
    "d": [0, 1, 0, 0],
    "gn": [0, 0, 1, 0],  # radically different glycol replacement
    "x": [0, 0, 0, 1] # Hexitol
}

REVERSE_SUGAR_TYPE_MAP = {tuple(v): k for k, v in SUGAR_TYPE_MAP.items()}

# Similar to sugar types
SUGAR_MODS = {
    "": [1, 0, 0, 0, 0, 0, 0, 0, 0],  # no additional mods
    "2OMe": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "f": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "s": [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4-thioribose
    "hd": [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Hexadecyl
    "+": [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Locked Nuc
    "-": [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Unlocked Nuc
    "da": [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Docosyloxy (long chain fatty alcohol)
    "am": [0, 0, 0, 0, 0, 0, 0, 0, 1],  # N-methylacetamide
}

REVERSE_SUGAR_MODS = {tuple(v): k for k, v in SUGAR_MODS.items()}

BACKBONE_MOD_MAPPING = {
    "2p": [1, 0, 0, 0, 0, 0],
    "P": [0, 1, 0, 0, 0, 0],
    "p": [0, 1, 0, 0, 0, 0],  # Normal Backbone, or not specified
    "s": [0, 0, 1, 0, 0, 0],  # phosphorothioate
    "VP": [0, 0, 0, 1, 0, 0], 
    "L96": [0, 0, 0, 0, 1, 0],  # GalNAc conjugate
    "L10": [0, 0, 0, 0, 0, 1],  # Cholesterol conjugate
    "N": [0, 0, 0, 0, 0, 0],

}

REVERSE_BACKBONE_MOD_MAPPING = {tuple(v): k for k, v in BACKBONE_MOD_MAPPING.items()}


# Define fixed mappings.
# Derive token classes instead of hard-coding RNA_BASES / BACKBONE_MODS
RNA_BASES = frozenset(k for k in BASE_MAPPING.keys() if k and k.isalpha())  # includes ambiguity codes and inosine
BACKBONE_MODS = frozenset(k for k in BACKBONE_MOD_MAPPING.keys())  # include default

@lru_cache(maxsize=None)
def categorize_component(component: str) -> Tuple[str, str, str, str]:
    """
    Categorizes the component based on string parsing.
    Returns a tuple (base, sugar_type, sugar_mod, backbone)
    """
    # Clean the input once.
    comp = component.strip("() ")

    if comp == "PAD":
        return "PAD", "PAD", "PAD", "PAD"
    
    # Check fixed cases first.
    if comp in RNA_BASES:
        return comp, "r", "", "p"
    if comp == "T":
        return comp, "d", "", "p"
    if comp in BACKBONE_MODS:
        return "", "", "", comp

    # Use a list of tuples to check patterns.
    # Each tuple is: (condition_function, function_to_return_tuple)

    patterns = [
        # N-methylacetamide modifications from abrv_table
        (lambda c: c in {"m5Cam"}, lambda c: ("C", "r", "am", "p")),
        (lambda c: c in {"m5Cams"}, lambda c: ("C", "r", "am", "s")),
        (lambda c: c.endswith("ams") and c[0] in {"A", "G", "T"}, lambda c: (c[0], "r", "am", "s")),
        (lambda c: c.endswith("am") and c[0] in {"A", "G", "T"}, lambda c: (c[0], "r", "am", "p")),
        (lambda c: c.endswith("da"), lambda c: (c[0], "r", "da", "p")),  # docosyloxy
        (lambda c: c.endswith("hd"), lambda c: (c[0], "r", "hd", "p")),
        (lambda c: c.endswith("2p"), lambda c: (c[0], "r", "", "2p")),
        (lambda c: c.endswith("gn"), lambda c: (c[0], "gn", "", "p")),
        (lambda c: c.endswith("x"), lambda c: (c[0], "x", "", "p")),
        (lambda c: c.endswith("f"), lambda c: (c[0], "r", "f", "p")),
        (lambda c: c.endswith("s"), lambda c: (c[0], "r", "s", "p")),
        (lambda c: c.startswith('+'), lambda c: (c[1], "r", "+", "p")),
        (lambda c: c.startswith('-'), lambda c: (c[1], "r", "-", "p")),
        (lambda c: c.startswith("d"), lambda c: (c[1], "d", "", "p")),
        (lambda c: c in {'Y34', 'Y44'}, lambda c: (c, "r", "", "p")),  # Abasic modifications
        (lambda c: c in {"a", "c", "g", "u"}, lambda c: (c.upper(), "r", "2OMe", "p")),
        (lambda c: c in {"i"}, lambda c: ("I", "r", "2OMe", "p")),
        (lambda c: c in {"t"}, lambda c: (c.upper(), "d", "2OMe", "p")),
    ]

    for condition, result_func in patterns:
        if condition(comp):
            return result_func(comp)

    raise ValueError(f"Unacceptable term {component} used in input sequence, has no relevant encoding")


def seq_list_to_seq(seq_list: str) -> str:
    nucs = []
    for comp in seq_list.split():
        if comp in ['p', 'P']:
            continue
        cate = categorize_component(comp)
        nuc = cate[0]
        nucs.append(nuc)

    seq = ''.join(nucs).strip()

    return seq



def helm_one_hot_encode_old(seq: str, one_hot_len: int = 27, one_hot_seq_len: int = 25) -> list:
    """One-hot encodes the sequence with base, base mod, sugar mod, and backbone mod."""

    base_mapping = BASE_MAPPING
    sugar_type_map = SUGAR_TYPE_MAP
    sugar_mods = SUGAR_MODS
    backbone_mod_mapping = BACKBONE_MOD_MAPPING

    # Tokenize the sequence (split by space)
    seq_list = seq.split()
    encoded_seq = []
    comp_len = len(seq_list) - 1

    if seq_list[0] not in backbone_mod_mapping:
        seq_list = ["P"] + seq_list

    nuc_num = 0
    for ind, component in enumerate(seq_list):
        if ind not in [0] and component in backbone_mod_mapping:
            continue
        base, sugar_type, sugar_mod, backbone = categorize_component(component)
        if ind < comp_len:
            if seq_list[ind+1] in backbone_mod_mapping:
                backbone = seq_list[ind+1]
        encoded_pos = (
            base_mapping[base]
            + sugar_type_map[sugar_type]
            + sugar_mods[sugar_mod]
            + backbone_mod_mapping[backbone]
        )
        diff = one_hot_len - len(encoded_pos)
        nuc_num += 1
        oos = [0] * diff
        encoded_pos.extend(oos)
        encoded_seq.append(encoded_pos)
        if nuc_num >= one_hot_seq_len:
            logger.warning(
                f"Sequence {seq} exceeds or equals one-hot sequence length of {one_hot_seq_len}. "
                "Truncating the sequence at seq len 24"
            )
            break

    encoded_seq = encoded_seq[:one_hot_seq_len]

    for _ in range(one_hot_seq_len - len(encoded_seq)):
        oos = [0] * one_hot_len
        encoded_seq.append(oos)

    return encoded_seq

def helm_one_hot_encode(seq: str, 
                        one_hot_len: int = 27,  # 8 (base) + 4 (sugar_type) + 9 (sugar_mod) + 6 (backbone)
                        one_hot_seq_len: int = 25, 
                        collect_errors: bool = True) -> list:
    """Optimized version using numpy and pre-allocated arrays"""
    
    # Pre-allocate result array
    result = np.zeros((one_hot_seq_len, one_hot_len), dtype=np.int8)
    
    seq_list = seq.split()
    if seq_list[0] not in BACKBONE_MOD_MAPPING:
        seq_list = ["P"] + seq_list
    
    nuc_num = 0
    comp_len = len(seq_list) - 1

    errors = []
    
    for ind, component in enumerate(seq_list):
        if nuc_num >= one_hot_seq_len:
            break

        if ind > 0 and component in BACKBONE_MOD_MAPPING:
            continue

        try:
            base, sugar_type, sugar_mod, backbone = categorize_component(component)
            
            # Check for backbone modification
            if ind < comp_len and seq_list[ind+1] in BACKBONE_MOD_MAPPING:
                backbone = seq_list[ind+1]
            
            # Direct array assignment instead of list concatenation
            pos = 0
            
            # Base encoding
            base_enc = BASE_MAPPING[base]
            result[nuc_num, pos:pos+len(base_enc)] = base_enc
            pos += len(base_enc)
            
            # Sugar type encoding
            sugar_type_enc = SUGAR_TYPE_MAP[sugar_type]
            result[nuc_num, pos:pos+len(sugar_type_enc)] = sugar_type_enc
            pos += len(sugar_type_enc)
            
            # Sugar mod encoding
            sugar_mod_enc = SUGAR_MODS[sugar_mod]
            result[nuc_num, pos:pos+len(sugar_mod_enc)] = sugar_mod_enc
            pos += len(sugar_mod_enc)
            
            # Backbone encoding
            backbone_enc = BACKBONE_MOD_MAPPING[backbone]
            result[nuc_num, pos:pos+len(backbone_enc)] = backbone_enc
            
            nuc_num += 1
            
        except (ValueError, KeyError) as e:
            if collect_errors:
                errors.append((component, str(e)))
            continue

    if collect_errors and errors:
        comp_counts = Counter(c for c, _ in errors)
        summary = ", ".join(f"{c}:{n}" for c, n in comp_counts.most_common())
        logger.warning(f"helm_one_hot_encode skipped {len(errors)} components "
                       f"({len(comp_counts)} unique): {summary}")
    
    return result.tolist()


def helm_one_hot_decode(one_hot_matrix):
    """
    Decodes a one-hot encoded matrix back into a HELM-style string.
    Handles special cases and patterns like lowercase 2'OMe, backbone patterns, etc.

    Args:
        one_hot_matrix: List of lists containing one-hot encoded nucleotide data
        
    Returns:
        str: HELM-formatted nucleotide sequence
    """
    # Get slice lengths from mappings
    base_len = len(next(iter(BASE_MAPPING.values())))
    sugar_type_len = len(next(iter(SUGAR_TYPE_MAP.values())))
    sugar_mod_len = len(next(iter(SUGAR_MODS.values())))
    backbone_len = len(next(iter(BACKBONE_MOD_MAPPING.values())))
    
    # Calculate slice positions
    sugar_type_start = base_len
    sugar_mod_start = sugar_type_start + sugar_type_len
    backbone_start = sugar_mod_start + sugar_mod_len
    
    seq_components = []
    prev_backbone = None

    one_hot_seq_len = len(one_hot_matrix)
    
    for ind, pos in enumerate(one_hot_matrix):
        # Get the components
        base = tuple(pos[0:sugar_type_start])
        sugar_type = tuple(pos[sugar_type_start:sugar_mod_start])
        sugar_mod = tuple(pos[sugar_mod_start:backbone_start])
        backbone = tuple(pos[backbone_start:backbone_start + backbone_len])

        # Skip zero-padded rows emitted by helm_one_hot_encode.
        if all(x == 0 for x in pos):
            continue

        # Get basic components
        base_str = REVERSE_BASE_MAPPING.get(base, "")
        sugar_type_str = REVERSE_SUGAR_TYPE_MAP.get(sugar_type, "")
        sugar_mod_str = REVERSE_SUGAR_MODS.get(sugar_mod, "")
        backbone_str = REVERSE_BACKBONE_MOD_MAPPING.get(backbone, "")

        if not base_str:
            # Ignore empty/default rows. Keep explicit leading non-default backbone tokens.
            if ind == 0 and backbone_str and backbone_str not in {"P", "p", "N"}:
                comp = backbone_str
                seq_components.append(comp)
            continue

        # Handle special cases (following categorize_component patterns)
        comp = ""

        # Handle 2'OMe (lowercase nucleotides)
        if sugar_mod_str == "2OMe":
            comp = base_str.lower()
        else:
            # Handle other cases
            if base_str:
                comp = base_str
            # Handle sugar types and mods
            if sugar_type_str == "d":
                comp = "d" + comp
            elif sugar_type_str == "gn":
                comp += "gn"
            elif sugar_type_str == "x":
                comp += "x"
            # Handle sugar mods
            if sugar_mod_str == "f":
                comp += "f"
            elif sugar_mod_str == "s" and ind != 0:
                comp += "s"
            elif sugar_mod_str == "am":
                comp += "am"
            elif sugar_mod_str == "hd":
                comp += "hd"
            elif sugar_mod_str == "da":
                comp += "da"
            elif sugar_mod_str == "+":
                comp = "+" + comp
            elif sugar_mod_str == "-":
                comp = "-" + comp

        # Handle backbone modifications
        if backbone_str:
            # Only emit explicit backbone tokens for non-default linkers.
            if backbone_str in BACKBONE_MOD_MAPPING and backbone_str not in {"P", "p"}:
                comp += ' ' + backbone_str
        if comp:
            seq_components.append(comp)

    # Clean up the sequence
    # Remove duplicate backbone modifiers
    cleaned_components = []
    prev_comp = None
    for comp in seq_components:
        if comp not in BACKBONE_MOD_MAPPING or comp != prev_comp:
            cleaned_components.append(comp)
        prev_comp = comp

    return " ".join(cleaned_components)


def generate_column_labels(sequence_length=25, one_hot_len=27, feature_map=FEATURE_MAP):
    """
    Generates column labels for the flattened array.

    Parameters:
    - sequence_length (int): Number of positions in the sequence (default is 27).
    - feature_map (dict): Dictionary defining the meta features and their possible values.
                          If None, a default feature_map is used.

    Returns:
    - column_names (list): List of column names corresponding to the flattened array.
    """

    # Define the possible features based on your encoding mappings
    # Base features (8 dims including inosine and abasic Y34/Y44 channels)
    base_features = ['nucA', 'nucC', 'nucG', 'nucU', 'nucT', 'nucI', 'nucY34', 'nucY44']

    # Sugar type features
    sugar_type_features = ['sugar_type_r', 'sugar_type_d', 'sugar_type_gn', 'sugar_type_x']

    # Sugar modification features
    sugar_mod_features = ['no_sugar_mod', 'sugar_mod_2OMe', 'sugar_mod_f', 'sugar_mod_thioribose','sugar_mod_hexadecyl', 'sugar_mod_lock', 'sugar_mod_unlock', 'sugar_mod_docosyloxy', 'sugar_mod_am']

    # Backbone modification features
    backbone_mod_features = ['2p_backbone', 'backbone_mod_P', 'backbone_mod_s', 'backbone_mod_VP', 'backbone_mod_L96', 'backbone_mod_L10']

    # Total features per position before padding
    position_features = base_features + sugar_type_features + sugar_mod_features + backbone_mod_features  # 16 features

    # Number of padding zeros to reach 27 features per position
    padding_zeros = one_hot_len - len(position_features)  #  padding zeros
    padding_features = [f'pad_{i}' for i in range(1, padding_zeros + 1)]  # pad_1 to pad_11

    # Total features per position including padding
    total_position_features = position_features + padding_features  # 27 features

    # Generate column names for each position in the sequence
    column_names = []

    for pos in range(1, sequence_length + 1):
        for feature in total_position_features:
            column_names.append(f'pos_{pos}_{feature}')

    # Generate meta feature column names
    meta_columns = []
    for key, unique_values in feature_map.items():
        if isinstance(unique_values, list):  # Categorical feature
            for value in unique_values:
                # Sanitize the value to create a valid column name
                sanitized_value = value.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                meta_columns.append(f'{key}_{sanitized_value}')
        else:  # Numeric feature
            meta_columns.append(key)

    # Append meta feature column names to the main column names list
    column_names.extend(meta_columns)

    return column_names

