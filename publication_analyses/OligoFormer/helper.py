# Third-party benchmark utilities for OligoFormer comparison.
# OligoFormer: https://github.com/jerryji1993/OligoFormer
# Feature engineering functions adapted from the OligoFormer pipeline
# and LuLab thermodynamic descriptor code.
import pandas as pd

from bioinutils.seqfeatures.utils import (calc_homo, 
                                          calc_hetero,
                                          calc_tm, 
                                          calculate_lc, 
                                          homopolymer_counter,
                                          calc_3_prime_stability,
)
from bioinutils.vienna import rfold
from Bio.SeqUtils import gc_fraction

from primer3.bindings import calc_end_stability, calc_hairpin
import warnings
#warnings.filterwarnings('ignore') # Don't do this


lulab_td_columns=['ends',
 'DG_1',
 'DH_1',
 'U_1',
 'G_1',
 'DH_all',
 'U_all',
 'UU_1',
 'G_all',
 'GG_1',
 'GC_1',
 'GG_all',
 'DG_2',
 'UA_all',
 'U_2',
 'C_1',
 'CC_all',
 'DG_18',
 'CC_1',
 'GC_all',
 'CG_1',
 'DG_13',
 'UU_all',
 'A_19']

def reformat_td_columns(hu_data):
    hu_data2 = hu_data.copy()# [['siRNA', 'mRNA', 'y', 'label', 'td']]

    for ind, col in enumerate(lulab_td_columns):
        hu_data2.loc[:, col] = hu_data2['td'].map(lambda x: float(x.split(',')[ind]))
    
    hu_data2.drop(columns='td', inplace=True)
    
    return hu_data2


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

def calc_3_prime_stability(seq, signature):
    dg = calc_end_stability(seq,
            signature,
            mv_conc=200,
            dv_conc=100,
            dntp_conc=50,
            dna_conc=50,
            temp_c=37,
        ).dg / 1000

    return dg

hydro_nuc_paired = {
    'A': 0.393,
    'G': 0.342,
    'U': 0.350,
    'C': 0.384
}

hydro_nuc_unpaired = {
    'A': 0.798,
    'G': 0.827,
    'U': 0.967,
    'C': 0.962
}


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
        raise ValueError("siRNA sequence is too short for the given number of positions.")
    
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
    scale_factor = 3.5 / (middle**2)  # Adjust to get the values to around 4 at the ends and 1 in the middle
    diff = middle -i
    v = scale_factor * (diff)**2
    v *= (diff / abs(diff))
    
    return v

def DG_distribution_bias(seq: str):
    seq = seq.replace('T', 'U')
    slen = len(seq)
    middle = slen / 2
    dg_score = 0
    for i in range(slen - 1):
        nu = seq[i:i+2]
        dg = DeltaG[nu]
        modifier = calc_i_weight(i, slen)
        dg_score += dg * modifier

    return dg_score

def apply_internal_features(dataframe):
    
    dataframe['siRNA'] = dataframe['siRNA'].map(lambda x: x.replace('U', 'T'))
    dataframe['mRNA'] = dataframe['mRNA'].map(lambda x: x.replace('U', 'T').replace('X', ''))
    
    # Calculate the homodimer delta G
    dataframe['homodimer_dG'] = dataframe['siRNA'].map(calc_homo)
    
    # Calculate the heterodimer delta G
    dataframe['heterodimer_dG'] = dataframe.apply(lambda x: calc_hetero(x['siRNA'], x['mRNA']), axis=1)
    
    # Calculate the melting temperature
    dataframe['tm'] = dataframe['siRNA'].map(calc_tm)
    
    # Calculate the GC content
    dataframe['GC'] = dataframe['siRNA'].map(gc_fraction)
    
    dataframe['TTT_count'] = dataframe['siRNA'].map(lambda x: x.count('TTT'))
    dataframe['AAA_count'] = dataframe['siRNA'].map(lambda x: x.count('AAA'))
    dataframe['GGG_count'] = dataframe['siRNA'].map(lambda x: x.count('GGG'))
    dataframe['CCC_count'] = dataframe['siRNA'].map(lambda x: x.count('CCC'))
    
    dataframe['5p_4nt_GC'] = dataframe['siRNA'].map(lambda x: gc_fraction(x[:4]))
    dataframe['3p_4nt_GC'] = dataframe['siRNA'].map(lambda x: gc_fraction(x[-4:]))
    dataframe['ends_4nt_tm_asymetry'] = dataframe['siRNA'].map(calculate_tm_asymmetry)
    dataframe['ends_3nt_tm_asymetry'] = dataframe['siRNA'].map(lambda x: calculate_tm_asymmetry(x, num_positions=3))
    dataframe['ends_2nt_tm_asymetry'] = dataframe['siRNA'].map(lambda x: calculate_tm_asymmetry(x, num_positions=2))
    dataframe['dg_dist_bias'] = dataframe['siRNA'].map(DG_distribution_bias)

    dataframe['polarity'] = dataframe['siRNA'].map(calculate_polarity)
    dataframe['GC_skew'] = dataframe['siRNA'].apply(lambda x: (x.count('G') - x.count('C')) / (x.count('G') + x.count('C')))
    
    siRNA_r_dict = rfold(dataframe['siRNA'].to_dict())
    dataframe['siRNA_structured_percentage'] = [siRNA_r_dict[i].unpaired_seq_perc for i in siRNA_r_dict]
    dataframe['siRNA_vienna_mfe'] = [float(siRNA_r_dict[i].mfe) for i in siRNA_r_dict]

    mRNA_r_dict = rfold(dataframe['mRNA'].to_dict())
    dataframe['mRNA_structured_percentage'] = [mRNA_r_dict[i].unpaired_seq_perc for i in mRNA_r_dict]
    dataframe['mRNA_vienna_mfe'] = [float(mRNA_r_dict[i].mfe) for i in mRNA_r_dict]
    
    # Calculate the linguistic complexity 
    dataframe['lc'] = dataframe['siRNA'].map(calculate_lc)
    
    dataframe['hairpin_dg'] = dataframe['siRNA'].map(calc_hairpin_dg)
    
    # Count homopolymers
    dataframe['homopolymers_3long'] = dataframe['siRNA'].map(lambda x: homopolymer_counter(x, limit=3))
    
    # Calculate 3' prime stability
    dataframe['3_prime_stability'] = dataframe.apply(lambda x: calc_3_prime_stability(x['siRNA'], x['mRNA']), axis=1) 

    return dataframe

import time
from collections import OrderedDict

def time_apply_internal_features(dataframe):
    """
    Applies various feature calculations to the input dataframe and logs the time taken for each feature.
    
    Parameters:
    - dataframe (pd.DataFrame): The input dataframe containing 'siRNA' and 'mRNA' columns.
    
    Returns:
    - dataframe (pd.DataFrame): The dataframe with new features added.
    """
    
    # Dictionary to store timing information
    feature_timings = OrderedDict()
    
    # Helper function to time each feature computation
    def time_feature(feature_name, func):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        feature_timings[feature_name] = elapsed_time
    
    # 1. Replace 'U' with 'T' in 'siRNA' and remove 'X' in 'mRNA'
    time_feature('Replace U with T and clean mRNA', lambda: (
        dataframe.__setitem__('siRNA', dataframe['siRNA'].map(lambda x: x.replace('U', 'T'))),
        dataframe.__setitem__('mRNA', dataframe['mRNA'].map(lambda x: x.replace('U', 'T').replace('X', '')))
    ))
    
    # 2. Homodimer delta G
    time_feature('homodimer_dG', lambda: dataframe.__setitem__('homodimer_dG', dataframe['siRNA'].map(calc_homo)))
    
    # 3. Heterodimer delta G
    time_feature('heterodimer_dG', lambda: dataframe.__setitem__('heterodimer_dG', dataframe.apply(lambda x: calc_hetero(x['siRNA'], x['mRNA']), axis=1)))
    
    # 4. Melting temperature
    time_feature('tm', lambda: dataframe.__setitem__('tm', dataframe['siRNA'].map(calc_tm)))
    
    # 5. GC content
    time_feature('GC', lambda: dataframe.__setitem__('GC', dataframe['siRNA'].map(gc_fraction)))
    
    # 6. Triplet nucleotide counts
    for triplet in ['TTT', 'AAA', 'GGG', 'CCC']:
        time_feature(f'{triplet}_count', lambda triplet=triplet: dataframe.__setitem__(f'{triplet}_count', dataframe['siRNA'].map(lambda x: x.count(triplet))))
    
    # 7. GC content at 5' and 3' 4 nucleotides
    time_feature('5p_4nt_GC', lambda: dataframe.__setitem__('5p_4nt_GC', dataframe['siRNA'].map(lambda x: gc_fraction(x[:4]))))
    time_feature('3p_4nt_GC', lambda: dataframe.__setitem__('3p_4nt_GC', dataframe['siRNA'].map(lambda x: gc_fraction(x[-4:]))))
    
    # 8. Ends 4nt asymmetry
    time_feature('ends_4nt_asymetry', lambda: dataframe.__setitem__('ends_4nt_asymetry', dataframe['siRNA'].map(calculate_gibbs_asymmetry)))
    
    # 9. Polarity
    time_feature('polarity', lambda: dataframe.__setitem__('polarity', dataframe['siRNA'].map(calculate_polarity)))
    
    # 10. GC skew
    def compute_gc_skew():
        dataframe['GC_skew'] = dataframe['siRNA'].apply(lambda x: (x.count('G') - x.count('C')) / (x.count('G') + x.count('C') + 1e-6))  # Added small value to avoid division by zero
    time_feature('GC_skew', compute_gc_skew)
    
    # 11. siRNA structured percentage (ViennaRNA)
    time_feature('siRNA_structured_percentage', lambda: dataframe.__setitem__('siRNA_structured_percentage', dataframe['siRNA'].map(calc_structured_perc_vienna)))
    
    # 12. siRNA Vienna MFE
    time_feature('siRNA_vienna_mfe', lambda: dataframe.__setitem__('siRNA_vienna_mfe', dataframe['siRNA'].map(calc_structure_mfe_vienna)))
    
    # 13. mRNA structured percentage (ViennaRNA)
    time_feature('mRNA_structured_percentage', lambda: dataframe.__setitem__('mRNA_structured_percentage', dataframe['mRNA'].map(calc_structured_perc_vienna)))
    
    # 14. mRNA Vienna MFE
    time_feature('mRNA_vienna_mfe', lambda: dataframe.__setitem__('mRNA_vienna_mfe', dataframe['mRNA'].map(calc_structure_mfe_vienna)))
    
    # 15. Linguistic complexity
    time_feature('lc', lambda: dataframe.__setitem__('lc', dataframe['siRNA'].map(calculate_lc)))
    
    # 16. Hairpin ΔG
    time_feature('hairpin_dg', lambda: dataframe.__setitem__('hairpin_dg', dataframe['siRNA'].map(calc_hairpin_dg)))
    
    # 17. Homopolymers (3 long)
    time_feature('homopolymers_3long', lambda: dataframe.__setitem__('homopolymers_3long', dataframe['siRNA'].map(lambda x: homopolymer_counter(x, limit=3))))
    
    # 18. 3' prime stability
    time_feature('3_prime_stability', lambda: dataframe.__setitem__('3_prime_stability', dataframe.apply(lambda x: calc_3_prime_stability(x['siRNA'], x['mRNA']), axis=1)))
    
    # Log the timing results
    print("\nFeature Computation Timings:")
    for feature, elapsed in feature_timings.items():
        print(f"{feature}: {elapsed:.4f} seconds")
    
    # Identify the slowest feature computation
    slowest_feature = max(feature_timings, key=feature_timings.get)
    print(f"\nSlowest Feature: {slowest_feature} ({feature_timings[slowest_feature]:.4f} seconds)")
    
    return dataframe


DeltaG = {'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24,  'AC': -2.24, 'GA': -2.35,  'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42, 'init': 4.09, 'endAU': 0.45, 'sym': 0.43}
DeltaH = {'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 'CU': -10.48, 'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 'GU': -11.40,  'AC': -11.40, 'GA': -12.44,  'UC': -12.44, 'CG': -10.64, 'GG': -13.39, 'CC': -13.39, 'GC': -14.88, 'init': 3.61, 'endAU': 3.72, 'sym': 0}

def antiRNA(RNA):
    antiRNA = []
    for i in RNA:
        if i == 'A' or i == 'a':
            antiRNA.append('U')
        elif i == 'U' or i == 'u' or i == 'T' or i == 't':
            antiRNA.append('A')
        elif i == 'C' or i == 'c':
            antiRNA.append('G')
        elif i == 'G' or i == 'g':
            antiRNA.append('C')
        elif i == 'X' or i == 'x':
            antiRNA.append('X')
    return ''.join(antiRNA[::-1])

def Calculate_DGH(seq):
    DG_all = 0
    DG_all += DeltaG['init']
    DG_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaG['endAU']
    DG_all += DeltaG['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DG_all += DeltaG[seq[i] + seq[i+1]]
    DH_all = 0
    DH_all += DeltaH['init']
    DH_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaH['endAU']
    DH_all += DeltaH['sym'] if antiRNA(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        DH_all += DeltaH[seq[i] + seq[i+1]]
    return DG_all,DH_all

def Calculate_end_diff(siRNA):
    count = 0
    _5 = siRNA[:2] # 5'end
    _3 = siRNA[-2:] # 3' end
    if _5 in ['AC','AG','UC','UG']:
        count += 1
    elif _5 in ['GA','GU','CA','CU']:
        count -= 1
    if _3 in ['AC','AG','UC','UG']:
        count += 1
    elif _3 in ['GA','GU','CA','CU']:
        count -= 1
    return float('{:.2f}'.format(DeltaG[_5] - DeltaG[_3] + count * 0.45))

def calculate_td(df):
    # Precompute values for the entire DataFrame using vectorized operations
    siRNA_series = df['siRNA']
    
    # Initialize the columns
    df['ends'] = siRNA_series.apply(Calculate_end_diff)
    df['DG_1'] = siRNA_series.str[0:2].map(DeltaG)
    df['DH_1'] = siRNA_series.str[0:2].map(DeltaH)
    df['U_1'] = (siRNA_series.str[0] == 'U').astype(int)
    df['G_1'] = (siRNA_series.str[0] == 'G').astype(int)
    df['DH_all'] = siRNA_series.apply(lambda x: Calculate_DGH(x)[1])
    df['U_all'] = siRNA_series.apply(lambda x: x.count('U') / 19)
    df['UU_1'] = (siRNA_series.str[0:2] == 'UU').astype(int)
    df['G_all'] = siRNA_series.apply(lambda x: x.count('G') / 19)
    df['GG_1'] = (siRNA_series.str[0:2] == 'GG').astype(int)
    df['GC_1'] = (siRNA_series.str[0:2] == 'GC').astype(int)
    df['GG_all'] = siRNA_series.apply(lambda x: sum(1 for i in range(18) if x[i:i+2] == 'GG') / 18)
    df['DG_2'] = siRNA_series.str[1:3].map(DeltaG)
    df['UA_all'] = siRNA_series.apply(lambda x: sum(1 for i in range(18) if x[i:i+2] == 'UA') / 18)
    df['U_2'] = (siRNA_series.str[1] == 'U').astype(int)
    df['C_1'] = (siRNA_series.str[0] == 'C').astype(int)
    df['CC_all'] = siRNA_series.apply(lambda x: sum(1 for i in range(18) if x[i:i+2] == 'CC') / 18)
    df['DG_18'] = siRNA_series.str[17:19].map(DeltaG)
    df['CC_1'] = (siRNA_series.str[0:2] == 'CC').astype(int)
    df['GC_all'] = siRNA_series.apply(lambda x: sum(1 for i in range(18) if x[i:i+2] == 'GC') / 18)
    df['CG_1'] = (siRNA_series.str[0:2] == 'CG').astype(int)
    df['DG_13'] = siRNA_series.str[12:14].map(DeltaG)
    df['UU_all'] = siRNA_series.apply(lambda x: sum(1 for i in range(18) if x[i:i+2] == 'UU') / 18)
    df['A_19'] = (siRNA_series.str[18] == 'A').astype(int)

    return df

def calculate_td_slow(df):
	df['ends'] = df['siRNA']
	df['DG_1'] = df['siRNA']
	df['DH_1'] = df['siRNA']
	df['U_1'] = df['siRNA']
	df['G_1'] = df['siRNA']
	df['DH_all'] = df['siRNA']
	df['U_all'] = df['siRNA']
	df['UU_1'] = df['siRNA']
	df['G_all'] = df['siRNA']
	df['GG_1'] = df['siRNA']
	df['GC_1'] = df['siRNA']
	df['GG_all'] = df['siRNA']
	df['DG_2'] = df['siRNA']
	df['UA_all'] = df['siRNA']
	df['U_2'] = df['siRNA']
	df['C_1'] = df['siRNA']
	df['CC_all'] = df['siRNA']
	df['DG_18'] = df['siRNA']
	df['CC_1'] = df['siRNA']
	df['GC_all'] = df['siRNA']
	df['CG_1'] = df['siRNA']
	df['DG_13'] = df['siRNA']
	df['UU_all'] = df['siRNA']
	df['A_19'] = df['siRNA']

	for i in range(df.shape[0]):
		df['ends'] = [Calculate_end_diff(i) for i in df['siRNA']]
		df['DG_1'][i] = DeltaG[df.iloc[i,0][0:2]]
		df['DH_1'][i] = DeltaH[df.iloc[i,0][0:2]]
		df['U_1'][i] = int(df.iloc[i,0][0] == 'U')
		df['G_1'][i] = int(df.iloc[i,0][0] == 'G')
		df['DH_all'][i] = Calculate_DGH(df.iloc[i,0])[1]
		df['U_all'][i] = df.iloc[i,0].count('U') / 19
		df['UU_1'][i] = int(df.iloc[i,0][0:2] == 'UU')
		df['G_all'][i] = df.iloc[i,0].count('G') / 19
		df['GG_1'][i] = int(df.iloc[i,0][0:2] == 'GG')
		df['GC_1'][i] = int(df.iloc[i,0][0:2] == 'GC')
		df['GG_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('GG') / 18
		df['DG_2'][i] = DeltaG[df.iloc[i,0][1:3]]
		df['UA_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('UA') / 18
		df['U_2'][i] = int(df.iloc[i,0][1] == 'U')
		df['C_1'][i] = int(df.iloc[i,0][0] == 'C')
		df['CC_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('CC') / 18
		df['DG_18'][i] = DeltaG[df.iloc[i,0][17:19]]
		df['CC_1'][i] = int(df.iloc[i,0][0:2] == 'CC')
		df['GC_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('GC') / 18
		df['CG_1'][i] = int(df.iloc[i,0][0:2] == 'CG')
		df['DG_13'][i] = DeltaG[df.iloc[i,0][12:14]]
		df['UU_all'][i] = [df.iloc[i,0][j]+df.iloc[i,0][j+1] for j in range(18)].count('UU') / 18
		df['A_19'][i] = int(df.iloc[i,0][18] == 'A')
	return df


