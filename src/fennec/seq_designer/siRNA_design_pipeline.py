"""
siRNA Design Pipeline

A comprehensive pipeline for designing and predicting siRNA efficacy using chemical modifications.
Extracted and refactored from Jupyter notebooks to provide reusable functionality.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.nn import ZeroPad2d
from Bio import SeqIO
from Bio.Seq import reverse_complement_rna
import sh as _sh
def esearch(*a, **kw): return _sh.esearch(*a, **kw)
def efetch(*a, **kw):  return _sh.efetch(*a, **kw)
from tqdm import tqdm
from glob import glob
from itertools import product
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

try:
    from fennec.featurizer import helm_one_hot_encode, helm_one_hot_decode
    from fennec.preprocessing import preprocess_input_tracks
    from fennec.models import MODELS
    FENNEC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FENNEC modules not available: {e}")
    print("Some functionality will be limited.")
    FENNEC_AVAILABLE = False
    
    # Create dummy variables to prevent import errors
    helm_one_hot_encode = None
    helm_one_hot_decode = None
    preprocess_input_tracks = None
    MODELS = {}

try:
    from primer3.bindings import calc_tm
    PRIMER3_AVAILABLE = True
except ImportError:
    PRIMER3_AVAILABLE = False
    print("Warning: primer3 not available. Melting temperature calculation will be skipped.")


@dataclass
class DesignParameters:
    """Configuration class for siRNA design parameters."""
    
    # Sequence parameters
    kmer_sizes: List[int] = field(default_factory=lambda: [20, 21, 22, 23, 24])
    passenger_shifts: List[int] = field(default_factory=lambda: [0, 1, 2])
    gc_min: float = 0.30
    gc_max: float = 0.62
    
    # Chemical modification parameters
    chem_dim: int = 16
    seq_len: int = 25
    pattern_n_min: int = 30
    
    # Model parameters
    model_type: str = 'TCN_FM_multi_head'
    device: str = 'cuda'
    batch_size: int = 128
    
    # Feature parameters
    no_passenger: bool = False
    with_chem_ablate: bool = False
    with_thermo: bool = True
    fm_type: str = 'rna_fm'
    slice_length: int = 25
    only_cls_eos: bool = False
    
    # Target gene parameters
    gene_target_species: str = 'Homo sapiens'
    duration_after_transfection_h: int = 24
    transfection_method: str = 'Lipofectamine'
    cell_line_donor: str = 'Unknown_cell_line'
    siRNA_concentration: int = 10
    mRNA_remaining_pct: int = 50
    
    # Selection criteria
    std_cutoff: float = 12.0
    confidence_multiplier: float = 1.96
    strong_knockdown_n: int = 60
    weak_knockdown_n: int = 25
    
    # File filters
    exclude_L96: bool = True
    exclude_dna: bool = True


# HELM encoder functionality is now provided by fennec.featurizer
# We'll use helm_one_hot_encode and helm_one_hot_decode directly from fennec


class siRNADesigner:
    """Main siRNA design pipeline class."""
    
    def __init__(self, params: DesignParameters = None):
        """Initialize the designer with parameters."""
        self.params = params if params else DesignParameters()
        
    def fetch_sequence(self, fasta_id: str, output_path: str = None) -> SeqIO.SeqRecord:
        """
        Fetch a sequence from NCBI using the provided ID.
        
        Args:
            fasta_id: NCBI accession ID
            output_path: Path to save the FASTA file (optional)
            
        Returns:
            SeqIO record object
        """
        if output_path is None:
            output_path = f"{fasta_id}.fna"
            
        try:
            with open(output_path, 'w') as fasta_buffer:
                efetch('-db', 'nuccore', '-id', fasta_id, '-format', 'fasta', _out=fasta_buffer)
            record = SeqIO.read(output_path, 'fasta')
            return record
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sequence {fasta_id}: {e}")
    
    def load_chemical_patterns(self, guide_patterns_file: str, passenger_patterns_file: str,
                              guide_passenger_pairs_file: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load chemical modification patterns from CSV files.
        
        Args:
            guide_patterns_file: Path to guide strand chemical patterns CSV
            passenger_patterns_file: Path to passenger strand chemical patterns CSV  
            guide_passenger_pairs_file: Path to valid guide-passenger pairs CSV
            
        Returns:
            Tuple of (guide_patterns_dict, passenger_patterns_dict, valid_pairs_dict)
        """
        m = ZeroPad2d((0, 0, 5, 0))
        
        # Load guide patterns
        chem_patterns = pd.read_csv(guide_patterns_file)
        mat_pats = {}
        for i in range(1, len(chem_patterns)):
            n_reps = chem_patterns.iloc[i][-1].item()
            if n_reps < self.params.pattern_n_min:
                continue
            matp = chem_patterns.iloc[i][:-1].values.reshape(self.params.chem_dim, self.params.seq_len)
            matp = torch.Tensor(matp)
            matp = m(matp)
            slen = len([p for p in matp.sum(axis=0) if p != 0]) - 1
            mat_pats[i] = {
                'pattern': matp,
                'pattern_seq_len': slen,
                'seq_with_pattern_n': n_reps
            }
        
        # Load passenger patterns
        chem_patterns_pass = pd.read_csv(passenger_patterns_file)
        mat_pats_pass = {}
        for i in range(1, len(chem_patterns_pass)):
            n_reps = chem_patterns_pass.iloc[i][-1].item()
            if n_reps < self.params.pattern_n_min:
                continue
            matp = chem_patterns_pass.iloc[i][:-1].values.reshape(self.params.chem_dim, self.params.seq_len)
            matp = torch.Tensor(matp)
            matp = m(matp)
            slen = len([p for p in matp.sum(axis=0) if p != 0]) - 1
            mat_pats_pass[i] = {
                'pattern': matp,
                'pattern_seq_len': slen,
                'seq_with_pattern_n': n_reps
            }
        
        # Load valid pairs
        real_guide_pass_inds = pd.read_csv(guide_passenger_pairs_file, index_col=[0, 1])
        real_pairs = real_guide_pass_inds['0'].to_dict()
        
        return mat_pats, mat_pats_pass, real_pairs
    
    @staticmethod
    def calculate_gc_content(seq: str) -> float:
        """Calculate GC content of a sequence."""
        return (seq.count('C') + seq.count('G')) / len(seq)
    
    def generate_siRNA_designs(self, sequence: str, mat_pats: Dict, mat_pats_pass: Dict, 
                              real_pairs: Dict, gene_symbol: str, transcript_id: str) -> pd.DataFrame:
        """
        Generate siRNA designs from a target sequence.
        
        Args:
            sequence: Target mRNA sequence
            mat_pats: Guide strand chemical patterns
            mat_pats_pass: Passenger strand chemical patterns
            real_pairs: Valid guide-passenger pattern pairs
            gene_symbol: Gene symbol for metadata
            transcript_id: NCBI transcript ID for metadata
            
        Returns:
            DataFrame with siRNA designs
        """
        pattern_combinations = list(product(mat_pats.keys(), mat_pats_pass.keys()))
        
        unmod_siRNA = []
        siRNA_list = []
        siRNA_passenger_list = []
        guide_pass_inds = []
        
        seq = sequence.replace('T', 'U')
        
        for k in tqdm(self.params.kmer_sizes, desc="Processing kmer sizes"):
            for i in range(0, len(seq) - k):
                passenger_kmer = seq[i:i+k]
                sgc = self.calculate_gc_content(passenger_kmer)
                
                # Filter by GC content
                if sgc > self.params.gc_max or sgc < self.params.gc_min:
                    continue
                    
                tile = reverse_complement_rna(passenger_kmer)
                
                # Filter by 5' nucleotide
                if tile[0] not in ['U', 'A']:
                    continue
                    
                passenger = passenger_kmer[::-1]
                tile_list = ' '.join(list(tile))
                pass_list = ' '.join(list(passenger))
                
                try:
                    if not FENNEC_AVAILABLE:
                        raise ImportError("FENNEC not available. Please install fennec to use HELM encoding.")
                    tile_mat = torch.Tensor(helm_one_hot_encode(tile_list)).T
                    pass_mat = torch.Tensor(helm_one_hot_encode(pass_list)).T
                    tile_mat[5:, :] = 0
                    pass_mat[5:, :] = 0
                except Exception as e:
                    print(f"Warning: Failed to encode sequence {tile}: {e}")
                    continue
                
                for guide_ind, pass_ind in pattern_combinations:
                    if (guide_ind, pass_ind) not in real_pairs:
                        continue
                        
                    for pshift in self.params.passenger_shifts:
                        guide_pat_len = mat_pats[guide_ind]['pattern_seq_len']
                        pass_pat_len = mat_pats_pass[pass_ind]['pattern_seq_len']
                        
                        if guide_pat_len != k or pass_pat_len != k - pshift:
                            continue
                            
                        guide_pat = mat_pats[guide_ind]['pattern'].clone()
                        pass_pat = mat_pats_pass[pass_ind]['pattern'].clone()
                        pass_mat[:, k-pshift+1:] = 0
                        
                        new_tile = tile_mat + guide_pat
                        new_pass = pass_mat + pass_pat
                        
                        try:
                            if not FENNEC_AVAILABLE:
                                raise ImportError("FENNEC not available. Please install fennec to use HELM decoding.")
                            new_tile_str = helm_one_hot_decode(new_tile.T.tolist())
                            new_pass_str = helm_one_hot_decode(new_pass.T.tolist())
                        except Exception as e:
                            print(f"Warning: Failed to decode sequence: {e}")
                            continue
                        
                        unmod_siRNA.append(tile)
                        siRNA_list.append(new_tile_str)
                        siRNA_passenger_list.append(new_pass_str)
                        guide_pass_inds.append((guide_ind, pass_ind))
        
        # Create DataFrame
        design_df = pd.DataFrame({
            'siRNA_list': siRNA_list,
            'siRNA': unmod_siRNA,
            'siRNA_passenger_list': siRNA_passenger_list,
            'guide_passenger_pair': guide_pass_inds
        })
        
        # Apply filters
        if self.params.exclude_L96:
            design_df = design_df[design_df['siRNA_passenger_list'].str.contains('L96') == False]
        
        if self.params.exclude_dna:
            design_df = design_df[
                (design_df['siRNA_list'].str.contains(' d') == False) & 
                (design_df['siRNA_passenger_list'].str.contains(' d') == False)
            ]
        
        design_df = design_df.drop_duplicates()
        
        # Add metadata
        design_df['gene_target_species'] = self.params.gene_target_species
        design_df['gene_target_symbol_name'] = gene_symbol
        design_df['mRNA'] = sequence
        design_df['ncbi_transcript_unique_id'] = transcript_id
        design_df['Duration_after_transfection_h'] = self.params.duration_after_transfection_h
        design_df['Transfection_method'] = self.params.transfection_method
        design_df['cell_line_donor'] = self.params.cell_line_donor
        design_df['siRNA_concentration'] = self.params.siRNA_concentration
        design_df['mRNA_remaining_pct'] = self.params.mRNA_remaining_pct
        
        return design_df
    
    def predict_efficacy(self, design_df: pd.DataFrame, model_paths: List[str], 
                        rna_emb_path: str = '.') -> pd.DataFrame:
        """
        Predict siRNA efficacy using trained models.
        
        Args:
            design_df: DataFrame with siRNA designs
            model_paths: List of paths to trained model files
            rna_emb_path: Path to RNA embeddings
            
        Returns:
            DataFrame with predictions
        """
        if 'preprocess_input_tracks' not in globals():
            raise ImportError("fennec modules not available for prediction")
        
        if not FENNEC_AVAILABLE:
            raise ImportError("FENNEC not available. Please install fennec for preprocessing.")
        
        # Preprocess input
        X, y, X_meta, X_emb, mrna_map_index = preprocess_input_tracks(
                                                                        design_df, 
                                                                        rna_emb_path, 
                                                                        device=self.params.device,
                                                                        with_passenger=not self.params.no_passenger,
                                                                        with_chem_ablate=self.params.with_chem_ablate,
                                                                        with_thermo=self.params.with_thermo,
                                                                        fm_type=self.params.fm_type,
                                                                        slice_length=self.params.slice_length,
                                                                        only_cls_eos=self.params.only_cls_eos
                                                                    )

        # Training checkpoints are stored in float32, but some preprocessing
        # paths can surface float64 meta features at inference time.
        X = X.float()
        X_meta = X_meta.float()
        X_emb = X_emb.float()
        
        X_emb = X_emb[mrna_map_index].to(self.params.device)
        
        # Run predictions
        model_preds = {}
        
        for w_path in tqdm(model_paths, desc="Running model predictions"):
            model_name = os.path.basename(w_path).replace('.pt', '')
            n_X, total_channels, seq_len = X.shape
            meta_input_dim = X_meta.shape[-1]
            nembs, fm_seq_len, fm_embed_dim = X_emb.shape
            
            model = MODELS[self.params.model_type](
                input_channels=total_channels, 
                meta_input_dim=meta_input_dim,
                fm_seq_len=fm_seq_len,
                fm_embed_dim=fm_embed_dim,
                num_classes=1,
                regression=True,
            ).to(self.params.device)
            
            model.load_state_dict(torch.load(w_path, weights_only=False))
            model.eval()
            
            with torch.no_grad():
                y_pred = model(
                    X.to(self.params.device), 
                    X_meta.to(self.params.device), 
                    X_emb.to(self.params.device)
                ).squeeze(1).cpu().numpy()
            
            model_preds[model_name] = y_pred
            del model
            torch.cuda.empty_cache()
        
        # Aggregate predictions
        predictions_df = pd.DataFrame(model_preds)
        predictions_df['stdev'] = predictions_df.std(axis=1)
        predictions_df['mean_pred'] = predictions_df.mean(axis=1)
        
        return predictions_df
    
    def analyze_predictions(self, design_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze and annotate predictions with additional metrics.
        
        Args:
            design_df: Original design DataFrame
            predictions_df: Predictions DataFrame
            
        Returns:
            Combined DataFrame with analysis
        """
        # Combine dataframes
        result_df = pd.concat([design_df, predictions_df], axis=1)
        
        # Parse guide and passenger pattern indices
        result_df['guide_pat_ind'] = result_df['guide_passenger_pair'].map(
            lambda x: x.split(',')[0].strip('()')
        )
        result_df['passenger_pat_ind'] = result_df['guide_passenger_pair'].map(
            lambda x: x.split(',')[1].strip('()')
        )
        
        # Calculate siRNA location on mRNA
        locations = []
        percent_locations = []
        
        for sirna, mrna in zip(result_df['siRNA'], result_df['mRNA']):
            loc = mrna.replace('T', 'U').find(reverse_complement_rna(sirna))
            locations.append(loc)
            percent_locations.append(loc / len(mrna) if loc >= 0 else -1)
        
        result_df['siRNA_location'] = locations
        result_df['siRNA_percent_loc'] = percent_locations
        
        # Calculate melting temperatures if primer3 is available
        if PRIMER3_AVAILABLE:
            result_df['TM'] = result_df['siRNA'].map(
                lambda x: calc_tm(x.replace('U', 'T'))
            )
        
        # Calculate confidence intervals
        result_df['mean_conf'] = result_df['mean_pred'] + (
            self.params.confidence_multiplier * result_df['stdev']
        )
        
        return result_df
    
    def select_final_designs(self, result_df: pd.DataFrame, best_pattern: str = None) -> pd.DataFrame:
        """
        Select final siRNA designs based on prediction criteria.
        
        Args:
            result_df: DataFrame with predictions and analysis
            best_pattern: Specific pattern to focus on (optional)
            
        Returns:
            DataFrame with selected designs
        """
        if best_pattern is None:
            # Find best performing pattern
            best_pattern = (result_df.groupby('guide_passenger_pair')['mean_pred']
                          .mean().sort_values(ascending=False).index[0])
        
        # Filter to best pattern
        filtered_df = result_df[result_df['guide_passenger_pair'] == best_pattern]
        
        # Apply standard deviation filter
        filtered_df = filtered_df[filtered_df['stdev'] <= self.params.std_cutoff]
        
        # Select strong knockdown candidates (lowest predictions = highest knockdown)
        strong_candidates = (filtered_df.sort_values(by='mean_conf')
                           .head(self.params.strong_knockdown_n).copy())
        strong_candidates['prediction_class'] = 'Strong knockdown'
        
        # Select weak knockdown candidates for comparison
        weak_candidates = (filtered_df.sort_values(by='mean_pred')
                         .tail(self.params.weak_knockdown_n).copy())
        weak_candidates['prediction_class'] = 'Weak knockdown'
        
        # Combine selections
        final_designs = pd.concat([strong_candidates, weak_candidates])
        
        # Remove mRNA column to reduce file size
        final_designs = final_designs.drop(columns=['mRNA'], errors='ignore')
        
        return final_designs
    
    def run_full_pipeline(self, fasta_id: str, gene_symbol: str,
                         guide_patterns_file: str, passenger_patterns_file: str,
                         guide_passenger_pairs_file: str, model_paths: List[str],
                         output_prefix: str = None, rna_emb_path: str = '.') -> Dict[str, pd.DataFrame]:
        """
        Run the complete siRNA design pipeline.
        
        Args:
            fasta_id: NCBI accession ID for target sequence
            gene_symbol: Gene symbol
            guide_patterns_file: Path to guide chemical patterns CSV
            passenger_patterns_file: Path to passenger chemical patterns CSV
            guide_passenger_pairs_file: Path to valid pairs CSV
            model_paths: List of trained model file paths
            output_prefix: Prefix for output files (optional)
            rna_emb_path: Path to RNA embeddings
            
        Returns:
            Dictionary with 'designs', 'full_results', and 'final_selection' DataFrames
        """
        if output_prefix is None:
            output_prefix = f"{gene_symbol}_{fasta_id}"
        
        print("Step 1: Fetching target sequence...")
        record = self.fetch_sequence(fasta_id)
        sequence = str(record.seq)
        
        print("Step 2: Loading chemical modification patterns...")
        mat_pats, mat_pats_pass, real_pairs = self.load_chemical_patterns(
            guide_patterns_file, passenger_patterns_file, guide_passenger_pairs_file
        )
        
        print(f"Loaded {len(mat_pats)} guide patterns and {len(mat_pats_pass)} passenger patterns")
        
        print("Step 3: Generating siRNA designs...")
        designs_df = self.generate_siRNA_designs(
            sequence, mat_pats, mat_pats_pass, real_pairs, gene_symbol, fasta_id
        )
        
        print(f"Generated {len(designs_df)} siRNA designs")
        designs_df.to_csv(f"{output_prefix}_designs.csv", index=False)
        
        print("Step 4: Predicting efficacy...")
        predictions_df = self.predict_efficacy(designs_df, model_paths, rna_emb_path)
        
        print("Step 5: Analyzing results...")
        full_results = self.analyze_predictions(designs_df, predictions_df)
        
        # Find best pattern
        best_pattern = (full_results.groupby('guide_passenger_pair')['mean_pred']
                       .mean().sort_values(ascending=False).index[0])
        
        print(f"Best performing pattern: {best_pattern}")
        
        print("Step 6: Selecting final designs...")
        final_selection = self.select_final_designs(full_results, best_pattern)
        
        # Save results
        full_results.to_csv(f"{output_prefix}_full_results.csv", index=False)
        final_selection.to_csv(f"{output_prefix}_final_selection.csv", index=False)
        
        print(f"Pipeline complete! Selected {len(final_selection)} final designs.")
        
        return {
            'designs': designs_df,
            'full_results': full_results,
            'final_selection': final_selection,
            'best_pattern': best_pattern
        }


# Convenience functions for common use cases
def design_siRNA_for_gene(fasta_id: str, gene_symbol: str, 
                         data_dir: str = '.', model_dir: str = None,
                         output_dir: str = '.', custom_params: Dict = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to design siRNAs for a specific gene.
    
    Args:
        fasta_id: NCBI accession ID
        gene_symbol: Gene symbol
        data_dir: Directory containing chemical pattern files
        model_dir: Directory containing trained models (will search for *.pt files)
        output_dir: Directory for output files
        custom_params: Dictionary of custom parameters to override defaults
        
    Returns:
        Dictionary with pipeline results
    """
    # Set up parameters
    params = DesignParameters()
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
    
    # Set up file paths
    guide_patterns_file = os.path.join(data_dir, 'chemical_patterns_paddle.csv')
    passenger_patterns_file = os.path.join(data_dir, 'chemical_patterns_passenger_paddle.csv')
    guide_passenger_pairs_file = os.path.join(data_dir, 'guide_passenger_index_pairs.csv')
    
    # Find model files
    if model_dir:
        model_pattern = os.path.join(model_dir, '*.pt')
    else:
        model_pattern = os.path.join(data_dir, '*model_output.pt')
    
    model_paths = glob(model_pattern)
    
    if not model_paths:
        raise FileNotFoundError(f"No model files found with pattern: {model_pattern}")
    
    print(f"Found {len(model_paths)} model files")
    
    # Initialize designer and run pipeline
    designer = siRNADesigner(params)
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        results = designer.run_full_pipeline(
            fasta_id=fasta_id,
            gene_symbol=gene_symbol,
            guide_patterns_file=guide_patterns_file,
            passenger_patterns_file=passenger_patterns_file,
            guide_passenger_pairs_file=guide_passenger_pairs_file,
            model_paths=model_paths,
            output_prefix=f"{gene_symbol}_{fasta_id}"
        )
    finally:
        os.chdir(original_dir)
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="siRNA Design Pipeline")
    parser.add_argument("--fasta_id", required=True, help="NCBI accession ID")
    parser.add_argument("--gene_symbol", required=True, help="Gene symbol")
    parser.add_argument("--data_dir", default=".", help="Directory with chemical pattern files")
    parser.add_argument("--model_dir", help="Directory with trained models")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    results = design_siRNA_for_gene(
        fasta_id=args.fasta_id,
        gene_symbol=args.gene_symbol,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    print("Design pipeline completed successfully!")
    print(f"Final selection contains {len(results['final_selection'])} designs")