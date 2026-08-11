#!/usr/bin/env python
"""
Extract unique mRNA sequences from invitro_curated_train_preprocessed.csv
and write them to a FASTA file for future use.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load the preprocessed CSV
    csv_path = './src/fennec/support_files/invitro_curated_train_preprocessed.csv'
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Loaded {len(df)} rows from CSV")
    
    # Get unique gene -> mRNA mappings
    # Group by gene and take the first mRNA (they should be identical for each gene)
    gene_mrna = df[['gene_target_symbol_name', 'mRNA']].dropna().drop_duplicates(subset='gene_target_symbol_name')
    
    print(f"Found {len(gene_mrna)} unique genes with mRNA sequences")
    
    # Write to FASTA
    output_path = './helper_scripts/gene_mrna_sequences.fasta'
    
    with open(output_path, 'w') as f:
        for idx, row in gene_mrna.iterrows():
            gene = row['gene_target_symbol_name']
            mrna = row['mRNA']
            
            # Write FASTA header and sequence
            # Format: >GENE_NAME length=XXXXbp
            f.write(f">{gene} length={len(mrna)}bp\n")
            
            # Write sequence in 80 character lines (standard FASTA format)
            for i in range(0, len(mrna), 80):
                f.write(mrna[i:i+80] + '\n')
    
    print(f"\n✓ Wrote {len(gene_mrna)} unique mRNA sequences to: {output_path}")
    
    # Summary stats
    lengths = [len(row['mRNA']) for _, row in gene_mrna.iterrows()]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(lengths)} bp")
    print(f"  Max: {max(lengths)} bp")
    print(f"  Mean: {sum(lengths)//len(lengths)} bp")
    
    # Show a few examples
    print(f"\n=== Sample entries ===")
    for _, row in gene_mrna.head(5).iterrows():
        gene = row['gene_target_symbol_name']
        mrna_len = len(row['mRNA'])
        print(f"{gene}: {mrna_len} bp")

if __name__ == "__main__":
    main()
