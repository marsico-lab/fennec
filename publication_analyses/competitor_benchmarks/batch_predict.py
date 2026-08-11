#!/usr/bin/env python
"""Batch-submit siRNA candidates to the cm-siRNA-pred API (cellknowledge.com.cn).

Usage:
    python batch_predict.py --input candidates.csv --output predictions.csv
"""
import argparse
import os
import time

import pandas as pd
from sirna_predictor_api import submit_sirna_prediction


def main():
    p = argparse.ArgumentParser(description="Batch cm-siRNA-pred predictions")
    p.add_argument('--input', required=True,
                   help='CSV with targetsite and siRNA23mer columns')
    p.add_argument('--output', default='cellknowledge_predictions.csv')
    p.add_argument('--checkpoint', default='cellknowledge_checkpoint.csv')
    p.add_argument('--batch-size', type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.input)

    sense_mods = {
        '2-Methyl': [1, 2, 3, 4, 5, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        '2-O-Fluoro': [7, 9, 10, 11]
    }
    antisense_mods = {
        '2-Methyl': [1, 3, 4, 5, 7, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23],
        '2-O-Fluoro': [2, 6, 8, 9, 14, 16]
    }

    if os.path.exists(args.checkpoint):
        checkpoint_df = pd.read_csv(args.checkpoint)
        results = checkpoint_df.to_dict('records')
        processed_indices = set(checkpoint_df['index'].values)
        start_idx = max(processed_indices) + 1 if processed_indices else 0
        print(f"Resuming from checkpoint. Already processed: {len(results)} records")
    else:
        results = []
        processed_indices = set()
        start_idx = 0
        print("Starting fresh batch prediction")

    remaining_df = df[~df.index.isin(processed_indices) & (df.index >= start_idx)]

    for batch_start in range(0, len(remaining_df), args.batch_size):
        batch_df = remaining_df.iloc[batch_start:batch_start + args.batch_size]
        batch_indices = batch_df.index.tolist()
        batch_names = [f"AHSA1_{idx}" for idx in batch_indices]
        batch_sense = [row['targetsite'].replace('T', 'U') for _, row in batch_df.iterrows()]
        batch_antisense = batch_df['siRNA23mer'].tolist()

        print(f"Processing batch {batch_start // args.batch_size + 1}: "
              f"indices {batch_indices[0]}-{batch_indices[-1]} ({len(batch_indices)} sequences)")

        batch_results = submit_sirna_prediction(
            sirna_name=batch_names,
            sense_seq=batch_sense,
            antisense_seq=batch_antisense,
            sense_mods=sense_mods,
            antisense_mods=antisense_mods
        )

        for idx, name, sense, antisense, result in zip(
                batch_indices, batch_names, batch_sense, batch_antisense, batch_results):
            results.append({
                'index': idx,
                'sirna_name': name,
                'sense_seq': sense,
                'antisense_seq': antisense,
                'status': result.get('status'),
                'predicted_score': result.get('siRNA_score'),
                'response_seq': result.get('siRNA_seq'),
                'response_mod': result.get('siRNA_mod'),
                'folder': result.get('folderName'),
                'error_message': result.get('message') if result.get('status') != 'success' else None
            })

        pd.DataFrame(results).to_csv(args.checkpoint, index=False)
        time.sleep(1.0)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)

    print(f"\nCompleted {len(results)} predictions. Saved to {args.output}")
    print(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Errors: {sum(1 for r in results if r['status'] != 'success')}")


if __name__ == '__main__':
    main()
