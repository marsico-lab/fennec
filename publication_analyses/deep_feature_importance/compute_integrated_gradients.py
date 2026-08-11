#!/usr/bin/env python
"""Compute Integrated Gradients on pre-trained FENNEC weights.

Featurizes the dataset once, then for each supplied checkpoint runs
Captum IG and saves attribution tensors to --save-dir.

Usage:
    python compute_integrated_gradients.py \\
        --data-csv path/to/data.csv.gz \\
        --checkpoints weights/seed0.pt weights/seed1.pt \\
        --save-dir ig_outputs/
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import torch
from captum.attr import IntegratedGradients

from fennec.preprocessing import preprocess_input_tracks
from fennec.models import MODELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Integrated Gradients on pre-trained FENNEC weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-csv', required=True,
                   help='Path to siRNA data CSV (gzip ok)')
    p.add_argument('--checkpoints', nargs='+', required=True, metavar='PT',
                   help='One or more .pt weight files (one per seed / ensemble member)')
    p.add_argument('--save-dir', default='ig_outputs',
                   help='Directory where ig_<stem>.pt files are written')
    p.add_argument('--model', default='TCN_FM_multi_head',
                   choices=sorted(MODELS),
                   help='FENNEC model class matching the supplied weights')
    p.add_argument('--fm-type', default='orthrus_4track_triple')
    p.add_argument('--slice-length', type=int, default=100)
    p.add_argument('--with-thermo', action='store_true', default=True)
    p.add_argument('--modalities', nargs='+', choices=['x', 'meta', 'rna'],
                   default=['x', 'meta', 'rna'])
    p.add_argument('--ig-steps', type=int, default=128)
    p.add_argument('--ig-batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, default=None,
                   help='Cap number of samples (useful for smoke tests)')
    return p.parse_args()


def load_data(args: argparse.Namespace, device: str):
    import pandas as pd
    df = pd.read_csv(args.data_csv, index_col=0)
    tmp = Path('_ig_tmp_emb') / str(uuid.uuid4())
    tmp.mkdir(parents=True, exist_ok=True)
    x, y, meta, emb, mrna_idx = preprocess_input_tracks(
        df, str(tmp) + '/',
        device=device,
        with_thermo=args.with_thermo,
        fm_type=args.fm_type,
        slice_length=args.slice_length,
    )
    return (x.float().cpu(), y.float().cpu(),
            meta.float().cpu(), emb[mrna_idx].float().cpu())


def load_model(ckpt_path: Path, model_cls, x, meta, emb, device: str):
    raw = torch.load(ckpt_path, map_location='cpu')
    state_dict = raw.get('model_state', raw) if isinstance(raw, dict) else raw
    model = model_cls(
        input_channels=int(x.shape[1]),
        meta_input_dim=int(meta.shape[-1]),
        fm_seq_len=int(emb.shape[1]),
        fm_embed_dim=int(emb.shape[2]),
        num_classes=1,
        regression=True,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def run_ig(model, x, meta, emb, args: argparse.Namespace, device: str) -> dict:
    n = x.shape[0] if args.limit is None else min(args.limit, x.shape[0])
    x, meta, emb = x[:n], meta[:n], emb[:n]

    ig_funcs = {
        'x':   IntegratedGradients(lambda xi, mi, ei: model(xi, mi, ei)),
        'meta': IntegratedGradients(lambda mi, xi, ei: model(xi, mi, ei)),
        'rna':  IntegratedGradients(lambda ei, xi, mi: model(xi, mi, ei)),
    }
    inputs_map = {'x': x,    'meta': meta, 'rna': emb}
    extras_map  = {'x': (meta, emb), 'meta': (x, emb), 'rna': (x, meta)}

    attrs = {}
    for mod in args.modalities:
        parts = []
        inp   = inputs_map[mod]
        extra = extras_map[mod]
        for start in range(0, n, args.ig_batch_size):
            b_inp = inp[start:start + args.ig_batch_size].to(device)
            b_ext = tuple(t[start:start + args.ig_batch_size].to(device) for t in extra)
            attr  = ig_funcs[mod].attribute(
                b_inp,
                baselines=torch.zeros_like(b_inp),
                additional_forward_args=b_ext,
                n_steps=args.ig_steps,
            )
            parts.append(attr.detach().cpu())
            del b_inp, b_ext, attr
            torch.cuda.empty_cache()
        attrs[f'attr_{mod}'] = torch.cat(parts)

    return attrs


def main() -> None:
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_cls = MODELS[args.model]

    print('Featurizing dataset...')
    x, y, meta, emb = load_data(args, device)
    print(f'  x={tuple(x.shape)}  meta={tuple(meta.shape)}  emb={tuple(emb.shape)}')

    for ckpt in args.checkpoints:
        ckpt = Path(ckpt)
        print(f'Running IG for {ckpt.name}...')
        model = load_model(ckpt, model_cls, x, meta, emb, device)
        attrs = run_ig(model, x, meta, emb, args, device)
        out = save_dir / f'ig_{ckpt.stem}.pt'
        torch.save(attrs, out)
        print(f'  -> {out}')
        del model
        torch.cuda.empty_cache()

    print('Done.')


if __name__ == '__main__':
    main()
