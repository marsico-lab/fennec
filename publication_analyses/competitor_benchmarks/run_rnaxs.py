#!/usr/bin/env python
"""Run the modified RNAxs Perl script on an mRNA FASTA file.

RNAxs (Tafer et al., 2008) predicts siRNA target sites from mRNA sequence.
RNAxs_update.pl is a modified version that supports variable-length siRNAs
via the -k flag (original hardcoded 19-nt).

Usage:
    python run_rnaxs.py --input mRNA.fasta --length 23
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Run RNAxs on an mRNA FASTA")
    p.add_argument('--input', '-i', required=True, help='mRNA FASTA file')
    p.add_argument('--length', '-k', type=int, default=23, help='siRNA length (default: 23)')
    p.add_argument('--top-n', '-n', type=int, default=10000, help='Max candidates to return')
    p.add_argument('--script', default=str(Path(__file__).parent / 'RNAxs_update.pl'),
                   help='Path to RNAxs_update.pl')
    args = p.parse_args()

    cmd = [
        "perl", args.script,
        "-s", args.input,
        "-n", str(args.top_n),
        "-k", str(args.length),
        "-t", "0", "-u", "0", "-e", "0", "-a", "0", "-f", "0", "-g", "0",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
