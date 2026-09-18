"""
Environment smoke-test for fennec.

Tests that all required packages import correctly and that the core
featurization pipeline (one-hot encoding + thermo features) runs end-to-end
on a small subset of training data, without requiring a GPU or downloading
foundation model weights.

Run with:
    python tests/test_env.py
or through pytest:
    pytest tests/test_env.py -v
"""

import sys
import os
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(msg):
    print(f"  ✓ {msg}")

def fail(msg, exc=None):
    print(f"  ✗ {msg}")
    if exc:
        traceback.print_exc()
    return False


# ---------------------------------------------------------------------------
# 1. Core scientific stack
# ---------------------------------------------------------------------------

def test_core_imports():
    section("Core scientific packages")
    import numpy as np; ok(f"numpy {np.__version__}")
    import pandas as pd; ok(f"pandas {pd.__version__}")
    import scipy; ok(f"scipy {scipy.__version__}")
    import sklearn; ok(f"scikit-learn {sklearn.__version__}")
    import matplotlib; ok(f"matplotlib {matplotlib.__version__}")
    import seaborn; ok(f"seaborn {seaborn.__version__}")
    import tqdm; ok(f"tqdm {tqdm.__version__}")
    import click; ok(f"click {click.__version__}")


def test_torch():
    section("PyTorch")
    import torch
    ok(f"torch {torch.__version__}")
    ok(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ok(f"CUDA device: {torch.cuda.get_device_name(0)}")


def test_transformers():
    section("HuggingFace transformers")
    import transformers
    ok(f"transformers {transformers.__version__}")


# ---------------------------------------------------------------------------
# 2. Bioinformatics packages
# ---------------------------------------------------------------------------

def test_bioinformatics():
    section("Bioinformatics packages")
    from Bio.Seq import reverse_complement
    from Bio.SeqUtils import gc_fraction
    ok("biopython (Bio.Seq, Bio.SeqUtils)")

    import edlib
    result = edlib.align("ACGT", "AACGT", mode="HW", task="locations")
    assert result["editDistance"] == 0
    ok(f"edlib (alignment test passed)")

    from primer3.bindings import calc_hairpin, calc_heterodimer, calc_homodimer
    dg = calc_hairpin("AUGCAUGCAUGCAUGC", mv_conc=200, dv_conc=50,
                      dntp_conc=50, dna_conc=50, temp_c=37).dg / 1000
    ok(f"primer3 (hairpin dG = {dg:.2f} kcal/mol)")


# ---------------------------------------------------------------------------
# 3. BioInUtils (thermo features)
# ---------------------------------------------------------------------------

def test_bioinutils():
    section("BioInUtils (thermo features)")
    from bioinutils.seqfeatures.utils import (
        calc_homo, calc_hetero, calc_tm,
        calculate_lc, homopolymer_counter, calc_3_prime_stability,
    )
    ok("bioinutils.seqfeatures.utils")

    seq = "ATGCATGCATGCATGC"  # DNA (primer3 requires ACGT, not AUGC)
    tm = calc_tm(seq)
    ok(f"calc_tm('{seq}') = {tm:.1f} °C")

    lc = calculate_lc(seq)
    ok(f"calculate_lc('{seq}') = {lc:.3f}")
    seq = "AUGCAUGCAUGCAUGC"  # RNA for rfold

    from bioinutils.vienna import rfold
    result = rfold({"test": "GGGAAACCC"})
    ok(f"rfold (ViennaRNA): dot-bracket = {result['test'].dot_bracket}")


# ---------------------------------------------------------------------------
# 4. Orthrus gk_utils
# ---------------------------------------------------------------------------

def test_orthrus_gk_utils():
    section("Orthrus gk_utils")
    from orthrus.gk_utils import seq_to_oh
    import numpy as np

    seq = "ACGT"
    oh = seq_to_oh(seq)
    expected = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    assert np.array_equal(oh, expected), f"seq_to_oh mismatch: {oh}"
    ok(f"seq_to_oh('{seq}') correct shape {oh.shape}")


# ---------------------------------------------------------------------------
# 5. RNA-FM
# ---------------------------------------------------------------------------

def test_rna_fm():
    section("RNA-FM")
    try:
        import fm
        ok("fm module importable (RNA-FM T12 weights download on first use)")
    except ImportError as e:
        fail("rna-fm not installed", e)


# ---------------------------------------------------------------------------
# 6. Fennec package
# ---------------------------------------------------------------------------

def test_fennec_imports():
    section("Fennec package")
    import fennec
    ok("fennec")

    from fennec.helm_converter import helm_to_fennec
    ok("fennec.helm_converter")

    from fennec.featurizer import (
        helm_one_hot_encode, helm_one_hot_decode,
        generate_feature_vector_encoding,
        apply_internal_features_condensed,
    )
    ok("fennec.featurizer")

    from fennec.preprocessing import (
        prepare_meta_features_and_one_hot,
        preprocess_input_tracks,
    )
    ok("fennec.preprocessing")

    from fennec.models import TCN_FM_multi_head
    ok("fennec.models.TCN_FM_multi_head")


# ---------------------------------------------------------------------------
# 7. End-to-end featurization (one-hot + thermo, no GPU / FM weights needed)
# ---------------------------------------------------------------------------

def test_e2e_featurization():
    section("End-to-end featurization (one-hot + thermo, CPU only)")
    import pandas as pd
    import torch
    from fennec.preprocessing import prepare_meta_features_and_one_hot
    from fennec.featurizer import apply_internal_features_condensed

    # Locate the test subset CSV (created by this test suite or use full data)
    here = os.path.dirname(os.path.abspath(__file__))
    subset_path = os.path.join(here, "..", "src", "fennec", "support_files",
                               "test_subset_100.csv")
    if not os.path.exists(subset_path):
        # Fall back to creating a tiny inline dataframe
        data = {
            "siRNA_list": ["VP u s Uf s c p u p g p Gf p c p c p g p a p g p c p a p Uf"] * 5,
            "siRNA_passenger_list": ["g s a s g p a p c p c p g p g p c p u p Cf p Gf p Uf p a p C"] * 5,
            "mRNA": ["CTACTCAGCCCCAGCGGAGGTGAAGGACGTCCTTCCCCAGGAGCCGACTGGCCAATCACAGCCCAGGTCCAGACCAGCCCTGGAGGCCCGGG"] * 5,
            "mRNA_remaining_pct": [82.0, 55.0, 30.0, 90.0, 10.0],
            "gene_target_symbol_name": ["APOE"] * 5,
            "gene_target_species": ["Homo sapiens"] * 5,
            "cell_line_donor": ["HeLa"] * 5,
            "siRNA_concentration": [10.0] * 5,
            "Transfection_method": ["Lipofectamine"] * 5,
            # Required by generate_feature_vector_encoding; without it the meta
            # frame keeps an object-dtype column and torch.tensor() rejects it.
            "Duration_after_transfection_h": [24.0] * 5,
            # Unmodified guide (seq_list_to_seq of siRNA_list above);
            # apply_internal_features_condensed asserts on this column.
            "siRNA": ["UUCUGGCCGAGCAU"] * 5,
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(subset_path).head(20)

    ok(f"Loaded {len(df)} rows")

    # One-hot + metadata encoding
    X, y, X_meta = prepare_meta_features_and_one_hot(df, with_passenger=True)
    assert X.shape[0] == len(df), f"X row count mismatch: {X.shape}"
    assert y.shape[0] == len(df)
    assert X_meta.shape[0] == len(df)
    ok(f"One-hot siRNA shape:  {tuple(X.shape)}")
    ok(f"Metadata shape:       {tuple(X_meta.shape)}")
    ok(f"Targets shape:        {tuple(y.shape)}, range [{y.min():.1f}, {y.max():.1f}]")

    # Thermo features
    thermo_df = apply_internal_features_condensed(df.head(5))
    thermo_cols = [c for c in thermo_df.columns if c not in df.columns]
    ok(f"Thermo features added: {thermo_cols}")

    # Verify model can do a forward pass (CPU, random embeddings)
    from fennec.models import TCN_FM_multi_head
    import torch

    batch = 4
    seq_channels  = X.shape[1]      # 42 (guide+passenger, 21 each)
    seq_len        = X.shape[2]      # sequence length
    meta_dim       = X_meta.shape[1]
    emb_len        = 102             # RNA-FM token positions
    emb_dim        = 640             # RNA-FM embedding dim

    model = TCN_FM_multi_head(
        input_channels=seq_channels,
        meta_input_dim=meta_dim,
        fm_seq_len=emb_len,
        fm_embed_dim=emb_dim,
    )
    model.eval()

    x_in   = X[:batch].float()
    xm_in  = X_meta[:batch].float()
    xe_in  = torch.randn(batch, emb_len, emb_dim)

    with torch.no_grad():
        out = model(x_in, xm_in, xe_in)

    assert out.shape == (batch, 1), f"Unexpected output shape: {out.shape}"
    ok(f"Model forward pass OK — output shape {tuple(out.shape)}, "
       f"values [{out.min().item():.2f}, {out.max().item():.2f}]")


def test_transfection_unknowns_map_to_not_specified():
    import pandas as pd

    from fennec.featurizer import generate_feature_vector_encoding

    df = pd.DataFrame(
        {
            "gene_target_species": ["Homo sapiens", "Homo sapiens"],
            "cell_line_donor": ["HeLa", "HeLa"],
            "siRNA_concentration": [10.0, 10.0],
            "Transfection_method": ["Unknown_transfection", "Some new reagent"],
            "Duration_after_transfection_h": [24.0, 24.0],
        }
    )

    encoded = generate_feature_vector_encoding(df)

    assert encoded["Not specified"].tolist() == [1, 1]
    assert encoded["Unknown_transfection"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_core_imports,
        test_torch,
        test_transformers,
        test_bioinformatics,
        test_bioinutils,
        test_orthrus_gk_utils,
        test_rna_fm,
        test_fennec_imports,
        test_e2e_featurization,
    ]

    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            traceback.print_exc()

    section("Summary")
    passed = len(tests) - len(failures)
    print(f"  {passed}/{len(tests)} test groups passed")
    if failures:
        for name, exc in failures:
            print(f"  ✗ {name}: {exc}")
        sys.exit(1)
    else:
        print("  All checks passed!")
