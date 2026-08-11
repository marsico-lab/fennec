"""Smoke test: load a real trained checkpoint and run a forward pass.

Verifies that the model class definitions in this repo are compatible with
trained weights and that a basic inference loop works end-to-end on dummy input.

Set the checkpoint path via env var:

    FENNEC_TEST_WEIGHT=/path/to/checkpoint.pt python tests/test_load_trained_weight.py

If FENNEC_TEST_WEIGHT is not set or empty, the test is skipped.

The default weight was trained with:
  - Model: TCN_FM_multi_head (standard TCN + FM + multi-head attention pool)
  - Dataset: paper's v1.3.1.6 publication set
  - FM: Orthrus 4-track triple UTR embeddings (5'UTR + CDS + 3'UTR, 512 each)
  - Slice: 100bp, log1p target transform, thermo features included

The constructor kwargs below were reverse-engineered from state_dict shapes.
"""

import os
import sys
import inspect
import torch

# Kwargs that match the `_standard_triple_100bp_thermo_log1p_v132` family
KWARGS = dict(
    input_channels=54,
    resblock_channels=256,
    meta_input_dim=61,
    fm_embed_dim=1536,
    attn_num_heads=4,
    attn_dim=191,
)


def main() -> int:
    from fennec.models import MODELS

    weight_path = os.environ.get("FENNEC_TEST_WEIGHT", "")
    if not weight_path:
        print("[skip] FENNEC_TEST_WEIGHT not set; skipping weight-load test")
        return 0
    if not os.path.exists(weight_path):
        print(f"[skip] weight not found at {weight_path}")
        print("      set FENNEC_TEST_WEIGHT to a valid .pt state_dict to run")
        return 0

    print(f"[weight] {weight_path}")
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    print(f"[weight] state_dict with {len(sd)} keys")

    model_cls = MODELS["TCN_FM_multi_head"]
    model = model_cls(**KWARGS)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    print("[load] strict=True clean")

    model.eval()
    B = 2
    x_seq = torch.zeros(B, KWARGS["input_channels"], 100)
    meta = torch.zeros(B, KWARGS["meta_input_dim"])
    rna = torch.zeros(B, 102, KWARGS["fm_embed_dim"])

    print(f"[forward] signature: {inspect.signature(model.forward)}")
    with torch.no_grad():
        out = model(x_seq, meta, rna)
    print(f"[forward] output shape: {tuple(out.shape)}")
    print(f"[forward] output value (all-zero input): {out.view(-1)[0].item():.4f}")

    print("\nfennec_pub can load real trained weights and run inference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
