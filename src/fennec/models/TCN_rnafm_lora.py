"""TCN_RNAFM_LoRA — TCN_FM_multi_head with online RNA-FM inference + LoRA.

During the *split* step:  X_emb is saved as (N, 1, seq_len) float, where the
values are RNA-FM integer token IDs (CLS + slice_length nt + EOS).

During *training*:  this model loads RNA-FM once, patches Q and V projections
in every transformer layer with trainable LoRA matrices (base weights frozen),
and runs the FM online in each forward pass.  The resulting per-token
embeddings feed into the same MultiHeadAttentionPooling → fm_layers path as
TCN_FM_multi_head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_pooling import MultiHeadAttentionPooling, LoRAPatch
from .tcn_module import TCNResidualBlock1d


class TCNResNNWithRNAFM_LoRA(nn.Module):
    """TCN + meta-feature + RNA-FM with LoRA fine-tuning.

    Parameters
    ----------
    lora_r : int
        Rank of each LoRA adapter.  Smaller = fewer trainable params.
    lora_alpha : float
        LoRA scaling factor (effective scale = alpha / r).
    lora_layers : list[int] | None
        Which RNA-FM transformer layer indices to patch.  None = all 12.
    fm_seq_len, fm_embed_dim : int
        Accepted for API compatibility with train.py's build_model() call;
        the actual FM output dims are discovered from the loaded model.
    """

    def __init__(
        self,
        input_channels: int = 25,
        resblock_channels: int = 256,
        kernel_1: int = 3,
        kernel_2: int = 5,
        kernel_3: int = 7,
        meta_input_dim: int = 25,
        # fm_seq_len / fm_embed_dim are ignored — determined from RNA-FM itself
        fm_seq_len: int = 52,
        fm_embed_dim: int = 640,
        num_classes: int = 1,
        regression: bool = True,
        lstm_hidden_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: float = 1.0,
        lora_layers=None,
        attn_num_heads: int = 4,
        attn_dim: int = 191,
        attn_dropout: float = 0.02,
    ):
        super().__init__()

        # ── Load RNA-FM and apply LoRA patches ───────────────────────────────
        try:
            import fm as rna_fm_pkg
        except ImportError as e:
            raise ImportError("RNA-FM package not found.  Install it with: pip install fair-esm") from e

        rna_fm_model, alphabet = rna_fm_pkg.pretrained.rna_fm_t12()
        # Stay in eval mode — dropout disabled, only LoRA params get gradients.
        rna_fm_model.eval()

        FM_EMBED_DIM = 640  # RNA-FM t12 fixed dimension
        n_layers = len(rna_fm_model.layers)
        if lora_layers is None:
            lora_layers = list(range(n_layers))

        for i in lora_layers:
            attn = rna_fm_model.layers[i].self_attn
            attn.q_proj = LoRAPatch(attn.q_proj, r=lora_r, alpha=lora_alpha)
            attn.v_proj = LoRAPatch(attn.v_proj, r=lora_r, alpha=lora_alpha)

        # Freeze everything that is NOT a LoRA adapter.
        for name, param in rna_fm_model.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad_(False)

        self.rna_fm = rna_fm_model

        # ── Sequence (siRNA) processing ──────────────────────────────────────
        self.resblock1 = TCNResidualBlock1d(input_channels, resblock_channels,
                                            kernel_1, dilation=1, dropout_p=0.3)
        self.resblock2 = TCNResidualBlock1d(resblock_channels, resblock_channels,
                                            kernel_2, dilation=2, dropout_p=0.3)
        self.resblock3 = TCNResidualBlock1d(resblock_channels, resblock_channels,
                                            kernel_3, dilation=4, dropout_p=0.3)
        self.cnn_mlp = nn.Sequential(
            nn.Linear(resblock_channels * 3, lstm_hidden_dim), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),       nn.ELU(), nn.Dropout(0.2),
        )

        # ── Meta-feature MLP ─────────────────────────────────────────────────
        self.meta_layers = nn.Sequential(
            nn.Linear(meta_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 64),             nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 64),             nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.1),
        )
        self._meta_bn_layers = [m for m in self.meta_layers if isinstance(m, nn.BatchNorm1d)]

        # ── FM embedding pooling ─────────────────────────────────────────────
        self.attn_pool = MultiHeadAttentionPooling(FM_EMBED_DIM,
                                                   hidden_dim=attn_dim,
                                                   num_heads=attn_num_heads,
                                                   dropout=attn_dropout)
        fm_in_dim = FM_EMBED_DIM * attn_num_heads

        self.fm_layers = nn.Sequential(
            nn.Linear(fm_in_dim, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 128),       nn.ELU(), nn.Dropout(0.3),
        )

        # ── Combined head ────────────────────────────────────────────────────
        fc_input_dim = lstm_hidden_dim + 64 + 128
        self.combined_fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.3),
        )
        self._combined_bn_layers = [m for m in self.combined_fc if isinstance(m, nn.BatchNorm1d)]
        self.output = nn.Linear(64, 1) if regression else nn.Linear(64, num_classes)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, meta_features, token_input):
        """
        x            : (N, input_channels, seq_len)
        meta_features: (N, meta_input_dim)
        token_input  : (N, 1, seq_len_tokens) float — token IDs stored as float
        """
        # Recover integer token IDs: (N, seq_len_tokens)
        token_ids = token_input.squeeze(1).long()

        # Run RNA-FM with LoRA (only lora_A/lora_B params carry gradients).
        results = self.rna_fm(token_ids, repr_layers=[12])
        mrna_emb = results["representations"][12]  # (N, seq_len_tokens, 640)

        # siRNA TCN branch
        out1 = self.resblock1(x)
        out2 = self.resblock2(out1)
        out3 = self.resblock3(out2)
        combined_seq = torch.cat([out1, out2, out3], dim=1).permute(0, 2, 1)
        cnn_out = self.cnn_mlp(combined_seq).mean(dim=1)

        # Meta branch
        meta = self._forward_with_bn_guard(self.meta_layers, self._meta_bn_layers, meta_features)

        # FM branch
        fm_pooled = self.attn_pool(mrna_emb)
        fm_out = self.fm_layers(fm_pooled)

        # Combine
        combined = torch.cat((cnn_out, meta, fm_out), dim=1)
        combined = self._forward_with_bn_guard(self.combined_fc, self._combined_bn_layers, combined)
        return self.output(combined)

    def _forward_with_bn_guard(self, module, bn_layers, tensor):
        if self.training and tensor.shape[0] == 1 and bn_layers:
            original_states = [bn.training for bn in bn_layers]
            try:
                for bn in bn_layers:
                    bn.eval()
                return module(tensor)
            finally:
                for bn, state in zip(bn_layers, original_states):
                    if state:
                        bn.train()
        return module(tensor)
