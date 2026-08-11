"""TCN_Orthrus_LoRA — TCN_FM_multi_head with online Orthrus inference + adapter.

During the *split* step:  X_emb is saved as (N, seq_len, 4) float, the raw
4-track one-hot encoding of the mRNA slice (A/C/G/T channels, no U→T needed
since seq_to_oh already writes T).

During *training*:  this model loads Orthrus once, freezes all its parameters,
and runs it inside the forward pass.  A trainable bottleneck adapter
  (D → adapter_r → D, residual connection)
is inserted after the pooled Orthrus representation, allowing the embedding
space to be task-adapted without touching the frozen backbone.  The adapter
is jointly trained with the TCN and meta branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_pooling import MultiHeadAttentionPooling
from .tcn_module import TCNResidualBlock1d


class _BottleneckAdapter(nn.Module):
    """Lightweight residual adapter:  x + W_up(act(W_down(x)))."""

    def __init__(self, dim: int, r: int):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up   = nn.Linear(r, dim)
        self.act  = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class TCNResNNWithOrthrus_LoRA(nn.Module):
    """TCN + meta-feature + Orthrus with trainable bottleneck adapter.

    Parameters
    ----------
    adapter_r : int
        Bottleneck dimension for the residual adapter applied after Orthrus.
    fm_seq_len, fm_embed_dim : int
        Accepted for API compatibility; the true Orthrus output dim is
        discovered via a tiny dummy forward pass in __init__.
    """

    def __init__(
        self,
        input_channels: int = 25,
        resblock_channels: int = 256,
        kernel_1: int = 3,
        kernel_2: int = 5,
        kernel_3: int = 7,
        meta_input_dim: int = 25,
        fm_seq_len: int = 150,    # ignored — kept for API compatibility
        fm_embed_dim: int = 512,  # default; overridden below from model
        num_classes: int = 1,
        regression: bool = True,
        lstm_hidden_dim: int = 128,
        adapter_r: int = 64,
    ):
        super().__init__()

        # ── Load Orthrus and freeze all parameters ───────────────────────────
        try:
            from transformers import AutoModel
            from orthrus.gk_utils import seq_to_oh as _seq_to_oh  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers / orthrus packages not found. "
                "Activate the orthrus conda environment."
            ) from e

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        orthrus = AutoModel.from_pretrained(
            "antichronology/orthrus-4-track", trust_remote_code=True
        ).to(_device)
        for param in orthrus.parameters():
            param.requires_grad_(False)
        orthrus.eval()
        self.orthrus = orthrus

        # Discover the representation dimension via a tiny dummy forward.
        with torch.no_grad():
            _dummy_oh = torch.zeros(1, 4, 100, device=_device)  # (1, 4, L)
            _dummy_len = torch.tensor([100], device=_device)
            _dummy_rep = orthrus.representation(_dummy_oh, _dummy_len)
        ORTHRUS_DIM = _dummy_rep.shape[-1]

        # ── Trainable adapter ─────────────────────────────────────────────────
        self.adapter = _BottleneckAdapter(ORTHRUS_DIM, r=adapter_r)

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

        # ── FM projection MLP (takes adapted Orthrus representation) ─────────
        # seq_len=1 for Orthrus, so no attn pooling needed — just fm_layers.
        self.fm_layers = nn.Sequential(
            nn.Linear(ORTHRUS_DIM, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 128),         nn.ELU(), nn.Dropout(0.3),
        )

        # ── Combined head ────────────────────────────────────────────────────
        fc_input_dim = lstm_hidden_dim + 64 + 128
        self.combined_fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(0.3),
        )
        self._combined_bn_layers = [m for m in self.combined_fc if isinstance(m, nn.BatchNorm1d)]
        self.output = nn.Linear(64, 1) if regression else nn.Linear(64, num_classes)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, meta_features, one_hot_input):
        """
        x             : (N, input_channels, seq_len)
        meta_features : (N, meta_input_dim)
        one_hot_input : (N, seq_len, 4) float — 4-track one-hot mRNA slice
        """
        # Orthrus expects (N, 4, seq_len).
        oh = one_hot_input.permute(0, 2, 1)                      # (N, 4, L)
        lengths = torch.tensor([oh.shape[2]], device=oh.device)

        # Frozen Orthrus forward — no grad through backbone.
        with torch.no_grad():
            rep = self.orthrus.representation(oh, lengths)        # (N, D)

        # Trainable adapter on top of frozen representation.
        adapted = self.adapter(rep)                               # (N, D), grads here

        # siRNA TCN branch
        out1 = self.resblock1(x)
        out2 = self.resblock2(out1)
        out3 = self.resblock3(out2)
        combined_seq = torch.cat([out1, out2, out3], dim=1).permute(0, 2, 1)
        cnn_out = self.cnn_mlp(combined_seq).mean(dim=1)

        # Meta branch
        meta = self._forward_with_bn_guard(self.meta_layers, self._meta_bn_layers, meta_features)

        # FM branch
        fm_out = self.fm_layers(adapted)

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
