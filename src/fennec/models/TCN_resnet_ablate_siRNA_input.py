import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attn_pooling import MultiHeadAttentionPooling
from .tcn_module import TCNResidualBlock1d


class TCNResNNWithMetaAndEmbeddings(nn.Module):
    def __init__(
        self,
        input_channels=1,
        resblock_channels=256,
        kernel_1=3,
        kernel_2=5,
        kernel_3=7,
        meta_input_dim=25,
        fm_embed_dim=640,
        fm_seq_len=102,
        num_classes=1,
        regression=True,
        adaptive_avg_output=3,
        lstm_hidden_dim=128,
        lstm_bidirectional=False,
        lstm_layers=1,
        lora_r=128,  # LoRA rank
        lora_alpha=5.0,  # LoRA scaling factor
        use_lora=False,  # If True, use LoRA layers; if False, use standard Linear layers
        use_attention_pooling=True,  # If True, use attention pooling on RNA-FM embeddings instead of simple mean
        attn_num_heads=4,  # these were all optimized using a random search
        attn_dim=191,
        attn_dropout=0.02,
    ):
        super(TCNResNNWithMetaAndEmbeddings, self).__init__()
        # TCN and LSTM completely removed from here

        # 3) Meta-data processing
        self.meta_layers = nn.Sequential( # Dropout here before Dense layer?
            nn.Linear(meta_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(p=0.1),
        )
        self._meta_bn_layers = [m for m in self.meta_layers if isinstance(m, nn.BatchNorm1d)]

        # 4) RNA fm-embeddings processing
        # Optionally use attention pooling instead of a simple mean.
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attn_pool = MultiHeadAttentionPooling(fm_embed_dim, 
                                                       hidden_dim=attn_dim, 
                                                       num_heads=attn_num_heads, 
                                                       dropout=attn_dropout)
            fm_in_dim = fm_embed_dim * self.attn_pool.num_heads
        else:
            fm_in_dim = fm_embed_dim

        self.fm_layers = nn.Sequential(
            nn.Linear(fm_in_dim, 128),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(p=0.3),
        )

        # 5) Final combination + MLP
        fc_input_dim = 64 + 128
        self.combined_fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(p=0.3)
        )
        self._combined_bn_layers = [m for m in self.combined_fc if isinstance(m, nn.BatchNorm1d)]

        if regression:
            self.output = nn.Linear(64, 1)  # For regression
        else:
            self.output = nn.Linear(64, num_classes)  # For classification

    def forward(self, x, meta_features, mrna_embeddings, adaptive_avg_output=3):
        # x shape: (N, input_channels, seq_length)
        # meta_features shape: (N, meta_input_dim)
        # mrna_embeddings shape: (N, fm_seq_len, fm_embed_dim)

        # 1) ResNet (Conv) blocks
        # removed

        # 2) LSTM processing
        # removed

        # 3) Meta-data processing
        meta = self._forward_with_bn_guard(self.meta_layers, self._meta_bn_layers, meta_features)

        # 4) RNA fm-embeddings processing
        if self.use_attention_pooling:
            fm_pooled = self.attn_pool(mrna_embeddings)
        else:
            fm_pooled = mrna_embeddings.mean(dim=1)
        fm_out = self.fm_layers(fm_pooled)

        # 5) Combine and final MLP
        combined = torch.cat((meta, fm_out), dim=1)
        combined = self._forward_with_bn_guard(self.combined_fc, self._combined_bn_layers, combined)
        combined = self.output(combined)
        return combined

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