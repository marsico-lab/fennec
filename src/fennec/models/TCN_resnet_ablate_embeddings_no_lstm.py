import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn_pooling import MultiHeadAttentionPooling
from .tcn_module import TCNResidualBlock1d


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) linear layer.
    
    Implements: output = W*x + (alpha/r) * B*A*x
    where W is frozen, A and B are trainable low-rank matrices.
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        
        # Base linear layer parameters (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weight with Xavier/Glorot
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Low-rank matrices for adaptation
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        
        # Freeze the base linear layer
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False
    
    def forward(self, x):
        # Base output: x @ W^T + b
        base_out = F.linear(x, self.weight, self.bias)
        
        # LoRA adaptation: x @ A^T @ B^T with scaling
        lora_out = (x @ self.A.T) @ self.B.T
        
        return base_out + (self.alpha / self.r) * lora_out


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
        lstm_hidden_dim=128,
        lstm_bidirectional=False,
        lstm_layers=1,
        lora_r=128,
        lora_alpha=5.0,
        use_attention_pooling=True,
        attn_num_heads=4,
        attn_dim=191,
        attn_dropout=0.02,
    ):
        super(TCNResNNWithMetaAndEmbeddings, self).__init__()

        # 1) Residual blocks for sequence data (x)
        self.resblock1 = TCNResidualBlock1d(
            in_channels=input_channels,
            out_channels=resblock_channels,
            kernel_size=kernel_1,
            dilation=1,
            dropout_p=0.3,
        )
        self.resblock2 = TCNResidualBlock1d(
            in_channels=resblock_channels,
            out_channels=resblock_channels,
            kernel_size=kernel_2,
            dilation=2,
            dropout_p=0.3,
        )
        self.resblock3 = TCNResidualBlock1d(
            in_channels=resblock_channels,
            out_channels=resblock_channels,
            kernel_size=kernel_3,
            dilation=4,
            dropout_p=0.3,
        )

        # 2) MLP replacing LSTM (ablation: no LSTM)
        self.cnn_mlp = nn.Sequential(
            nn.Linear(resblock_channels * 3, lstm_hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.2),
        )

        # 3) Meta-data processing
        self.meta_layers = nn.Sequential(
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

        # 4) No FM embeddings branch (ablated)

        # 5) Final combination + MLP
        fc_input_dim = lstm_hidden_dim + 64
        self.combined_fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(p=0.3)
        )
        self._combined_bn_layers = [m for m in self.combined_fc if isinstance(m, nn.BatchNorm1d)]

        if regression:
            self.output = nn.Linear(64, 1)
        else:
            self.output = nn.Linear(64, num_classes)

    def forward(self, x, meta_features, rna_fm_embeddings):
        # x shape: (N, input_channels, seq_length)
        # meta_features shape: (N, meta_input_dim)
        # rna_fm_embeddings: accepted but not used (embeddings ablated)

        # 1) ResNet (Conv) blocks
        out1 = self.resblock1(x)
        out2 = self.resblock2(out1)
        out3 = self.resblock3(out2)

        combined_out = torch.cat([out1, out2, out3], dim=1)
        combined_seq = combined_out.permute(0, 2, 1)

        # 2) MLP in place of LSTM
        lstm_final = self.cnn_mlp(combined_seq).mean(dim=1)

        # 3) Meta-data processing
        meta = self._forward_with_bn_guard(self.meta_layers, self._meta_bn_layers, meta_features)

        # 4) Combine and final MLP (no embeddings)
        combined = torch.cat((lstm_final, meta), dim=1)
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
