import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attn_pooling import MultiHeadAttentionPooling


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



class ResidualMLPBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        """
        A residual block that uses an MLP instead of convolutions.
        Expects input of shape (N, in_channels, seq_length).
        """
        super(ResidualMLPBlock1d, self).__init__()
        # This MLP is applied per time-step.
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
            nn.Dropout(p=dropout_p)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x shape: (N, in_channels, seq_length)
        # Transpose to (N, seq_length, in_channels) to apply the MLP on the feature dim.
        x_perm = x.transpose(1, 2)
        out = self.fc(x_perm)  # (N, seq_length, out_channels)
        shortcut = self.shortcut(x_perm)  # (N, seq_length, out_channels)
        out = out + shortcut
        out = F.elu(out)
        # Transpose back to (N, out_channels, seq_length)
        out = out.transpose(1, 2)
        return out


class NNWithMetaAndEmbeddings(nn.Module):
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
        use_lora=True,  # If True, use LoRA layers; if False, use standard Linear layers
        use_attention_pooling=True,  # If True, use attention pooling on RNA-FM embeddings instead of simple mean
        attn_num_heads=4, # these were all optimized using a random search
        attn_dim=191,
        attn_dropout=0.02,
    ):
        super(NNWithMetaAndEmbeddings, self).__init__()

        # 1) MLP-based residual blocks for sequence data (x)
        self.mlpblock1 = ResidualMLPBlock1d(
            in_channels=input_channels,
            out_channels=resblock_channels,
            dropout_p=0.3,
        )
        self.mlpblock2 = ResidualMLPBlock1d(
            in_channels=resblock_channels,
            out_channels=resblock_channels,
            dropout_p=0.3,
        )
        self.mlpblock3 = ResidualMLPBlock1d(
            in_channels=resblock_channels,
            out_channels=resblock_channels,
            dropout_p=0.3,
        )

        # 2) LSTM to process final features
        # Expecting an input size of (resblock_channels * 3) per time-step.
        # self.lstm = nn.LSTM(
        #     input_size=resblock_channels * 3,
        #     hidden_size=lstm_hidden_dim,
        #     batch_first=True,
        #     bidirectional=lstm_bidirectional,
        #     num_layers=lstm_layers
        # )
        self.cnn_mlp = nn.Sequential(
        nn.Linear(resblock_channels*3, lstm_hidden_dim),
        nn.ELU(),
        nn.Dropout(p=0.3),
        nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
        nn.ELU(),
        nn.Dropout(p=0.2),
        )

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
            self.attn_pool = MultiHeadAttentionPooling(fm_embed_dim)
            fm_in_dim = fm_embed_dim * self.attn_pool.num_heads
        else:
            fm_in_dim = fm_embed_dim

        self.fm_layers = nn.Sequential(
            LoRALinear(fm_in_dim, 64, r=lora_r, alpha=lora_alpha),
            nn.ELU(),
            nn.Dropout(p=0.3),
            LoRALinear(64, 128, r=lora_r, alpha=lora_alpha),
            nn.ELU(),
            nn.Dropout(p=0.3),
        )

        # 5) Final combination + MLP
        fc_input_dim = lstm_hidden_dim + 64 + 128
        self.combined_fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(p=0.3)
        )
        self._combined_bn_layers = [m for m in self.combined_fc if isinstance(m, nn.BatchNorm1d)]

        if regression:
            self.output = nn.Linear(64, 1)  # For regression
        else:
            self.output = nn.Linear(64, num_classes)  # For classification

    def forward(self, x, meta_features, rna_fm_embeddings):
        # x shape: (N, input_channels, seq_length)
        # meta_features shape: (N, meta_input_dim)
        # rna_fm_embeddings shape: (N, fm_seq_len, fm_embed_dim)

        # 1) Process sequence data using MLP blocks
        out1 = self.mlpblock1(x)
        out2 = self.mlpblock2(out1)
        out3 = self.mlpblock3(out2)

        # Use adaptive average pooling to get a fixed temporal dimension (e.g., 3)
        # out1_ds = F.adaptive_avg_pool1d(out1, output_size=adaptive_avg_output)
        # out2_ds = F.adaptive_avg_pool1d(out2, output_size=adaptive_avg_output)
        # out3_ds = F.adaptive_avg_pool1d(out3, output_size=adaptive_avg_output)
        #combined_out = torch.cat([out1_ds, out2_ds, out3_ds], dim=1)
        combined_out = torch.cat([out1, out2, out3], dim=1)  # shape: (N, resblock_channels*3, 3)
        combined_seq = combined_out.permute(0, 2, 1)  # shape: (N, 3, resblock_channels*3)

        # 2) LSTM processing
        # lstm_out, (h_n, c_n) = self.lstm(combined_seq)
        # lstm_final = h_n[-1]
        lstm_final = self.cnn_mlp(combined_seq).mean(dim=1)

        # 3) Meta-data processing
        meta = self._forward_with_bn_guard(self.meta_layers, self._meta_bn_layers, meta_features)

        # 4) RNA fm-embeddings processing
        if self.use_attention_pooling:
            fm_pooled = self.attn_pool(rna_fm_embeddings)
        else:
            fm_pooled = rna_fm_embeddings.mean(dim=1)
        fm_out = self.fm_layers(fm_pooled)

        # 5) Combine and final MLP
        combined = torch.cat((lstm_final, meta, fm_out), dim=1)
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
