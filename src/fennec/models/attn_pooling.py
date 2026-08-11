import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAPatch(nn.Module):
    """Wraps a frozen pretrained nn.Linear with a trainable low-rank delta.

    Implements:   output = base_linear(x) + scale * B(A(x))
    where base_linear is frozen and A, B are trainable low-rank matrices.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        # Keep the pretrained weights exactly as-is and freeze them.
        self.linear = base_linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        # Low-rank trainable adapter matrices.
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=6, hidden_dim=127, dropout=0.06):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        # Project embeddings up to (num_heads * hidden_dim)
        self.proj = nn.Linear(embed_dim, num_heads * hidden_dim)

        # Learn one context vector per head
        self.context_vectors = nn.Parameter(torch.randn(num_heads, hidden_dim))

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        Returns: (batch_size, num_heads * embed_dim) or (batch_size, embed_dim) after an optional final Linear
        """
        batch_size, seq_len, _ = x.shape
        
        # Project x => (batch_size, seq_len, num_heads * hidden_dim)
        proj_out = torch.tanh(self.proj(x))
        proj_out = self.dropout(proj_out)
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, hidden_dim)
        proj_out = proj_out.view(batch_size, seq_len, self.num_heads, self.hidden_dim)

        # Expand context vectors to shape (1, 1, num_heads, hidden_dim)
        context = self.context_vectors.unsqueeze(0).unsqueeze(0)
        
        # Compute attention scores (dot product between [num_heads, hidden_dim] and each token)
        # shape => (batch_size, seq_len, num_heads)
        attn_scores = (proj_out * context).sum(dim=-1)

        # Softmax over seq_len
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, num_heads)

        # Weight the original embeddings (need to broadcast attn_weights)
        # Expand attn_weights => (batch_size, seq_len, num_heads, 1)
        attn_weights = attn_weights.unsqueeze(-1)

        # Also expand x => (batch_size, seq_len, 1, embed_dim)
        x_expanded = x.unsqueeze(2)

        # Weighted sum across sequence
        # => (batch_size, num_heads, embed_dim)
        pooled = (x_expanded * attn_weights).sum(dim=1)

        # Option 1: return a flattened vector of shape (batch_size, num_heads * embed_dim)
        pooled = pooled.view(batch_size, self.num_heads * x.shape[-1])
        
        # Option 2: project back down to (batch_size, embed_dim) with a final linear
        # pooled = self.out_proj(pooled)
        
        return pooled
