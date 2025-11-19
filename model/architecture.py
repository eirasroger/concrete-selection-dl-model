import torch
import torch.nn as nn

class SetTransformerBlock(nn.Module):
    """One self-attention block with pre-norm, residuals, and FFN."""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        # x: (B, S, D)
        # 1) Self-attention with pre-norm
        x2 = self.norm1(x)
        attn_mask = ~mask          # True where padding
        attn_out, _ = self.attn(x2, x2, x2,
                                key_padding_mask=attn_mask)
        x = x + attn_out           # residual

        # 2) Feed-forward with pre-norm
        x2 = self.norm2(x)
        x = x + self.ffn(x2)       # residual
        return x

class SetRanker(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 hidden_dims: list[int],
                 dropout: float,
                 num_heads: int = 2,
                 num_attn_blocks: int = 2,
                 ffn_factor: float = 4.0):
        """
        feat_dim: size of each input feature vector
        hidden_dims: e.g. [512, 256, 128]
        dropout: dropout rate
        num_heads: heads in multi-head attention
        num_attn_blocks: how many transformer blocks to stack
        ffn_factor: inner dim factor for feed-forward (ffn_dim = final_dim * ffn_factor)
        """
        super().__init__()
        assert isinstance(hidden_dims, list) and len(hidden_dims) >= 1

        # Per-element MLP encoder
        layers = []
        in_dim = feat_dim

        for h in hidden_dims:
            layers += [
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(dropout)
            ]
            in_dim = h


        # remove last Dropout
        if isinstance(layers[-1], nn.Dropout):
            layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        # Stack of Transformer‐style self-attention blocks
        final_dim = hidden_dims[-1]
        ffn_dim = int(final_dim * ffn_factor)
        self.attn_blocks = nn.ModuleList([
            SetTransformerBlock(embed_dim=final_dim,
                                num_heads=num_heads,
                                ffn_dim=ffn_dim,
                                dropout=dropout)
            for _ in range(num_attn_blocks)
        ])

        # Scoring head: a small MLP after concatenation
        self.scorer = nn.Sequential(
            nn.Linear(2 * final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, 1)
        )

    def forward(self, X, mask):
        """
        X: (B, S, F) feature tensor
        mask: (B, S) boolean mask (True = valid, False = pad)
        """
        B, S, F = X.shape

        # — encode each element independently
        h = self.encoder(X.view(B * S, F))   # (B*S, final_dim)
        h = h.view(B, S, -1)                 # (B, S, final_dim)

        # — apply each self-attention block
        for block in self.attn_blocks:
            h = block(h, mask)

        # — zero out padded positions
        h = h * mask.unsqueeze(-1)

        # — compute scenario context (mean of valid embeddings)
        denom = mask.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        context = h.sum(1) / denom
        context = context.unsqueeze(1).expand(-1, S, -1)     # (B, S, final_dim)

        # — concat per-element and scenario context
        cat = torch.cat([h, context], dim=-1)                # (B, S, 2*final_dim)

        # — final scoring MLP
        scores = self.scorer(cat).squeeze(-1)                # (B, S)
        scores = torch.sigmoid(scores)
        scores = scores * mask.float()
        return scores
