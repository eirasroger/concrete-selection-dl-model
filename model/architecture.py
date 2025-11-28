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
        # Self-attention with pre-norm
        x2 = self.norm1(x)
        attn_mask = ~mask          # True where padding
        attn_out, _ = self.attn(x2, x2, x2,
                                key_padding_mask=attn_mask)
        x = x + attn_out           # residual

        # Feed-forward with pre-norm
        x2 = self.norm2(x)
        x = x + self.ffn(x2)       # residual
        return x

class SetRanker(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 scenario_dim: int, 
                 hidden_dims: list[int],
                 dropout: float,
                 num_heads: int = 2,
                 num_attn_blocks: int = 2,
                 ffn_factor: float = 4.0):
        """
        feat_dim: size of the ITEM features only (excluding the scenario vector)
        scenario_dim: size of the SCENARIO one-hot vector (at the end of input)
        hidden_dims: e.g. [512, 256, 128]
        """
        super().__init__()
        assert isinstance(hidden_dims, list) and len(hidden_dims) >= 1
        
        self.scenario_dim = scenario_dim
        self.item_feat_dim = feat_dim - scenario_dim # Auto-calculate split point

        # --- Dedicated Encoder for Item Features ---
        layers_feat = []
        in_dim = self.item_feat_dim
        
        # We use the first N-1 hidden dims for the initial MLP
        # The LAST hidden dim is the target embedding size
        target_dim = hidden_dims[-1]
        
        for h in hidden_dims[:-1]:
            layers_feat += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        
        # Final projection to target_dim
        layers_feat += [nn.Linear(in_dim, target_dim)]
        self.feature_encoder = nn.Sequential(*layers_feat)

        # --- Dedicated Encoder for Scenario Context ---
        # This ensures the scenario vector isn't ignored. 
        # It projects the small one-hot vector up to the same embedding space.
        self.scenario_encoder = nn.Sequential(
            nn.Linear(scenario_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim) # Project to match feature embedding
        )

        # --- Transformer Blocks ---
        final_dim = target_dim
        ffn_dim = int(final_dim * ffn_factor)
        
        self.attn_blocks = nn.ModuleList([
            SetTransformerBlock(embed_dim=final_dim,
                                num_heads=num_heads,
                                ffn_dim=ffn_dim,
                                dropout=dropout)
            for _ in range(num_attn_blocks)
        ])

        # --- Scoring Head ---
        self.scorer = nn.Sequential(
            nn.Linear(2 * final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, 1)
        )

    def forward(self, X, mask):
        """
        X: (B, S, F) where F = item_features + scenario_vector
        """
        B, S, F = X.shape
        
        # SPLIT input into Item Features and Scenario Vector
        # We assume the scenario vector is appended at the END
        X_items = X[:, :, :-self.scenario_dim]      # (B, S, item_feat_dim)
        X_scenario = X[:, :, -self.scenario_dim:]   # (B, S, scenario_dim)

        # Encode them separately
        # Flatten items for MLP: (B*S, item_dim)
        h_items = self.feature_encoder(X_items.reshape(B * S, -1))
        h_items = h_items.view(B, S, -1) # (B, S, final_dim)
        
        # Flatten scenario for MLP: (B*S, scen_dim)
        h_scen = self.scenario_encoder(X_scenario.reshape(B * S, -1))
        h_scen = h_scen.view(B, S, -1)   # (B, S, final_dim)

        # FUSION: Add the scenario context to the item representation
        # This forces the model to condition every item on the scenario
        h = h_items + h_scen 

        # --- Apply Transformer Blocks ---
        for block in self.attn_blocks:
            h = block(h, mask)

        # --- Zero out padded positions ---
        h = h * mask.unsqueeze(-1)

        # --- Global Context Aggregation (Set Context) ---
        denom = mask.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        context = h.sum(1) / denom
        context = context.unsqueeze(1).expand(-1, S, -1) 

        # --- Scoring ---
        cat = torch.cat([h, context], dim=-1)
        scores = self.scorer(cat).squeeze(-1)
        
        scores = torch.sigmoid(scores)
        scores = scores * mask.float()
        
        return scores