import torch
import torch.nn as nn

class FusionConcatMag(nn.Module):
    """
    Упрощённый fusion:
      - real → pr → ReLU → r
      - complex (h_r, h_i) → |h| → pabs → ReLU → a
      - fused = [r, a] → MLP → output
    Без гейта: complex всегда участвует наравне с real.
    """
    def __init__(
        self,
        real_hidden_dim: int,
        complex_hidden_dim: int,
        output_dim: int,
        hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.pr   = nn.Linear(real_hidden_dim,    hidden)
        self.pabs = nn.Linear(complex_hidden_dim, hidden)

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

        # инициализация
        nn.init.xavier_uniform_(self.pr.weight)
        nn.init.xavier_uniform_(self.pabs.weight)
        if self.pr.bias is not None:
            nn.init.zeros_(self.pr.bias)
        if self.pabs.bias is not None:
            nn.init.zeros_(self.pabs.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_real, h_r, h_i):
        eps = 1e-6
        h_abs = torch.sqrt(h_r**2 + h_i**2 + eps)         # [B, C]
        r = F.relu(self.pr(h_real))                       # [B, H]
        a = F.relu(self.pabs(h_abs))                      # [B, H]

        fused = torch.cat([r, a], dim=-1)                 # [B, 2H]
        return self.mlp(fused)                            # [B, output_dim]