import torch
import torch.nn as nn
import math

class FusionMagPhase(nn.Module):
    def __init__(self, real_hidden_dim, complex_hidden_dim, output_dim,
                 hidden=128, dropout=0.2, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax

        # проекции real, |complex| и phase в общее пространство
        self.pr   = nn.Linear(real_hidden_dim,    hidden)
        self.pabs = nn.Linear(complex_hidden_dim, hidden)
        self.pphi = nn.Linear(complex_hidden_dim*2, hidden)  # для sin/cos фазы

        # гейт: решает, сколько брать real vs |complex|
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden*3),
            nn.Linear(hidden*3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # веса для [real, |complex|]
        )

        # голова после слияния
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, h_real, h_r, h_i):
        # 1) модуль и фаза
        eps = 1e-6
        h_abs   = torch.sqrt(h_r**2 + h_i**2 + eps)      # (B, C)
        h_phase = torch.atan2(h_i, h_r)                  # (B, C)
        # чтобы не было разрывов -pi/+pi, лучше использовать sin/cos фазы
        phase_feat = torch.cat([torch.sin(h_phase), torch.cos(h_phase)], dim=-1)  # (B, 2C)

        # 2) проекции
        r  = torch.relu(self.pr(h_real))          # (B, H)
        a  = torch.relu(self.pabs(h_abs))         # (B, H)
        ph = torch.relu(self.pphi(phase_feat))    # (B, H) — фаза как отдельный «контекст»

        # 3) гейт: решает, сколько real vs |complex|, с учётом фазы
        gates_logits = self.gate(torch.cat([r, a, ph], dim=-1))  # (B, 2)
        if self.use_softmax:
            w = torch.softmax(gates_logits, dim=-1)  # (B, 2), сумма=1
        else:
            w = torch.sigmoid(gates_logits)          # независимая 0..1

        # 4) слияние real и |complex|
        fused = w[:,0:1]*r + w[:,1:2]*a              # (B, H)

        # 5) голова
        return self.mlp(fused)                       # (B, output_dim)