import math
import torch
import torch.nn as nn

class FusionGated(nn.Module):
    def __init__(self, real_hidden_dim, complex_hidden_dim, output_dim, hidden=128, dropout=0.2, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax
        # отдельные линейные проекции каждого потока в общее пространство
        self.pr = nn.Linear(real_hidden_dim,    hidden)
        self.pc = nn.Linear(complex_hidden_dim, hidden)
        self.pi = nn.Linear(complex_hidden_dim, hidden)

        # «внимание» к потокам (gates)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden*3),
            nn.Linear(hidden*3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

        # после слияния — голова
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, h_real, h_r, h_i):
        r = torch.relu(self.pr(h_real))
        cr = torch.relu(self.pc(h_r))
        ci = torch.relu(self.pi(h_i))

        gates_logits = self.gate(torch.cat([r, cr, ci], dim=-1))  # (B,3)
        if self.use_softmax:
            w = torch.softmax(gates_logits, dim=-1)               # суммируются в 1
        else:
            w = torch.sigmoid(gates_logits)                       # независимые 0..1

        # взвешенное слияние (одинаковая размерность hidden)
        fused = w[:,0:1]*r + w[:,1:2]*cr + w[:,2:3]*ci
        return self.mlp(fused)