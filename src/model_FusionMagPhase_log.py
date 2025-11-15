import torch
import torch.nn as nn

class FusionMagPhase(nn.Module):
    def __init__(self, real_hidden_dim, complex_hidden_dim, output_dim,
                 hidden=128, dropout=0.2, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax

        # проекции real, |complex| и фазового блока в общее пространство
        self.pr   = nn.Linear(real_hidden_dim,    hidden)
        self.pabs = nn.Linear(complex_hidden_dim, hidden)
        self.pphi = nn.Linear(complex_hidden_dim * 2, hidden)  # для sin/cos фазы

        # гейт: решает, сколько брать real vs |complex|, с учётом фазового контекста
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # [w_real, w_abs]
        )

        # голова после слияния
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

    def _encode_streams(self, h_real, h_r, h_i):
        """
        Внутренний кодировщик трёх потоков:
        - real
        - модуль complex
        - фазовый контекст sin/cos(phase)
        """
        eps = 1e-6
        h_abs   = torch.sqrt(h_r**2 + h_i**2 + eps)      # (B, C)
        h_phase = torch.atan2(h_i, h_r)                  # (B, C)

        # фазу представляем через sin/cos, чтобы убрать разрыв на -π/+π
        phase_feat = torch.cat([torch.sin(h_phase), torch.cos(h_phase)], dim=-1)  # (B, 2C)

        r  = torch.relu(self.pr(h_real))          # (B, H)
        a  = torch.relu(self.pabs(h_abs))         # (B, H)
        ph = torch.relu(self.pphi(phase_feat))    # (B, H)

        return r, a, ph

    def _compute_gates(self, h_real, h_r, h_i):
        """
        Считает веса гейта для батча.
        Возвращает w формы (B, 2): [w_real, w_abs].
        """
        r, a, ph = self._encode_streams(h_real, h_r, h_i)
        gates_logits = self.gate(torch.cat([r, a, ph], dim=-1))  # (B, 2)
        if self.use_softmax:
            w = torch.softmax(gates_logits, dim=-1)
        else:
            w = torch.sigmoid(gates_logits)
        return w, r, a   # r,a пригодятся в forward

    def forward(self, h_real, h_r, h_i):
        w, r, a = self._compute_gates(h_real, h_r, h_i)  # w: (B,2)

        # слияние real и |complex|
        fused = w[:, 0:1] * r + w[:, 1:2] * a            # (B, H)


        return self.mlp(fused)                           # (B, output_dim)

    @torch.no_grad()
    def get_gate_weights(self, h_real, h_r, h_i, aggregate: bool = True):
        """
        Удобный метод для диагностики.
        Если aggregate=True -> вернёт средние веса по батчу (2-элементный вектор на CPU).
        Если False -> вернёт тензор (B,2) на CPU.
        """
        w, _, _ = self._compute_gates(h_real, h_r, h_i)
        if aggregate:
        # средний вес по батчу
            return w.mean(dim=0).detach().cpu().numpy()  # shape (2,)
        return w.detach().cpu().numpy()                  # shape (B,2)