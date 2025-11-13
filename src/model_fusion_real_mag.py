import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """
    Комплексный linear:
      (W_r + i W_i)(h_r + i h_i) + (b_r + i b_i)
    На вход:  h_r, h_i: [B, in_dim]
    На выход: out_r, out_i: [B, out_dim]
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.Wr = nn.Linear(in_dim, out_dim, bias=False)
        self.Wi = nn.Linear(in_dim, out_dim, bias=False)
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_dim))
            self.bias_i = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias_r = None
            self.bias_i = None

        # инициализация
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Wi.weight)

    def forward(self, h_r: torch.Tensor, h_i: torch.Tensor):
        # (W_r h_r - W_i h_i) + i( W_r h_i + W_i h_r )
        re = self.Wr(h_r) - self.Wi(h_i)
        im = self.Wr(h_i) + self.Wi(h_r)
        if self.bias_r is not None:
            re = re + self.bias_r
            im = im + self.bias_i
        return re, im

class FusionComplexLinear(nn.Module):
    """
    Fusion v2:
      - real-ветка → Linear → ReLU
      - complex-ветка → ComplexLinear → (re, im)

      complex_mode:
        "real" -> используем ReLU(out_re) как вклад complex
        "mag"  -> используем ReLU(|out|) = ReLU(sqrt(re^2 + im^2))
    """
    def __init__(
        self,
        real_hidden_dim: int,
        complex_hidden_dim: int,
        output_dim: int,
        hidden: int = 128,
        dropout: float = 0.2,
        complex_mode: str = "real"  # "real" | "mag"
    ):
        super().__init__()
        assert complex_mode in ("real", "mag")
        self.complex_mode = complex_mode

        # real-поток в общее пространство
        self.pr = nn.Linear(real_hidden_dim, hidden)

        # комплексный linear-проектор complex-ветки
        self.cproj = ComplexLinear(complex_hidden_dim, hidden, bias=True)

        # голова после слияния
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

        # инициализация real-проекции и головы
        nn.init.xavier_uniform_(self.pr.weight)
        if self.pr.bias is not None:
            nn.init.zeros_(self.pr.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_real: torch.Tensor, h_r: torch.Tensor, h_i: torch.Tensor):
        # real-поток
        r = F.relu(self.pr(h_real))  # [B, H]

        # complex-поток через комплексный Linear
        c_re, c_im = self.cproj(h_r, h_i)  # [B, H], [B, H]

        if self.complex_mode == "real":
            c = F.relu(c_re)
        else:  # "mag"
            eps = 1e-6
            c = torch.sqrt(c_re**2 + c_im**2 + eps)
            c = F.relu(c)

        # слияние: конкат real и complex
        fused = torch.cat([r, c], dim=-1)  # [B, 2H]

        # голова
        out = self.mlp(fused)  # [B, output_dim]
        return out