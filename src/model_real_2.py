import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_num_groups(channels: int, prefer_groups: int = 8) -> int:
    """Подбирает число групп для GroupNorm, чтобы делилось на channels."""
    if channels % prefer_groups == 0:
        return prefer_groups
    for g in range(prefer_groups - 1, 1, -1):
        if channels % g == 0:
            return g
    return 1


def init_weights(module: nn.Module):
    """
    Инициализация весов:
      - Conv1d: Kaiming Normal под ReLU
      - Linear: Xavier Uniform
      - LSTM: Xavier для входов, Orthogonal для рекуррентных весов, forget-bias = 1
    """
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                hidden_dim = param.shape[0] // 4
                # forget-gate bias
                param.data[hidden_dim:2*hidden_dim] = 1.0


class ConvBlock1D(nn.Module):
    """
    Residual-блок:
      (Conv1d → GroupNorm → ReLU → Dropout) ×2 + skip-коннект.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        groups = _safe_num_groups(out_ch, prefer_groups=8)

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.gn1   = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.gn2   = nn.GroupNorm(groups, out_ch)
        self.drop  = nn.Dropout(dropout)

        # проекция для residual, если меняется число каналов
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        residual = x if self.proj is None else self.proj(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)

        return out + residual  # [B, C_out, T]


class RealCNNLSTM(nn.Module):
    """
    Вход:  x [B, T, F]  (batch, seq_len, num_real_features)
    Выход: h [B, H]     (скрытый вектор для fusion)
    """
    def __init__(
        self,
        num_real_features: int,
        hidden_dim: int,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        kernel_size: int = 3,
        bidirectional: bool = False,
        take: str = "last_timestep",  # "last_timestep" | "h_n"
        proj_out: bool = True,        # проекция до hidden_dim (если bi-LSTM удваивает размер)
    ):
        super().__init__()
        assert take in ("last_timestep", "h_n")
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.take = take

        # --- CNN-часть ---
        pad = kernel_size // 2
        self.conv_in = nn.Conv1d(num_real_features, hidden_dim, kernel_size, padding=pad)
        self.gn_in   = nn.GroupNorm(_safe_num_groups(hidden_dim), hidden_dim)

        self.block1  = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.block2  = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout)

        # --- LSTM-часть ---
        lstm_input_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        out_dim_after_lstm = hidden_dim * (2 if bidirectional else 1)

        # нормализация выхода LSTM
        self.ln_lstm = nn.LayerNorm(out_dim_after_lstm)
        self.dropout = nn.Dropout(dropout)

        # проекция выхода LSTM к hidden_dim (если bi или другая размерность)
        self.proj = nn.Linear(out_dim_after_lstm, hidden_dim) \
            if (proj_out and out_dim_after_lstm != hidden_dim) else None

        # 🔹 residual-проекция исходных фич последнего таймстепа → hidden_dim
        self.residual_proj = nn.Linear(num_real_features, hidden_dim)

        # финальная нормализация после сложения (чтобы стабилизировать skip-сумму)
        self.ln_out = nn.LayerNorm(hidden_dim)

        # инициализация всех весов
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {tuple(x.shape)}")

        B, T, F_in = x.shape

        # сохраним сырые фичи последнего таймстепа для residual-ветки
        x_last_raw = x[:, -1, :]           # [B, F]

        # --- CNN: [B, T, F] -> [B, H, T] ---
        x_cnn = x.permute(0, 2, 1).contiguous()  # [B, F, T]
        x_cnn = self.conv_in(x_cnn)
        x_cnn = self.gn_in(x_cnn)
        x_cnn = F.relu(x_cnn, inplace=True)

        x_cnn = self.block1(x_cnn)        # [B, H, T]
        x_cnn = self.block2(x_cnn)        # [B, H, T]

        # --- LSTM: [B, H, T] -> [B, T, H*(1|2)] ---
        x_seq = x_cnn.permute(0, 2, 1).contiguous()  # [B, T, H]
        x_lstm, (h_n, c_n) = self.lstm(x_seq)        # x_lstm: [B, T, H*(1|2)]
        x_lstm = self.dropout(x_lstm)

        # выбор представления по времени / слоям
        if self.take == "last_timestep":
            feat_lstm = x_lstm[:, -1, :]           # [B, H*(1|2)]
        else:
            last_layer_h = h_n[-(2 if self.bidirectional else 1):, :, :]  # [1|2, B, H]
            feat_lstm = last_layer_h.transpose(0, 1).reshape(B, -1)       # [B, H*(1|2)]

        # нормализуем представление LSTM (его собственную статистику)
        feat = self.ln_lstm(feat_lstm)             # [B, H*(1|2)]

        # приводим к hidden_dim, если нужно
        if self.proj is not None:
            feat = self.proj(feat)                 # [B, hidden_dim]
        else:
            # если proj=None, то out_dim_after_lstm == hidden_dim по определению
            # feat уже [B, hidden_dim]
            pass

        # --- residual от исходных фич последнего таймстепа ---
        res = self.residual_proj(x_last_raw)       # [B, hidden_dim]
        feat = feat + res                          # skip-connection

        # финальная нормализация
        feat = self.ln_out(feat)                   # [B, hidden_dim]

        return feat