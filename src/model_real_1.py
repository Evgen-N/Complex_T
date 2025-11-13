import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_num_groups(channels: int, prefer_groups: int = 8) -> int:
    """Подбирает число групп для GroupNorm, чтобы делилось на channels."""
    if channels % prefer_groups == 0:
        return prefer_groups
    # ищем ближайший делитель (<= prefer_groups), иначе 1 (эквивалент LayerNorm по каналам)
    for g in range(prefer_groups - 1, 1, -1):
        if channels % g == 0:
            return g
    return 1


def init_weights(module: nn.Module):
    """Инициализация весов: Kaiming для Conv, Xavier для Linear, orthogonal для LSTM."""
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
                # небольшая «подпорка» для забывания
                hidden_dim = param.shape[0] // 4
                param.data[hidden_dim:2*hidden_dim] = 1.0  # bias для forget gate = 1


class ConvBlock1D(nn.Module):
    """Conv1d → GroupNorm → ReLU → Dropout (×2) + residual (на канал hidden_dim)."""
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

    def forward(self, x):
        # x: [B, C_in, T]
        residual = x if self.proj is None else self.proj(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)

        return x + residual  # [B, C_out, T]


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

        # CNN (двухблочный с residual)
        self.conv_in = nn.Conv1d(num_real_features, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.gn_in   = nn.GroupNorm(_safe_num_groups(hidden_dim), hidden_dim)

        self.block1  = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.block2  = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dropout=dropout)

        # LSTM
        lstm_input_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Нормализации/Dropout после LSTM
        out_dim_after_lstm = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.ln_lstm = nn.LayerNorm(out_dim_after_lstm)

        # Финальная проекция, чтобы вернуть размерность к hidden_dim (удобно для fusion)
        self.proj = nn.Linear(out_dim_after_lstm, hidden_dim) if (proj_out and out_dim_after_lstm != hidden_dim) else None

        # Инициализация
        self.apply(init_weights)

    def forward(self, x):
        # x: [B, T, F]
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {tuple(x.shape)}")

        # → Conv1d ожидает [B, C, T]
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T]
        x = self.conv_in(x)
        x = self.gn_in(x)
        x = F.relu(x, inplace=True)

        x = self.block1(x)  # [B, H, T]
        x = self.block2(x)  # [B, H, T]

        # → обратно для LSTM: [B, T, H]
        x = x.permute(0, 2, 1).contiguous()

        # LSTM
        x, (h_n, c_n) = self.lstm(x)  # x: [B, T, H*(1|2)], h_n: [L*(1|2), B, H]
        x = self.dropout(x)

        if self.take == "last_timestep":
            # последний таймстеп
            feat = x[:, -1, :]   # [B, H*(1|2)]
        else:
            # скрытое состояние последнего слоя (и направления)
            # берём последний слой:
            last_layer_h = h_n[-(2 if self.bidirectional else 1):, :, :]  # [1|2, B, H]
            feat = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)     # [B, H*(1|2)]

        feat = self.ln_lstm(feat)  # [B, H*(1|2)]

        if self.proj is not None:
            feat = self.proj(feat)  # [B, H]

        return feat  # [B, hidden_dim]