# src/model_complex.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- утилиты ----------
def _safe_num_groups(channels: int, prefer_groups: int = 8) -> int:
    if channels % prefer_groups == 0:
        return prefer_groups
    for g in range(prefer_groups - 1, 1, -1):
        if channels % g == 0:
            return g
    return 1

def _init_linear(m: nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def _init_conv1d(m: nn.Conv1d):
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.zeros_(m.bias)

# ---------- комплексная свёртка ----------
class ComplexConv1D(nn.Module):
    """Комплексная Conv1d: (R,I) -> (R', I'), kernel общий по каналам."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        _init_conv1d(self.conv_r)
        _init_conv1d(self.conv_i)

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        # real/imag: [B, C_in, T]
        r = self.conv_r(real) - self.conv_i(imag)
        i = self.conv_r(imag) + self.conv_i(real)
        return r, i

class ComplexConvBlock1D(nn.Module):
    """(Conv → GN → ReLU → Dropout) ×2 + residual для комплексного сигнала."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ComplexConv1D(in_ch, out_ch, kernel_size)
        self.conv2 = ComplexConv1D(out_ch, out_ch, kernel_size)
        groups = _safe_num_groups(out_ch, 8)
        self.gn_r1 = nn.GroupNorm(groups, out_ch)
        self.gn_i1 = nn.GroupNorm(groups, out_ch)
        self.gn_r2 = nn.GroupNorm(groups, out_ch)
        self.gn_i2 = nn.GroupNorm(groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = None
        if in_ch != out_ch:
            # проекция residual ветки
            self.proj_r = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            self.proj_i = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            _init_conv1d(self.proj_r); _init_conv1d(self.proj_i)
            self.proj = True

    def forward(self, r: torch.Tensor, i: torch.Tensor):
        # r,i: [B, C_in, T]
        res_r, res_i = (r, i) if self.proj is None else (self.proj_r(r), self.proj_i(i))

        r, i = self.conv1(r, i)
        r = self.gn_r1(r); i = self.gn_i1(i)
        r = F.relu(r, inplace=True); i = F.relu(i, inplace=True)
        r = self.drop(r); i = self.drop(i)

        r, i = self.conv2(r, i)
        r = self.gn_r2(r); i = self.gn_i2(i)
        r = F.relu(r, inplace=True); i = F.relu(i, inplace=True)
        r = self.drop(r); i = self.drop(i)

        return r + res_r, i + res_i  # [B, C_out, T]

# ---------- комплексный LSTM-слой ----------
class ComplexLSTMCell(nn.Module):
    """
    Простая комплексная ячейка (tanh), без ворот (подобно Elman), чтобы сохранить стабильность.
    Можно заменить на полноценный комплексный LSTM с воротами, если нужно.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        _init_linear(self.W_ir); _init_linear(self.W_ii)
        _init_linear(self.W_hr); _init_linear(self.W_hi)

    def forward(self, r_in: torch.Tensor, i_in: torch.Tensor, h_r: torch.Tensor, h_i: torch.Tensor):
        # r_in, i_in, h_r, h_i: [B, H]
        r_out = torch.tanh(self.W_ir(r_in) - self.W_ii(i_in) + self.W_hr(h_r) - self.W_hi(h_i))
        i_out = torch.tanh(self.W_ir(i_in) + self.W_ii(r_in) + self.W_hr(h_i) + self.W_hi(h_r))
        return r_out, i_out

class ComplexRNNLayer(nn.Module):
    """Комплексный RNN-слой, который возвращает последовательность скрытых состояний [B,T,H]."""
    def __init__(self, hidden_dim: int, residual: bool = True):
        super().__init__()
        self.cell = ComplexLSTMCell(hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, r_seq: torch.Tensor, i_seq: torch.Tensor):
        # r_seq, i_seq: [B, T, H]
        B, T, H = r_seq.shape
        h_r = r_seq.new_zeros(B, H)
        h_i = r_seq.new_zeros(B, H)

        out_r = []
        out_i = []
        for t in range(T):
            r_t = r_seq[:, t, :]
            i_t = i_seq[:, t, :]
            h_r, h_i = self.cell(r_t, i_t, h_r, h_i)
            out_r.append(h_r)
            out_i.append(h_i)

        # соберём обратно в [B, T, H]
        r_out = torch.stack(out_r, dim=1)
        i_out = torch.stack(out_i, dim=1)

        if self.residual:
            # residual к входной последовательности
            r_out = r_out + r_seq
            i_out = i_out + i_seq

        return r_out, i_out   # [B, T, H]
    # """Каскад из ComplexLSTMCell по времени; опциональный residual по слою."""
    # def __init__(self, hidden_dim: int, residual: bool = True):
    #     super().__init__()
    #     self.cell = ComplexLSTMCell(hidden_dim, hidden_dim)
    #     self.residual = residual

    # def forward(self, r_seq: torch.Tensor, i_seq: torch.Tensor):
    #     # r_seq, i_seq: [B, T, H]
    #     print("r_seq.shape:", r_seq.shape)
    #     B, T, H = r_seq.shape
    #     h_r = r_seq.new_zeros(B, H)
    #     h_i = r_seq.new_zeros(B, H)
    #     for t in range(T):
    #         r_t = r_seq[:, t, :]
    #         i_t = i_seq[:, t, :]
    #         h_r, h_i = self.cell(r_t, i_t, h_r, h_i)
    #     if self.residual:
    #         # residual по последнему таймстепу входа (skip)
    #         r_last = r_seq[:, -1, :]
    #         i_last = i_seq[:, -1, :]
    #         h_r = h_r + r_last
    #         h_i = h_i + i_last
    #     return h_r, h_i  # [B, H]

# ---------- итоговая модель ----------
class ComplexCNNLSTM(nn.Module):
    """
    Вход:  x (complex) [B, T, C]  — C комплекс-каналов (например: week, month, quarter, year, real_time)
    Выход: h_r, h_i: [B, hidden_dim]
    """
    def __init__(
        self,
        in_channels: int,           # число комплекс-каналов на входе
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        kernel_size: int = 3,
        proj_out: bool = False      # если нужно дополнительно линейно спроецировать (обычно не нужно)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # входная комплексная свёртка до hidden_dim каналов
        self.conv_in = ComplexConv1D(in_channels, hidden_dim, kernel_size)
        groups = _safe_num_groups(hidden_dim, 8)
        self.gn_r_in = nn.GroupNorm(groups, hidden_dim)
        self.gn_i_in = nn.GroupNorm(groups, hidden_dim)

        # два residual-комплексных блока
        self.block1 = ComplexConvBlock1D(hidden_dim, hidden_dim, kernel_size, dropout)
        self.block2 = ComplexConvBlock1D(hidden_dim, hidden_dim, kernel_size, dropout)

        # комплексный «LSTM» слой × num_layers
        self.rnn_layers = nn.ModuleList([ComplexRNNLayer(hidden_dim, residual=True) for _ in range(num_layers)])

        # нормализация последнего скрытого состояния
        self.ln_r = nn.LayerNorm(hidden_dim)
        self.ln_i = nn.LayerNorm(hidden_dim)

        # опциональная общая проекция (обычно не требуется для fusion)
        self.proj = nn.Linear(hidden_dim, hidden_dim) if proj_out else None
        if self.proj is not None:
            _init_linear(self.proj)

    def forward(self, x: torch.Tensor):
        """
        x: complex tensor [B, T, C]
        """
        if not torch.is_complex(x):
            raise ValueError("ComplexCNNLSTM ожидает комплексный вход torch.complex dtype")

        B, T, C = x.shape

        # → Conv ожидает [B, C, T]
        r = x.real.permute(0, 2, 1).contiguous()  # [B, C, T]
        i = x.imag.permute(0, 2, 1).contiguous()  # [B, C, T]

        # входная комплексная свёртка
        r, i = self.conv_in(r, i)  # → [B, H, T]
        r = self.gn_r_in(r); i = self.gn_i_in(i)
        r = F.relu(r, inplace=True); i = F.relu(i, inplace=True)
        r = self.dropout(r); i = self.dropout(i)

        # два комплексных residual-блока
        r, i = self.block1(r, i)
        r, i = self.block2(r, i)

        # → для RNN нужно [B, T, H]
        r = r.permute(0, 2, 1).contiguous()  # [B, T, H]
        i = i.permute(0, 2, 1).contiguous()  # [B, T, H]

        # каскад комплексных RNN-слоёв
        # h_r, h_i = r, i
        # for layer in self.rnn_layers:
        #     h_r, h_i = layer(h_r, h_i)
        #     # dropout по слою
        #     h_r = self.dropout(h_r)
        #     h_i = self.dropout(h_i)

        # # LayerNorm по последнему состоянию
        # h_r = self.ln_r(h_r)
        # h_i = self.ln_i(h_i)

        # # опциональная проекция
        # if self.proj is not None:
        #     h_r = self.proj(h_r)
        #     h_i = self.proj(h_i)

        # return h_r, h_i  # [B, hidden_dim]



        # r, i: [B, T, H] после conv+permute
        r_seq, i_seq = r, i
        for layer in self.rnn_layers:
            r_seq, i_seq = layer(r_seq, i_seq)   # << теперь слои возвращают [B, T, H]
            r_seq = self.dropout(r_seq)
            i_seq = self.dropout(i_seq)

        # Берём последний таймстеп только один раз здесь:
        h_r = r_seq[:, -1, :]   # [B, H]
        h_i = i_seq[:, -1, :]   # [B, H]

        # LayerNorm + (опц) proj
        h_r = self.ln_r(h_r)
        h_i = self.ln_i(h_i)
        if self.proj is not None:
            h_r = self.proj(h_r)
            h_i = self.proj(h_i)

        return h_r, h_i