from __future__ import annotations

import torch
import torch.nn as nn


class ElmanCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ch = nn.Linear(hidden_size, hidden_size, bias=False)   # context -> hidden

    def init_state(self, batch: int, device=None):
        return torch.zeros(batch, self.hidden_size, device=device)

    def forward(self, x_t, state, out_layer):
        c_t = state                                   # context units = copy of h_{t-1}
        h_t = torch.tanh(self.W_xh(x_t) + self.W_ch(c_t))
        y_t = out_layer(h_t)
        return y_t, h_t                               # new context = h_t


class JordanCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.hidden_size, self.output_size = hidden_size, output_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.W_sh = nn.Linear(output_size, hidden_size, bias=False)   # state -> hidden

    def init_state(self, batch: int, device=None):
        return torch.zeros(batch, self.output_size, device=device)

    def forward(self, x_t, state, out_layer):
        s_t = state                                   # state units = copy of y_{t-1}
        h_t = torch.tanh(self.W_xh(x_t) + self.W_sh(s_t))
        y_t = out_layer(h_t)
        return y_t, y_t # new state = y_t


class MRNCell(nn.Module):
    """
    Multi-recurrent network (Ulbricht, 1994) with hidden- and output-layer
    feedback into K memory banks each, bank k having self-recurrence ak
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, decays=(0.25, 0.5, 0.75)):
        super().__init__()
        self.hidden_size, self.output_size = hidden_size, output_size
        self.register_buffer("a", torch.tensor(list(decays), dtype=torch.float32))
        self.K = len(decays)
        mem_size = self.K * (hidden_size + output_size)
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.W_mh = nn.Linear(mem_size, hidden_size, bias=False)

    def init_state(self, batch: int, device=None):
        Mh = torch.zeros(batch, self.K, self.hidden_size, device=device)
        My = torch.zeros(batch, self.K, self.output_size, device=device)
        h = torch.zeros(batch, self.hidden_size, device=device)
        y = torch.zeros(batch, self.output_size, device=device)
        return (Mh, My, h, y)

    def forward(self, x_t, state, out_layer):
        Mh, My, h_prev, y_prev = state
        a = self.a.view(1, -1, 1)
        Mh = a * Mh + (1 - a) * h_prev.unsqueeze(1)   # hidden-layer feedback banks
        My = a * My + (1 - a) * y_prev.unsqueeze(1)   # output-layer feedback banks
        m_t = torch.cat([Mh.flatten(1), My.flatten(1)], dim=1)
        h_t = torch.tanh(self.W_xh(x_t) + self.W_mh(m_t))
        y_t = out_layer(h_t)
        return y_t, (Mh, My, h_t, y_t)


class SimpleRNN(nn.Module):
    """Wraps a cell with the shared output layer and the explicit time loop."""

    def __init__(self, kind: str, hidden_size: int, input_size: int = 1, output_size: int = 1, decays=(0.25, 0.5, 0.75)):
        super().__init__()
        self.kind = kind
        if kind == "elman":
            self.cell = ElmanCell(input_size, hidden_size)
        elif kind == "jordan":
            self.cell = JordanCell(input_size, hidden_size, output_size)
        elif kind == "mrn":
            self.cell = MRNCell(input_size, hidden_size, output_size, decays)
        else:
            raise ValueError(kind)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """x: (batch, T, input_size) -> y: (batch, T) one-step predictions."""
        B, T, _ = x.shape
        state = self.cell.init_state(B, x.device)
        ys = []
        for t in range(T):
            y_t, state = self.cell(x[:, t, :], state, self.out)
            ys.append(y_t)
        return torch.cat(ys, dim=1)


def build_model(kind: str, cfg: dict) -> SimpleRNN:
    return SimpleRNN(kind, hidden_size=cfg["hidden"], decays=tuple(cfg.get("decays", (0.25, 0.5, 0.75))))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
