from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Format / category bookkeeping (shared by adapters and trainer)
# -----------------------------
REPRESENTATIVE_ORDER = [
    'pipe_serialized', 'token_serialized', 'space_serialized',
    'csv', 'tsv', 'html', 'markdown', 'latex', 'dict', 'json', 'xml',
    'shuffled_rows', 'shuffled_cols', 'transpose',
    'mschema', 'macschema', 'ddl',
]
CATEGORIES = {
    'popular':    ['pipe_serialized', 'token_serialized', 'space_serialized'],
    'data':       ['csv', 'tsv', 'html', 'markdown', 'latex', 'dict', 'json', 'xml'],
    'structural': ['shuffled_rows', 'shuffled_cols', 'transpose'],
    'schema':     ['mschema', 'macschema', 'ddl'],
}
CAT_NAMES = list(CATEGORIES.keys())
FMT2IDX = {f: i for i, f in enumerate(REPRESENTATIVE_ORDER)}
FMT2CAT = {f: CAT_NAMES.index(c) for c, fs in CATEGORIES.items() for f in fs}
N_FMT, N_CAT = len(REPRESENTATIVE_ORDER), len(CAT_NAMES)
_FMT2CAT_T = torch.tensor([FMT2CAT[f] for f in REPRESENTATIVE_ORDER])


# -----------------------------
# Original adapters (kept for backward compatibility)
# -----------------------------
class ResidualCentroidAdapter(nn.Module):
    """
    z = e + alpha * MLP(LayerNorm(e))
    Keeps output in the same latent space as input embedding e.
    """
    def __init__(self, d: int, hidden_mult: int = 4, alpha: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mult * d, d),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return e + self.alpha * self.mlp(self.ln(e))


class BottleneckResidualAdapter(nn.Module):
    """
    e' = e + alpha * Up( Dropout( GELU( Down( LN(e) ) ) ) )
    """
    def __init__(self, d, r=256, alpha=0.1, dropout=0.0, use_bias: bool = True):
        super().__init__()
        self.alpha = alpha
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r, bias=use_bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(r, d, bias=use_bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, e):
        x = self.ln(e)
        return e + self.alpha * self.up(self.drop(self.act(self.down(x))))


# -----------------------------
# Hierarchical bounded-denoising adapters (dot-product retrieval)
# -----------------------------
class FiLM(nn.Module):
    """Per-category scale/shift on the bottleneck. Zero-init means identity at start."""
    def __init__(self, n_cond: int, r: int):
        super().__init__()
        self.emb = nn.Embedding(n_cond, 2 * r)
        nn.init.zeros_(self.emb.weight)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.emb(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class HierDenseAdapter(nn.Module):
    """
    Dot-product consistent dense adapter. Output is NOT normalized.

        x = e - b_fmt                                (closed-form format offset, init mu_delta)
        z = x + Up( FiLM_cat( GELU( Down( LN(x) ) ) ) )

    Stage A removes the table-independent offset. Under dot-product retrieval that
    offset is rank-neutral, so Stage A cannot hurt frozen rankings and only
    canonicalises the input for Stage B, which handles table-dependent noise.

    fmt_head predicts the serialization from e so the adapter can run on
    mixed-format corpora (soft bias selection) when fmt is not given.
    """
    def __init__(self, d: int, r: int = 256, dropout: float = 0.05,
                 mu_delta: Optional[np.ndarray] = None):
        super().__init__()
        self.d, self.r = d, r
        self.bias = nn.Parameter(torch.zeros(N_FMT, d))
        if mu_delta is not None:
            self.bias.data.copy_(torch.as_tensor(mu_delta, dtype=torch.float32))
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r)
        self.film = FiLM(N_CAT, r)
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.fmt_head = nn.Linear(d, N_FMT)
        self.register_buffer("fmt2cat", _FMT2CAT_T.clone())

    def forward(self, e: torch.Tensor, fmt: Optional[torch.Tensor] = None):
        logits = self.fmt_head(e.detach())
        if fmt is None:
            p = logits.softmax(-1)
            b = p @ self.bias
            cat = self.fmt2cat[p.argmax(-1)]
        else:
            b = self.bias[fmt]
            cat = self.fmt2cat[fmt]
        x = e - b
        h = F.gelu(self.down(self.ln(x)))
        h = self.film(h, cat)
        z = x + self.up(self.drop(h))
        return z, logits


class HierSparseAdapter(nn.Module):
    """
    SPLADE adapter that always outputs in sparse space.

        u      = log1p(e)
        h      = FiLM_cat( GELU( Down(u) ) )
        gate   = exp( log_gate_fmt + GateUp(h) )        multiplicative, zeros stay zero
        keep   = e * gate
        expand = topk_{k_exp}( ReLU(ExpUp(h)) masked to e == 0 )
        z      = topk_{K}( keep + expand )

    Sparsity is bounded by construction. k_exp = 0 disables expansion.
    """
    def __init__(self, V: int, r: int = 256, dropout: float = 0.05,
                 k_exp: int = 0, topk: int = 256,
                 log_gate_init: Optional[np.ndarray] = None):
        super().__init__()
        self.V, self.r, self.k_exp, self.topk = V, r, k_exp, topk
        self.log_gate = nn.Parameter(torch.zeros(N_FMT, V))
        if log_gate_init is not None:
            self.log_gate.data.copy_(torch.as_tensor(log_gate_init, dtype=torch.float32))
        self.down = nn.Linear(V, r)
        self.film = FiLM(N_CAT, r)
        self.drop = nn.Dropout(dropout)
        self.gate_up = nn.Linear(r, V)
        self.exp_up = nn.Linear(r, V)
        for m in (self.gate_up, self.exp_up):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.fmt_head = nn.Linear(V, N_FMT)
        self.register_buffer("fmt2cat", _FMT2CAT_T.clone())

    @staticmethod
    def _topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.zeros_like(x)
        if k >= x.shape[-1]:
            return x
        _, idx = x.topk(k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, idx, 1.0)
        return x * mask

    def forward(self, e: torch.Tensor, fmt: Optional[torch.Tensor] = None):
        u = torch.log1p(e)
        logits = self.fmt_head(u.detach())
        if fmt is None:
            p = logits.softmax(-1)
            lg = p @ self.log_gate
            cat = self.fmt2cat[p.argmax(-1)]
        else:
            lg = self.log_gate[fmt]
            cat = self.fmt2cat[fmt]
        h = self.drop(self.film(F.gelu(self.down(u)), cat))
        gate = torch.exp((lg + self.gate_up(h)).clamp(-4.0, 4.0))
        keep = e * gate
        if self.k_exp > 0:
            expand = F.relu(self.exp_up(h)) * (e == 0).float()
            expand = self._topk_mask(expand, self.k_exp)
        else:
            expand = torch.zeros_like(e)
        z = keep + expand
        # final top-K keeps the row inside SPLADE's index budget
        if self.topk < self.V:
            _, idx = z.topk(self.topk, dim=-1)
            z = z * torch.zeros_like(z).scatter_(-1, idx, 1.0)
        return z, logits, expand