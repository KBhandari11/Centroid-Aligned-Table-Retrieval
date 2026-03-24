import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Losses (VICReg-style + identity preservation)
# -----------------------------
def invariance_to_centroid_loss(z: torch.Tensor, table_ids: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(table_ids)
    per = []
    for tid in unique:
        idx = (table_ids == tid).nonzero(as_tuple=True)[0]
        z_t = z.index_select(0, idx)
        c_t = z_t.mean(dim=0, keepdim=True).detach()
        per.append(((z_t - c_t) ** 2).sum(dim=1).mean())
    return torch.stack(per).mean()


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std) ** 2)


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    n = zc.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    cov = (zc.T @ zc) / (n - 1)
    return torch.mean(_offdiag(cov) ** 2)


def identity_preservation_loss(z: torch.Tensor, e: torch.Tensor, mode: str = "cos") -> torch.Tensor:
    if mode == "cos":
        return torch.mean(1.0 - F.cosine_similarity(z, e, dim=-1))
    if mode == "l2":
        return torch.mean((z - e) ** 2)
    raise ValueError("mode must be 'cos' or 'l2'")