
from dataclasses import dataclass
import torch

# -----------------------------
# Training config + loop
# -----------------------------
@dataclass
class TrainCfg:
    steps: int = 10000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_views: int = 4

    lam_inv: float = 25.0
    lam_var: float = 25.0
    lam_cov: float = 1.0

    lam_id: float = 1.0
    id_mode: str = "cos"

    gamma_std: float = 0.2

    hidden_mult: int = 4 # this is for ResidualCentroidAdapter
    r: int = 256 # this is for BottleneckResidualAdapter
    use_bias: bool =True
    alpha: float = 0.10
    dropout: float = 0.05

    lam_sparse: float = 0.0  # L1 sparsity regularization (SPLADE only)

    log_every: int = 200
    ckpt_every: int = 200


# =============================================================================
# Config
# =============================================================================
@dataclass
class HierCfg:
    # optimization
    steps            = 30000    # was 20000 — compensate for lower lr
    tables_per_batch = 96       # was 32 — larger batches → better NQ gradient signal
    lr               = 1e-4     # was 3e-4 — more conservative, less domain drift
    bias_lr_mult     = 0.05     # was 0.1 — format offsets change even more slowly
    weight_decay     = 1e-3     # was 1e-4 — more regularization overall
    clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    # model
    r: int = 1028
    dropout: float = 0.05
    k_exp    = 0      # keep gate-only; expansion with NQ hurt by vocabulary mismatch
    # loss weights
    w_all: float = 1.0
    w_cat: float = 1.0
    w_prog: float = 1.0
    w_bound          = 8.0      # was 5.0 — stronger penalty for straying from frozen
    w_sep: float = 1.0
    w_fmt: float = 0.1
    w_flops  = 5e-3   # was 1e-3 — penalize sparse expansion more aggressively
    # loss geometry (relative squared euclidean, scale free)
    delta_all        = 0.01     # was 0.02 — tighter centroid proximity target
    delta_cat: float = 0.02
    margin: float = 0.01
    rho              = 0.5      # was 0.75 — tighter displacement from frozen embedding
    anneal_frac      = 0.4      # was 0.3 — spend more time in format-align phase
    # cache / data
    cache_prefix: str = "chunked_"       # "" if the retrieval-test cache uses plain rep keys
    stat_sample: int = 10000              # tables used for mu_delta / tau estimation
    # logging
    log_every: int = 200
    ckpt_every: int = 500

'''
class HierCfg:
    # optimisation
    steps: int = 20000
    tables_per_batch: int = 32           # every available view of each table is emitted
    lr: float = 3e-4
    bias_lr_mult: float = 0.1            # slower lr for closed-form offsets (bias / log_gate)
    weight_decay: float = 1e-4
    clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    # model
    r: int = 256
    dropout: float = 0.05
    k_exp: int = 0                       # SPLADE expansion budget; 0 = gate only
    # loss weights
    w_all: float = 1.0
    w_cat: float = 1.0
    w_prog: float = 1.0
    w_bound: float = 5.0
    w_sep: float = 1.0
    w_fmt: float = 0.1
    w_flops: float = 1e-3
    # loss geometry (relative squared euclidean, scale free)
    delta_all: float = 0.02
    delta_cat: float = 0.02
    margin: float = 0.01
    rho: float = 0.75                    # displacement bound = rho * median frozen distance
    anneal_frac: float = 0.3
    # cache / data
    cache_prefix: str = "chunked_"       # "" if the retrieval-test cache uses plain rep keys
    stat_sample: int = 5000              # tables used for mu_delta / tau estimation
    # logging
    log_every: int = 200
    ckpt_every: int = 500
'''