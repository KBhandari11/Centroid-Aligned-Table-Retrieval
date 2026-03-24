
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

    log_every: int = 200
    ckpt_every: int = 200