
import torch
import torch.nn as nn


# -----------------------------
# Model: Residual centroid adapter (same latent space)
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
    Form:
      e' = e + α * Up( Dropout( GELU( Down( LN(e) ) ) ) )

    Key idea:
      - Down projects d → r where r << d (cheap)
      - Up projects r → d to produce a residual correction in the original space
      - Residual connection keeps the adapter as a "perturbation" rather than a full rewrite
    """
    def __init__(self, d, r=256, alpha=0.1, dropout=0.0,use_bias: bool = True):
        super().__init__()
        self.alpha = alpha
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r, bias=use_bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(r, d,bias=use_bias)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        # often helps stability
        nn.init.zeros_(self.up.weight)

    def forward(self, e):
        x = self.ln(e)
        return e + self.alpha * self.up(self.drop(self.act(self.down(x))))