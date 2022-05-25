import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, config, dneck):
        super().__init__()
        nx = config.n_embd if hasattr(config, "n_embd") else config.d_model
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon) if hasattr(config, "layer_norm_epsilon") else nn.LayerNorm(nx)
        self.we = nn.Linear(nx, dneck, bias=False)
        self.wd = nn.Linear(dneck, nx, bias=False)

    def forward(self, x):
        a = self.we(self.ln(x))
        m = self.wd(F.relu(a))
        output = x + m
        return output