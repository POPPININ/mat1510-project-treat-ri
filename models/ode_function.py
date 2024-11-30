import torch
import torch.nn as nn

class GNNODEFunc(nn.Module):
    def __init__(self, latent_dim=16):
        super(GNNODEFunc, self).__init__()
        self.gnn = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, t, z):
        return self.gnn(z)
