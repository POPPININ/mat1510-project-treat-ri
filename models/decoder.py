import torch.nn as nn

class MultiAgentDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=2):
        super(MultiAgentDecoder, self).__init__()
        self.mlp = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.mlp(z)
