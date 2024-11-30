import torch
import numpy as np

class TemporalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(TemporalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x, t):
        return x + self.pe[t.long()]
