import torch
import torch.nn as nn

class SwishBeta(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)