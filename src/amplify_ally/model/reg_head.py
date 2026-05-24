import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..inference import SWE_Pooling


class LambdaNet(nn.Module):
    def __init__(self, input_dim):
        super(LambdaNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.layers(x) 
