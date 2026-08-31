from typing import Optional

from torch import nn
import torch.nn.functional as F


INIT_METHODS = ("kaiming", "xavier")


class LambdaNet(nn.Module):
    def __init__(self, input_dim, init_method: Optional[str] = None):
        super(LambdaNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        if init_method is not None:
            self.apply(lambda m: init_weights(m, init_method))

    def forward(self, x):
        return self.layers(x)


def init_weights(m: nn.Module, init_method: str) -> None:
    if not isinstance(m, nn.Linear):
        return

    if init_method == "kaiming":
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
    elif init_method == "xavier":
        nn.init.xavier_normal_(m.weight)
    else:
        raise ValueError(f"Unknown init_method '{init_method}', expected one of {INIT_METHODS}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)
