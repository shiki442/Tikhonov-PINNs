from typing import Callable, List

import torch
import torch.nn as nn
from torch import Tensor


class ReLU2(nn.Module):
    def forward(self, x):
        return torch.pow(torch.relu(x), 1.5)


def get_network(**kwargs) -> nn.Module:
    activation = nn.Tanh()
    return MLP(in_features=2, out_features=1, activation=activation, **kwargs)


@torch.no_grad()
def _init_weights_bias(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def box_proj(val: Tensor, lower: float, upper: float):
    return (upper - lower) * torch.sigmoid(val) + lower


class MLP(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, width_list: List[int], box: List[float], activation: Callable[[Tensor], Tensor]
    ) -> None:
        super(MLP, self).__init__()
        self.in_layer = nn.Linear(in_features, width_list[0])
        self.hiddens = nn.ModuleList([nn.Linear(width_list[i], width_list[i + 1]) for i in range(len(width_list) - 1)])
        self.out_layer = nn.Linear(width_list[-1], out_features)
        self.box = box
        self.act = activation
        self.apply(_init_weights_bias)

    def forward(self, x: Tensor):
        x = self.in_layer(x)
        x = self.act(x)
        for layer in self.hiddens:
            residual = x  # Save the current input as residual
            x = layer(x)  # Forward pass through the layer
            x += residual  # Add residual connection
            x = self.act(x)  # Apply activation function
        out = self.out_layer(x)
        if isinstance(self.box, list):
            out = box_proj(out, self.box[0], self.box[1])
        return out
