from typing import Callable, List

import torch
import torch.nn as nn
from torch import Tensor

def ReLU6p(inputs):
    return torch.square(nn.ReLU6()(inputs))

def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    """Get activation function by name."""
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'relu6': nn.ReLU6(),
        'relu6p': ReLU6p,
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name.lower()]

def get_network(**kwargs) -> nn.Module:
    activation_name = kwargs.pop('activation', 'tanh')
    activation = get_activation(activation_name)
    return MLP(out_features=1, activation=activation, **kwargs)


@torch.no_grad()
def _init_weights_bias(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def box_proj(val: Tensor, lower: float, upper: float):
    return (upper - lower) * torch.sigmoid(val) + lower

class Block(nn.Module):
    def __init__(self, n_features, width, act):
        super(Block, self).__init__()
        self.dense1 = nn.Linear(in_features=n_features, out_features=width)
        self.dense2 = nn.Linear(in_features=width, out_features=n_features)
        self.act = act

    def forward(self, x) -> torch.Tensor:
        residual = self.dense1(x)
        residual = self.act(residual)
        residual = self.dense2(residual)
        residual = self.act(residual)
        x = torch.add(x, residual)
        return x

class MLP(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, width: int, depth: int, box: List[float], activation: Callable[[Tensor], Tensor]
    ) -> None:
        super(MLP, self).__init__()
        self.in_layer = nn.Linear(in_features, width)
        self.hiddens = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(width, out_features)
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
