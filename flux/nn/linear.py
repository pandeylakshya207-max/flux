import numpy as np
from flux.tensor.tensor import Tensor
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming uniform init
        k = np.sqrt(1.0 / in_features)
        self.weight = Tensor(np.random.uniform(-k, k, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def __repr__(self):
        return f'Linear(in={self.in_features}, out={self.out_features})'
