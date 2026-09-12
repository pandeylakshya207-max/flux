import numpy as np
from .module import Module
from flux.tensor.tensor import Tensor

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.02,
            requires_grad=True
        )

    def forward(self, idx):
        idx = np.array(idx) if not isinstance(idx, np.ndarray) else idx
        out_data = self.weight.data[idx]
        out = Tensor(out_data, requires_grad=True,
                     _children=(self.weight,), _op='embedding')
        def _backward():
            if self.weight.requires_grad:
                self.weight._init_grad()
                np.add.at(self.weight.grad, idx, out.grad)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return f'Embedding({self.num_embeddings}, {self.embedding_dim})'
