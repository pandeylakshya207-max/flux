import numpy as np
from .module import Module
from flux.tensor.tensor import Tensor

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta  = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x):
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = x.data.mean(axis=axis, keepdims=True)
        var  = x.data.var(axis=axis,  keepdims=True)
        xhat = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * xhat + self.beta.data
        out = Tensor(out_data, requires_grad=True,
                     _children=(x, self.gamma, self.beta), _op='layernorm')
        def _backward():
            N = 1
            for s in self.normalized_shape:
                N *= s
            if self.gamma.requires_grad:
                self.gamma._init_grad()
                self.gamma.grad += (out.grad * xhat).sum(
                    axis=tuple(range(x.data.ndim - len(self.normalized_shape))),
                    keepdims=False
                )
            if self.beta.requires_grad:
                self.beta._init_grad()
                self.beta.grad += out.grad.sum(
                    axis=tuple(range(x.data.ndim - len(self.normalized_shape))),
                    keepdims=False
                )
            if x.requires_grad:
                x._init_grad()
                dxhat = out.grad * self.gamma.data
                dvar  = (-0.5 * dxhat * (x.data - mean) *
                         (var + self.eps) ** -1.5).sum(axis=axis, keepdims=True)
                dmean = ((-dxhat / np.sqrt(var + self.eps)).sum(axis=axis, keepdims=True)
                         + dvar * (-2 * (x.data - mean)).mean(axis=axis, keepdims=True))
                x.grad += (dxhat / np.sqrt(var + self.eps)
                           + dvar * 2 * (x.data - mean) / N
                           + dmean / N)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f'LayerNorm({self.normalized_shape})'


class Dropout(Module):
    def __init__(self, p=0.5):
        assert 0 <= p < 1
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p) / (1.0 - self.p)
        out_data = x.data * mask
        out = Tensor(out_data, requires_grad=x.requires_grad,
                     _children=(x,), _op='dropout')
        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += out.grad * mask
        out._backward = _backward
        return out

    def train(self): self.training = True
    def eval(self):  self.training = False

    def __repr__(self):
        return f'Dropout(p={self.p})'
