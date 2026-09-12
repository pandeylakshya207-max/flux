import numpy as np
from flux.tensor.tensor import Tensor

def fused_linear_relu(x, weight, bias=None):
    """Fused matmul + relu: single forward pass, single backward."""
    z_data = x.data @ weight.data
    if bias is not None:
        z_data += bias.data
    mask = z_data > 0
    out_data = z_data * mask
    children = (x, weight) + ((bias,) if bias is not None else ())
    out = Tensor(out_data, requires_grad=True, _children=children, _op='fused_linear_relu')
    def _backward():
        grad = out.grad * mask
        if x.requires_grad:
            x._init_grad()
            x.grad += grad @ weight.data.T
        if weight.requires_grad:
            weight._init_grad()
            weight.grad += x.data.T @ grad
        if bias is not None and bias.requires_grad:
            bias._init_grad()
            bias.grad += grad.sum(axis=0)
    out._backward = _backward
    return out

def fused_linear_sigmoid(x, weight, bias=None):
    """Fused matmul + sigmoid."""
    z_data = x.data @ weight.data
    if bias is not None:
        z_data += bias.data
    s = 1.0 / (1.0 + np.exp(-z_data))
    out = Tensor(s, requires_grad=True, _children=(x, weight) + ((bias,) if bias is not None else ()), _op='fused_linear_sigmoid')
    def _backward():
        ds = out.grad * s * (1 - s)
        if x.requires_grad:
            x._init_grad()
            x.grad += ds @ weight.data.T
        if weight.requires_grad:
            weight._init_grad()
            weight.grad += x.data.T @ ds
        if bias is not None and bias.requires_grad:
            bias._init_grad()
            bias.grad += ds.sum(axis=0)
    out._backward = _backward
    return out
