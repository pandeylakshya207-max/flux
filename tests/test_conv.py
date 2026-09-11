import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.conv import Conv2D, BatchNorm1D, Flatten

def test_conv2d_forward_shape():
    layer = Conv2D(1, 8, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 8, 8, 8)

def test_conv2d_no_padding_shape():
    layer = Conv2D(3, 16, kernel_size=3)
    x = Tensor(np.random.randn(4, 3, 32, 32), requires_grad=True)
    out = layer(x)
    assert out.shape == (4, 16, 30, 30)

def test_conv2d_stride_shape():
    layer = Conv2D(1, 4, kernel_size=2, stride=2)
    x = Tensor(np.random.randn(1, 1, 8, 8), requires_grad=True)
    out = layer(x)
    assert out.shape == (1, 4, 4, 4)

def test_conv2d_backward():
    layer = Conv2D(1, 2, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 1, 8, 8), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert x.grad.shape == x.shape

def test_conv2d_parameters():
    layer = Conv2D(3, 8, kernel_size=3)
    params = layer.parameters()
    assert len(params) == 2

def test_conv2d_no_bias():
    layer = Conv2D(1, 4, kernel_size=3, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1

def test_batchnorm1d_forward_shape():
    bn = BatchNorm1D(8)
    x = Tensor(np.random.randn(16, 8), requires_grad=True)
    out = bn(x)
    assert out.shape == (16, 8)

def test_batchnorm1d_normalizes():
    bn = BatchNorm1D(4)
    x = Tensor(np.random.randn(100, 4) * 10 + 5, requires_grad=True)
    out = bn(x)
    assert np.allclose(out.data.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out.data.std(axis=0), 1.0, atol=1e-2)

def test_batchnorm1d_backward():
    bn = BatchNorm1D(4)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    out = bn(x)
    out.sum().backward()
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None
    assert x.grad is not None

def test_batchnorm1d_eval_mode():
    bn = BatchNorm1D(4)
    x = Tensor(np.random.randn(16, 4))
    bn(x)  # update running stats
    bn.eval()
    x2 = Tensor(np.random.randn(1, 4))
    out = bn(x2)
    assert out.shape == (1, 4)

def test_flatten():
    flat = Flatten()
    x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)
    out = flat(x)
    assert out.shape == (4, 192)
    out.sum().backward()
    assert x.grad is not None
