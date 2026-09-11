import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.module import Module
from flux.nn.linear import Linear
from flux.nn.activations import ReLU, Sigmoid, Tanh
from flux.nn.sequential import Sequential
from flux.nn.losses import mse_loss, cross_entropy_loss

# --- Linear ---

def test_linear_forward_shape():
    layer = Linear(4, 8)
    x = Tensor.randn(3, 4)
    out = layer(x)
    assert out.shape == (3, 8)

def test_linear_no_bias():
    layer = Linear(4, 8, bias=False)
    assert layer.bias is None
    x = Tensor.randn(2, 4)
    out = layer(x)
    assert out.shape == (2, 8)

def test_linear_parameters():
    layer = Linear(4, 8)
    params = layer.parameters()
    assert len(params) == 2  # weight + bias
    assert params[0].shape == (4, 8)
    assert params[1].shape == (8,)

def test_linear_backward():
    layer = Linear(3, 2)
    x = Tensor.randn(4, 3, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None
    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (2,)

def test_linear_zero_grad():
    layer = Linear(3, 2)
    x = Tensor.randn(4, 3)
    out = layer(x)
    out.sum().backward()
    layer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)

# --- Activations ---

def test_relu_layer():
    layer = ReLU()
    x = Tensor([-1.0, 0.0, 2.0])
    out = layer(x)
    assert np.allclose(out.data, [0.0, 0.0, 2.0])

def test_sigmoid_layer():
    layer = Sigmoid()
    x = Tensor([0.0])
    out = layer(x)
    assert np.isclose(out.data[0], 0.5)

def test_tanh_layer():
    layer = Tanh()
    x = Tensor([0.0])
    out = layer(x)
    assert np.isclose(out.data[0], 0.0)

# --- Sequential ---

def test_sequential_forward():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor.randn(3, 4)
    out = model(x)
    assert out.shape == (3, 2)

def test_sequential_parameters():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    params = model.parameters()
    # Linear(4,8): w(4,8)+b(8) + Linear(8,2): w(8,2)+b(2) = 4 params
    assert len(params) == 4

def test_sequential_backward():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
    x = Tensor.randn(5, 4)
    out = model(x)
    out.sum().backward()
    for p in model.parameters():
        assert p.grad is not None

def test_sequential_zero_grad():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor.randn(3, 4)
    model(x).sum().backward()
    model.zero_grad()
    for p in model.parameters():
        assert np.all(p.grad == 0)

# --- MSE Loss ---

def test_mse_loss_zero():
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.0, 2.0, 3.0])
    loss = mse_loss(pred, target)
    assert np.isclose(loss.data, 0.0)

def test_mse_loss_value():
    pred = Tensor([0.0, 0.0], requires_grad=True)
    target = Tensor([1.0, 3.0])
    loss = mse_loss(pred, target)
    # ((0-1)^2 + (0-3)^2) / 2 = (1+9)/2 = 5.0
    assert np.isclose(loss.data, 5.0)

def test_mse_loss_backward():
    pred = Tensor([0.0, 0.0], requires_grad=True)
    target = Tensor([1.0, 3.0])
    loss = mse_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    # d/dpred = 2*(pred-target)/N = [-1.0, -3.0]
    assert np.allclose(pred.grad, [-1.0, -3.0])

# --- CrossEntropy Loss ---

def test_cross_entropy_shape():
    logits = Tensor(np.random.randn(4, 3), requires_grad=True)
    targets = np.array([0, 1, 2, 0])
    loss = cross_entropy_loss(logits, targets)
    assert loss.data.shape == ()

def test_cross_entropy_perfect():
    # very confident correct predictions -> low loss
    logits = Tensor(np.array([[10.0, 0.0, 0.0],
                               [0.0, 10.0, 0.0]]), requires_grad=True)
    targets = np.array([0, 1])
    loss = cross_entropy_loss(logits, targets)
    assert loss.data < 0.01

def test_cross_entropy_backward():
    logits = Tensor(np.random.randn(3, 4), requires_grad=True)
    targets = np.array([0, 2, 1])
    loss = cross_entropy_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (3, 4)

# --- end-to-end: train linear on simple data ---

def test_linear_trains_on_simple_data():
    np.random.seed(42)
    # y = 2*x1 + 3*x2, learn it
    X = Tensor(np.random.randn(20, 2))
    y = Tensor((2 * X.data[:, 0] + 3 * X.data[:, 1]).reshape(-1, 1))
    model = Linear(2, 1)
    for _ in range(500):
        pred = model(X)
        loss = mse_loss(pred, y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.01 * p.grad
    final_loss = mse_loss(model(X), y)
    assert final_loss.data < 0.1
