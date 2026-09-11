import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.linear import Linear
from flux.nn.losses import mse_loss
from flux.optim.sgd import SGD
from flux.optim.adam import Adam
from flux.optim.adagrad import AdaGrad
from flux.optim.rmsprop import RMSProp
from flux.optim.schedulers import StepLR, ExponentialLR, CosineAnnealingLR, clip_grad_norm

def make_problem():
    np.random.seed(0)
    layer = Linear(2, 1)
    X = Tensor(np.random.randn(10, 2))
    y = Tensor(np.random.randn(10, 1))
    return layer, X, y

def train(layer, X, y, opt, steps=100):
    for _ in range(steps):
        pred = layer(X)
        loss = mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mse_loss(layer(X), y).data

# --- SGD ---

def test_sgd_reduces_loss():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.01)
    loss = train(layer, X, y, opt)
    assert loss < 2.0

def test_sgd_momentum_reduces_loss():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.01, momentum=0.9)
    loss = train(layer, X, y, opt)
    assert loss < 2.0

def test_sgd_weight_decay():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.01, weight_decay=1e-4)
    loss = train(layer, X, y, opt)
    assert loss < 5.0

def test_sgd_zero_grad_clears():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.01)
    pred = layer(X)
    mse_loss(pred, y).backward()
    opt.zero_grad()
    for p in layer.parameters():
        assert np.all(p.grad == 0)

# --- Adam ---

def test_adam_reduces_loss():
    layer, X, y = make_problem()
    opt = Adam(layer.parameters(), lr=0.01)
    loss = train(layer, X, y, opt)
    assert loss < 2.0

def test_adam_bias_correction():
    layer, X, y = make_problem()
    opt = Adam(layer.parameters(), lr=0.001, betas=(0.9, 0.999))
    assert opt.t == 0
    pred = layer(X); mse_loss(pred, y).backward()
    opt.step()
    assert opt.t == 1

def test_adam_weight_decay():
    layer, X, y = make_problem()
    opt = Adam(layer.parameters(), lr=0.01, weight_decay=1e-4)
    loss = train(layer, X, y, opt)
    assert loss < 5.0

# --- AdaGrad ---

def test_adagrad_reduces_loss():
    layer, X, y = make_problem()
    opt = AdaGrad(layer.parameters(), lr=0.1)
    loss = train(layer, X, y, opt)
    assert loss < 2.0

def test_adagrad_accumulates_G():
    layer, X, y = make_problem()
    opt = AdaGrad(layer.parameters(), lr=0.1)
    pred = layer(X); mse_loss(pred, y).backward()
    G_before = [g.copy() for g in opt.G]
    opt.step()
    for i, g in enumerate(opt.G):
        assert np.any(g != G_before[i])

# --- RMSProp ---

def test_rmsprop_reduces_loss():
    layer, X, y = make_problem()
    opt = RMSProp(layer.parameters(), lr=0.01)
    loss = train(layer, X, y, opt)
    assert loss < 2.0

def test_rmsprop_updates_v():
    layer, X, y = make_problem()
    opt = RMSProp(layer.parameters(), lr=0.01)
    pred = layer(X); mse_loss(pred, y).backward()
    v_before = [v.copy() for v in opt.v]
    opt.step()
    for i, v in enumerate(opt.v):
        assert np.any(v != v_before[i])

# --- Schedulers ---

def test_steplr_decays():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.1)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    sched.step(); sched.step()
    assert np.isclose(opt.lr, 0.05)

def test_steplr_no_decay_before_step():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.1)
    sched = StepLR(opt, step_size=5, gamma=0.1)
    sched.step()
    assert np.isclose(opt.lr, 0.1)

def test_exponential_lr():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=0.1)
    sched = ExponentialLR(opt, gamma=0.9)
    sched.step()
    assert np.isclose(opt.lr, 0.09)
    sched.step()
    assert np.isclose(opt.lr, 0.081)

def test_cosine_annealing():
    layer, X, y = make_problem()
    opt = SGD(layer.parameters(), lr=1.0)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    sched.step()
    assert 0.0 < opt.lr < 1.0
    for _ in range(9): sched.step()
    assert opt.lr < 0.01

# --- Grad clipping ---

def test_clip_grad_norm():
    layer = Linear(10, 10)
    X = Tensor.randn(5, 10)
    y = Tensor.randn(5, 10)
    loss = mse_loss(layer(X), y)
    loss.backward()
    norm = clip_grad_norm(layer.parameters(), max_norm=1.0)
    for p in layer.parameters():
        assert np.linalg.norm(p.grad) <= 1.0 + 1e-6

def test_clip_grad_norm_returns_total():
    layer = Linear(4, 4)
    X = Tensor.randn(3, 4)
    y = Tensor.randn(3, 4)
    mse_loss(layer(X), y).backward()
    total = clip_grad_norm(layer.parameters(), max_norm=100.0)
    assert total > 0
