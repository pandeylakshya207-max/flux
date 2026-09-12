import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.normalization import LayerNorm, Dropout

# --- LayerNorm ---

def test_layernorm_shape():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    out = ln(x)
    assert out.shape == (4, 8)

def test_layernorm_normalizes():
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(32, 16) * 10 + 5, requires_grad=True)
    out = ln(x)
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.data.std(axis=-1),  1.0, atol=1e-4)

def test_layernorm_3d():
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(2, 8, 16), requires_grad=True)
    out = ln(x)
    assert out.shape == (2, 8, 16)
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)

def test_layernorm_backward():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    out = ln(x)
    out.sum().backward()
    assert x.grad is not None
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None
    assert x.grad.shape == x.shape

def test_layernorm_gamma_beta_grad():
    ln = LayerNorm(4)
    x = Tensor(np.random.randn(8, 4), requires_grad=True)
    out = ln(x)
    out.sum().backward()
    assert ln.gamma.grad.shape == (4,)
    assert ln.beta.grad.shape == (4,)

def test_layernorm_numerical_grad():
    ln = LayerNorm(4)
    xd = np.random.randn(3, 4)
    x = Tensor(xd.copy(), requires_grad=True)
    out = ln(x)
    out.sum().backward()
    h = 1e-5
    ng = np.zeros_like(xd)
    for i in range(3):
        for j in range(4):
            ln2 = LayerNorm(4)
            ln2.gamma.data = ln.gamma.data.copy()
            ln2.beta.data  = ln.beta.data.copy()
            xp = xd.copy(); xp[i,j] += h
            xm = xd.copy(); xm[i,j] -= h
            fwd = ln2(Tensor(xp)).data.sum()
            ln3 = LayerNorm(4)
            ln3.gamma.data = ln.gamma.data.copy()
            ln3.beta.data  = ln.beta.data.copy()
            bwd = ln3(Tensor(xm)).data.sum()
            ng[i,j] = (fwd - bwd) / (2*h)
    assert np.allclose(x.grad, ng, atol=1e-4)

def test_layernorm_parameters():
    ln = LayerNorm(8)
    assert len(ln.parameters()) == 2

# --- Dropout ---

def test_dropout_training_zeros_some():
    np.random.seed(0)
    drop = Dropout(p=0.5)
    x = Tensor(np.ones((100, 100)), requires_grad=True)
    out = drop(x)
    zero_frac = (out.data == 0).mean()
    assert 0.4 < zero_frac < 0.6

def test_dropout_eval_passthrough():
    drop = Dropout(p=0.5)
    drop.eval()
    x = Tensor(np.ones((10, 10)), requires_grad=True)
    out = drop(x)
    assert np.allclose(out.data, x.data)

def test_dropout_scales_correctly():
    np.random.seed(1)
    drop = Dropout(p=0.5)
    x = Tensor(np.ones((1000,)), requires_grad=True)
    out = drop(x)
    kept = out.data[out.data != 0]
    assert np.allclose(kept, 2.0)

def test_dropout_backward():
    np.random.seed(2)
    drop = Dropout(p=0.3)
    x = Tensor(np.random.randn(5, 5), requires_grad=True)
    out = drop(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

def test_dropout_zero_p():
    drop = Dropout(p=0.0)
    x = Tensor(np.ones((4, 4)), requires_grad=True)
    out = drop(x)
    assert np.allclose(out.data, x.data)

def test_dropout_train_eval_toggle():
    drop = Dropout(p=0.9)
    drop.eval()
    assert not drop.training
    drop.train()
    assert drop.training
