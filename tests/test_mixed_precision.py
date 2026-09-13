import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.sequential import Sequential
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.nn.losses import mse_loss, cross_entropy_loss
from flux.optim.adam import Adam
from flux.utils.mixed_precision import MixedPrecisionWrapper, autocast, fp16_safe

def make_model():
    np.random.seed(0)
    return Sequential(Linear(8, 16), ReLU(), Linear(16, 4))

# --- autocast / fp16_safe ---

def test_autocast_roundtrip():
    x = Tensor(np.random.randn(4, 8))
    out = autocast(x)
    assert out.dtype == np.float64

def test_fp16_safe_clips():
    x = Tensor(np.array([1e6, -1e6, 0.5]))
    out = fp16_safe(x)
    assert np.all(np.abs(out.data) <= 65504)

def test_fp16_safe_preserves_small():
    x = Tensor(np.array([1.0, -1.0, 0.5]))
    out = fp16_safe(x)
    assert np.allclose(out.data, [1.0, -1.0, 0.5], atol=1e-2)

# --- MixedPrecisionWrapper ---

def test_wrapper_forward_shape():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    out = wrapper(x)
    assert out.shape == (4, 4)

def test_wrapper_output_is_float64():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    out = wrapper(x)
    assert out.data.dtype == np.float64

def test_wrapper_weights_restored_after_forward():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model)
    original = [p.data.copy() for p in model.parameters()]
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    wrapper(x)
    for orig, p in zip(original, model.parameters()):
        assert p.data.dtype == orig.dtype

def test_scale_loss():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=64.0)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    out  = wrapper(x)
    loss = out.mean()
    scaled = wrapper.scale_loss(loss)
    assert np.isclose(scaled.data, loss.data * 64.0)

def test_unscale_grads():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=128.0)
    x = Tensor(np.random.randn(4, 8))
    out  = wrapper(x)
    loss = out.mean()
    wrapper.zero_grad()
    loss.backward()
    # manually set a known grad
    for p in model.parameters():
        if p.grad is not None:
            p.grad = np.ones_like(p.grad) * 128.0
    wrapper.unscale_grads()
    for p in model.parameters():
        if p.grad is not None:
            assert np.allclose(p.grad, 1.0)

def test_overflow_detection():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=128.0)
    # inject inf grad
    for p in model.parameters():
        p.grad = np.full_like(p.data, np.inf)
    wrapper.unscale_grads()
    assert wrapper.overflow is True

def test_no_overflow_clean_grads():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=128.0)
    for p in model.parameters():
        p.grad = np.ones_like(p.data)
    wrapper.unscale_grads()
    assert wrapper.overflow is False

def test_wrapper_trains():
    np.random.seed(1)
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=128.0)
    opt     = Adam(model.parameters(), lr=1e-3)
    X = Tensor(np.random.randn(16, 8))
    y = Tensor(np.random.randn(16, 4))
    losses = []
    for _ in range(20):
        out  = wrapper(X)
        loss = mse_loss(out, y)
        wrapper.zero_grad()
        loss.backward()
        wrapper.unscale_grads()
        if not wrapper.overflow:
            opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]

def test_zero_grad_clears():
    model   = make_model()
    wrapper = MixedPrecisionWrapper(model, loss_scale=128.0)
    x = Tensor(np.random.randn(4, 8))
    out = wrapper(x)
    out.mean().backward()
    wrapper.zero_grad()
    for p in model.parameters():
        assert np.all(p.grad == 0)
