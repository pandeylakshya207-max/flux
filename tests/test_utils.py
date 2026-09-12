import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.nn.sequential import Sequential
from flux.utils.fusion import fused_linear_relu, fused_linear_sigmoid
from flux.utils.graph import build_dot, save_dot
from flux.utils.profiler import MemoryProfiler

# --- fusion ---

def test_fused_linear_relu_forward():
    x = Tensor(np.array([[1.0, -1.0], [2.0, 3.0]]), requires_grad=True)
    W = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), requires_grad=True)
    out = fused_linear_relu(x, W)
    assert out.shape == (2, 2)
    assert np.all(out.data >= 0)

def test_fused_linear_relu_matches_unfused():
    np.random.seed(0)
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    W = Tensor(np.random.randn(8, 4), requires_grad=True)
    fused = fused_linear_relu(x, W)
    x2 = Tensor(x.data.copy(), requires_grad=True)
    W2 = Tensor(W.data.copy(), requires_grad=True)
    unfused = (Tensor(x2.data @ W2.data)).relu()
    assert np.allclose(fused.data, unfused.data)

def test_fused_linear_relu_backward():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    W = Tensor(np.random.randn(4, 5), requires_grad=True)
    b = Tensor(np.zeros(5), requires_grad=True)
    out = fused_linear_relu(x, W, b)
    out.sum().backward()
    assert x.grad is not None
    assert W.grad is not None
    assert b.grad is not None

def test_fused_linear_relu_grad_matches_unfused():
    np.random.seed(1)
    xd = np.random.randn(4, 6)
    Wd = np.random.randn(6, 3)
    x1 = Tensor(xd.copy(), requires_grad=True)
    W1 = Tensor(Wd.copy(), requires_grad=True)
    out1 = fused_linear_relu(x1, W1)
    out1.sum().backward()
    x2 = Tensor(xd.copy(), requires_grad=True)
    W2 = Tensor(Wd.copy(), requires_grad=True)
    z = x2 @ W2
    out2 = z.relu()
    out2.sum().backward()
    assert np.allclose(x1.grad, x2.grad, atol=1e-6)
    assert np.allclose(W1.grad, W2.grad, atol=1e-6)

def test_fused_linear_sigmoid_forward():
    x = Tensor(np.zeros((2, 3)), requires_grad=True)
    W = Tensor(np.ones((3, 2)), requires_grad=True)
    out = fused_linear_sigmoid(x, W)
    assert np.allclose(out.data, 0.5)

def test_fused_linear_sigmoid_backward():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    W = Tensor(np.random.randn(4, 5), requires_grad=True)
    out = fused_linear_sigmoid(x, W)
    out.sum().backward()
    assert x.grad is not None and W.grad is not None

# --- graph ---

def test_build_dot_scalar():
    a = Tensor(np.array([2.0]), requires_grad=True)
    b = Tensor(np.array([3.0]), requires_grad=True)
    c = a + b
    dot = build_dot(c)
    assert "digraph" in dot
    assert "->" in dot

def test_build_dot_contains_op():
    a = Tensor(np.array([2.0]), requires_grad=True)
    b = a * Tensor(np.array([3.0]), requires_grad=True)
    dot = build_dot(b)
    assert "*" in dot

def test_save_dot(tmp_path):
    a = Tensor(np.array([1.0]), requires_grad=True)
    b = a + Tensor(np.array([2.0]))
    path = str(tmp_path / "graph.dot")
    save_dot(b, path)
    import os
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "digraph" in content

# --- profiler ---

def test_profiler_records_layers():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor(np.random.randn(5, 4))
    profiler = MemoryProfiler()
    profiler.profile(model, x)
    assert len(profiler.records) == 3

def test_profiler_output_shapes():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor(np.random.randn(5, 4))
    profiler = MemoryProfiler()
    profiler.profile(model, x)
    assert profiler.records[0]["output_shape"] == (5, 8)
    assert profiler.records[2]["output_shape"] == (5, 2)

def test_profiler_param_count():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    profiler = MemoryProfiler()
    n = profiler.param_count(model)
    assert n == 4*8 + 8 + 8*2 + 2

def test_profiler_report_runs(capsys):
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor(np.random.randn(3, 4))
    profiler = MemoryProfiler()
    profiler.profile(model, x)
    profiler.report()
    captured = capsys.readouterr()
    assert "TOTAL" in captured.out
