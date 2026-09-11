import numpy as np
import pytest
from flux.tensor.tensor import Tensor

# --- constructors ---

def test_from_list():
    t = Tensor([[1,2],[3,4]])
    assert t.shape == (2,2)
    assert t.dtype == np.float64

def test_zeros():
    t = Tensor.zeros(3,4)
    assert t.shape == (3,4)
    assert np.all(t.data == 0)

def test_ones():
    t = Tensor.ones(2,3)
    assert t.shape == (2,3)
    assert np.all(t.data == 1)

def test_randn():
    t = Tensor.randn(10,10)
    assert t.shape == (10,10)

def test_arange():
    t = Tensor.arange(5)
    assert list(t.data) == [0,1,2,3,4]

# --- basic ops forward ---

def test_add_forward():
    a = Tensor([[1,2],[3,4]])
    b = Tensor([[5,6],[7,8]])
    c = a + b
    assert np.allclose(c.data, [[6,8],[10,12]])

def test_mul_forward():
    a = Tensor([[1,2],[3,4]])
    b = Tensor([[2,2],[2,2]])
    c = a * b
    assert np.allclose(c.data, [[2,4],[6,8]])

def test_sub_forward():
    a = Tensor([5.0, 6.0])
    b = Tensor([1.0, 2.0])
    c = a - b
    assert np.allclose(c.data, [4.0, 4.0])

def test_pow_forward():
    a = Tensor([2.0, 3.0])
    b = a ** 2
    assert np.allclose(b.data, [4.0, 9.0])

def test_div_forward():
    a = Tensor([6.0, 8.0])
    b = Tensor([2.0, 4.0])
    c = a / b
    assert np.allclose(c.data, [3.0, 2.0])

def test_neg_forward():
    a = Tensor([1.0, -2.0])
    b = -a
    assert np.allclose(b.data, [-1.0, 2.0])

# --- matmul ---

def test_matmul_forward():
    a = Tensor([[1,2],[3,4]], requires_grad=True)
    b = Tensor([[1,0],[0,1]], requires_grad=True)
    c = a @ b
    assert np.allclose(c.data, [[1,2],[3,4]])

def test_matmul_backward():
    a = Tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
    b = Tensor([[5.0,6.0],[7.0,8.0]], requires_grad=True)
    c = a @ b
    c.backward()
    # dc/da = grad @ b.T
    assert np.allclose(a.grad, np.ones((2,2)) @ np.array([[5,6],[7,8]]).T)
    assert np.allclose(b.grad, np.array([[1,2],[3,4]]).T @ np.ones((2,2)))

# --- sum / mean ---

def test_sum_all():
    a = Tensor([[1,2],[3,4]], requires_grad=True)
    s = a.sum()
    s.backward()
    assert s.data == 10.0
    assert np.allclose(a.grad, np.ones((2,2)))

def test_sum_axis():
    a = Tensor([[1,2],[3,4]], requires_grad=True)
    s = a.sum(axis=0)
    s.backward()
    assert np.allclose(s.data, [4,6])
    assert np.allclose(a.grad, np.ones((2,2)))

def test_mean_all():
    a = Tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
    m = a.mean()
    m.backward()
    assert np.isclose(m.data, 2.5)
    assert np.allclose(a.grad, np.full((2,2), 0.25))

def test_mean_axis():
    a = Tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
    m = a.mean(axis=1)
    m.backward()
    assert np.allclose(m.data, [1.5, 3.5])

# --- reshape / transpose ---

def test_reshape():
    a = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
    b = a.reshape(3,2)
    b.backward()
    assert b.shape == (3,2)
    assert np.allclose(a.grad, np.ones((2,3)))

def test_transpose():
    a = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
    b = a.T
    b.backward()
    assert b.shape == (3,2)
    assert np.allclose(a.grad, np.ones((2,3)))

# --- activations backward ---

def test_relu_backward():
    a = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    b = a.relu()
    b.backward()
    assert np.allclose(b.data, [0.0, 0.0, 2.0])
    assert np.allclose(a.grad, [0.0, 0.0, 1.0])

def test_sigmoid_backward():
    a = Tensor([0.0], requires_grad=True)
    b = a.sigmoid()
    b.backward()
    assert np.isclose(b.data[0], 0.5)
    assert np.isclose(a.grad[0], 0.25)

def test_tanh_backward():
    a = Tensor([0.0], requires_grad=True)
    b = a.tanh()
    b.backward()
    assert np.isclose(b.data[0], 0.0)
    assert np.isclose(a.grad[0], 1.0)

def test_exp_backward():
    a = Tensor([1.0], requires_grad=True)
    b = a.exp()
    b.backward()
    assert np.isclose(b.data[0], np.e)
    assert np.isclose(a.grad[0], np.e)

def test_log_backward():
    a = Tensor([np.e], requires_grad=True)
    b = a.log()
    b.backward()
    assert np.isclose(b.data[0], 1.0)
    assert np.isclose(a.grad[0], 1.0/np.e)

# --- broadcasting ---

def test_broadcast_add():
    a = Tensor([[1,2,3],[4,5,6]], requires_grad=True)  # (2,3)
    b = Tensor([10,20,30], requires_grad=True)          # (3,)
    c = a + b
    c.backward()
    assert c.shape == (2,3)
    assert np.allclose(b.grad, [2,2,2])

def test_broadcast_mul():
    a = Tensor([[1,2],[3,4]], requires_grad=True)  # (2,2)
    b = Tensor([[2],[3]], requires_grad=True)       # (2,1)
    c = a * b
    c.backward()
    assert c.shape == (2,2)
    assert np.allclose(b.grad, [[3],[7]])

# --- slicing ---

def test_slice_forward():
    a = Tensor([[1,2,3],[4,5,6]])
    b = a[0]
    assert np.allclose(b.data, [1,2,3])

def test_slice_backward():
    a = Tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]], requires_grad=True)
    b = a[1]
    b.backward()
    assert np.allclose(a.grad[0], [0,0,0])
    assert np.allclose(a.grad[1], [1,1,1])

# --- numerical grad checker ---

def numerical_grad_tensor(f, t, h=1e-5):
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = t.data[idx]
        t.data[idx] = orig + h
        fwd = f(t).data.sum()
        t.data[idx] = orig - h
        bwd = f(t).data.sum()
        t.data[idx] = orig
        grad[idx] = (fwd - bwd) / (2 * h)
        it.iternext()
    return grad

def test_numerical_grad_matmul():
    np.random.seed(0)
    a = Tensor(np.random.randn(3,4), requires_grad=True)
    b = Tensor(np.random.randn(4,2), requires_grad=False)
    def f(t): return t.matmul(b)
    ng = numerical_grad_tensor(f, a)
    c = a @ b; c.backward()
    assert np.allclose(a.grad, ng, atol=1e-5)

def test_numerical_grad_relu():
    a = Tensor(np.array([0.5, -0.3, 1.2]), requires_grad=True)
    ng = numerical_grad_tensor(lambda t: t.relu(), a)
    b = a.relu(); b.backward()
    assert np.allclose(a.grad, ng, atol=1e-5)

def test_numerical_grad_sigmoid():
    a = Tensor(np.array([0.5, -0.3, 1.2]), requires_grad=True)
    ng = numerical_grad_tensor(lambda t: t.sigmoid(), a)
    b = a.sigmoid(); b.backward()
    assert np.allclose(a.grad, ng, atol=1e-5)
