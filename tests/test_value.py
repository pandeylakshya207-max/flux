import math
import pytest
from flux.engine.value import Value

def test_add():
    a, b = Value(2.0), Value(3.0)
    c = a + b; c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0 and b.grad == 1.0

def test_mul():
    a, b = Value(3.0), Value(4.0)
    c = a * b; c.backward()
    assert c.data == 12.0
    assert a.grad == 4.0 and b.grad == 3.0

def test_pow():
    a = Value(3.0)
    b = a ** 2; b.backward()
    assert b.data == 9.0
    assert abs(a.grad - 6.0) < 1e-6

def test_sub():
    a, b = Value(5.0), Value(3.0)
    c = a - b; c.backward()
    assert c.data == 2.0
    assert a.grad == 1.0 and b.grad == -1.0

def test_div():
    a, b = Value(6.0), Value(2.0)
    c = a / b; c.backward()
    assert abs(c.data - 3.0) < 1e-6
    assert abs(a.grad - 0.5) < 1e-6
    assert abs(b.grad - (-1.5)) < 1e-6

def test_neg():
    a = Value(3.0)
    b = -a; b.backward()
    assert b.data == -3.0 and a.grad == -1.0

def test_radd():
    a = Value(2.0)
    b = 3 + a; b.backward()
    assert b.data == 5.0 and a.grad == 1.0

def test_rmul():
    a = Value(4.0)
    b = 3 * a; b.backward()
    assert b.data == 12.0 and a.grad == 3.0

def test_relu_positive():
    a = Value(2.0)
    b = a.relu(); b.backward()
    assert b.data == 2.0 and a.grad == 1.0

def test_relu_negative():
    a = Value(-3.0)
    b = a.relu(); b.backward()
    assert b.data == 0.0 and a.grad == 0.0

def test_relu_zero():
    a = Value(0.0)
    b = a.relu(); b.backward()
    assert b.data == 0.0 and a.grad == 0.0

def test_sigmoid():
    a = Value(0.0)
    b = a.sigmoid(); b.backward()
    assert abs(b.data - 0.5) < 1e-6
    assert abs(a.grad - 0.25) < 1e-6

def test_tanh():
    a = Value(0.0)
    b = a.tanh(); b.backward()
    assert abs(b.data - 0.0) < 1e-6
    assert abs(a.grad - 1.0) < 1e-6

def test_exp():
    a = Value(1.0)
    b = a.exp(); b.backward()
    assert abs(b.data - math.e) < 1e-6
    assert abs(a.grad - math.e) < 1e-6

def test_log():
    a = Value(math.e)
    b = a.log(); b.backward()
    assert abs(b.data - 1.0) < 1e-6
    assert abs(a.grad - 1.0/math.e) < 1e-6

def test_log_nonpositive_raises():
    a = Value(-1.0)
    with pytest.raises(AssertionError):
        a.log()

def test_chain_add_mul():
    a, b, c = Value(2.0), Value(3.0), Value(4.0)
    d = (a + b) * c; d.backward()
    assert d.data == 20.0
    assert a.grad == 4.0 and b.grad == 4.0 and c.grad == 5.0

def test_shared_node_accumulates_grad():
    a = Value(3.0)
    b = a + a; b.backward()
    assert abs(a.grad - 2.0) < 1e-6

def test_deep_chain():
    a = Value(2.0)
    b = a * a * a; b.backward()
    assert abs(a.grad - 12.0) < 1e-6

def test_numerical_grad_sigmoid():
    a = Value(1.5)
    c = a.sigmoid(); c.backward()
    h = 1e-5
    fwd = Value(1.5 + h).sigmoid().data
    bwd = Value(1.5 - h).sigmoid().data
    ng = (fwd - bwd) / (2 * h)
    assert abs(a.grad - ng) < 1e-5

def test_numerical_grad_tanh():
    a = Value(0.8)
    c = a.tanh(); c.backward()
    h = 1e-5
    ng = (Value(0.8+h).tanh().data - Value(0.8-h).tanh().data) / (2*h)
    assert abs(a.grad - ng) < 1e-5

def test_numerical_grad_exp():
    a = Value(0.5)
    c = a.exp(); c.backward()
    h = 1e-5
    ng = (Value(0.5+h).exp().data - Value(0.5-h).exp().data) / (2*h)
    assert abs(a.grad - ng) < 1e-5

def test_repr():
    a = Value(1.0)
    assert 'Value' in repr(a) and 'data' in repr(a)
