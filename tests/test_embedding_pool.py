import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.embedding import Embedding
from flux.nn.conv import MaxPool2D

# --- Embedding ---

def test_embedding_forward_shape():
    emb = Embedding(10, 4)
    idx = np.array([0, 2, 5])
    out = emb(idx)
    assert out.shape == (3, 4)

def test_embedding_forward_values():
    emb = Embedding(5, 3)
    idx = np.array([1, 3])
    out = emb(idx)
    assert np.allclose(out.data[0], emb.weight.data[1])
    assert np.allclose(out.data[1], emb.weight.data[3])

def test_embedding_2d_idx():
    emb = Embedding(20, 8)
    idx = np.array([[0,1,2],[3,4,5]])
    out = emb(idx)
    assert out.shape == (2, 3, 8)

def test_embedding_backward():
    emb = Embedding(10, 4)
    idx = np.array([0, 2, 0])
    out = emb(idx)
    out.sum().backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == (10, 4)
    assert np.allclose(emb.weight.grad[0], 2.0)
    assert np.allclose(emb.weight.grad[2], 1.0)
    assert np.allclose(emb.weight.grad[1], 0.0)

def test_embedding_parameters():
    emb = Embedding(10, 4)
    assert len(emb.parameters()) == 1
    assert emb.parameters()[0].shape == (10, 4)

def test_embedding_repr():
    emb = Embedding(100, 16)
    assert "100" in repr(emb) and "16" in repr(emb)

# --- MaxPool2D ---

def test_maxpool_forward_shape():
    pool = MaxPool2D(kernel_size=2)
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    out = pool(x)
    assert out.shape == (2, 3, 4, 4)

def test_maxpool_forward_values():
    pool = MaxPool2D(kernel_size=2)
    x = Tensor(np.array([[[[1,2,3,4],
                            [5,6,7,8],
                            [9,10,11,12],
                            [13,14,15,16]]]]).astype(float), requires_grad=True)
    out = pool(x)
    assert out.shape == (1,1,2,2)
    assert np.allclose(out.data[0,0], [[6,8],[14,16]])

def test_maxpool_stride():
    pool = MaxPool2D(kernel_size=2, stride=1)
    x = Tensor(np.random.randn(1, 1, 6, 6), requires_grad=True)
    out = pool(x)
    assert out.shape == (1, 1, 5, 5)

def test_maxpool_backward():
    pool = MaxPool2D(kernel_size=2)
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 8, 8)

def test_maxpool_backward_routes_to_max():
    pool = MaxPool2D(kernel_size=2)
    x = Tensor(np.array([[[[1.0, 2.0],
                            [3.0, 4.0]]]]), requires_grad=True)
    out = pool(x)
    out.backward()
    assert x.grad[0,0,1,1] == 1.0
    assert x.grad[0,0,0,0] == 0.0

def test_maxpool_parameters():
    pool = MaxPool2D()
    assert pool.parameters() == []
