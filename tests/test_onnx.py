import os
import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.sequential import Sequential
from flux.nn.linear import Linear
from flux.nn.activations import ReLU, Sigmoid, Tanh
from flux.nn.normalization import Dropout
from flux.utils.onnx_export import export_onnx

def make_model():
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 2))

def test_export_creates_file(tmp_path):
    model = make_model()
    path = str(tmp_path / "model.onnx")
    export_onnx(model, input_shape=[1, 4], path=path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

def test_export_file_is_binary(tmp_path):
    model = make_model()
    path = str(tmp_path / "model.onnx")
    export_onnx(model, input_shape=[1, 4], path=path)
    with open(path, "rb") as f:
        data = f.read()
    assert isinstance(data, bytes)
    assert len(data) > 100

def test_export_sigmoid(tmp_path):
    model = Sequential(Linear(4, 8), Sigmoid(), Linear(8, 1))
    path = str(tmp_path / "sigmoid.onnx")
    export_onnx(model, input_shape=[1, 4], path=path)
    assert os.path.exists(path)

def test_export_tanh(tmp_path):
    model = Sequential(Linear(4, 8), Tanh(), Linear(8, 1))
    path = str(tmp_path / "tanh.onnx")
    export_onnx(model, input_shape=[1, 4], path=path)
    assert os.path.exists(path)

def test_export_dropout_becomes_identity(tmp_path):
    model = Sequential(Linear(4, 8), Dropout(p=0.5), Linear(8, 2))
    path = str(tmp_path / "dropout.onnx")
    export_onnx(model, input_shape=[1, 4], path=path)
    assert os.path.exists(path)

def test_export_unsupported_raises(tmp_path):
    from flux.nn.module import Module
    class UnsupportedLayer(Module):
        def forward(self, x): return x
    model = Sequential(UnsupportedLayer())
    path = str(tmp_path / "bad.onnx")
    with pytest.raises(ValueError, match="Unsupported"):
        export_onnx(model, input_shape=[1, 4], path=path)

def test_export_deep_model(tmp_path):
    model = Sequential(
        Linear(16, 32), ReLU(),
        Linear(32, 16), ReLU(),
        Linear(16, 8),  ReLU(),
        Linear(8, 4)
    )
    path = str(tmp_path / "deep.onnx")
    export_onnx(model, input_shape=[1, 16], path=path)
    assert os.path.getsize(path) > 500
