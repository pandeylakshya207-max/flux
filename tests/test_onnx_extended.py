import os
import numpy as np
import pytest
from flux.nn.sequential import Sequential
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.nn.conv import Conv2D, BatchNorm1D, Flatten
from flux.nn.normalization import Dropout
from flux.utils.onnx_export import export_onnx

def test_export_conv2d(tmp_path):
    model = Sequential(
        Conv2D(1, 4, 3, padding=1),
        ReLU(),
        Flatten(),
        Linear(4*8*8, 10)
    )
    path = str(tmp_path / "conv.onnx")
    export_onnx(model, input_shape=[1, 1, 8, 8], path=path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 200

def test_export_batchnorm(tmp_path):
    import numpy as np
    from flux.tensor.tensor import Tensor
    bn = BatchNorm1D(8)
    # populate running stats
    bn(Tensor(np.random.randn(16, 8)))
    model = Sequential(Linear(8, 8), bn, ReLU(), Linear(8, 4))
    path = str(tmp_path / "bn.onnx")
    export_onnx(model, input_shape=[1, 8], path=path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 100

def test_export_flatten(tmp_path):
    model = Sequential(Flatten(), Linear(32, 4))
    path = str(tmp_path / "flat.onnx")
    export_onnx(model, input_shape=[1, 32], path=path)
    assert os.path.exists(path)

def test_export_conv_no_bias(tmp_path):
    model = Sequential(
        Conv2D(3, 8, 3, padding=1, bias=False),
        ReLU(),
        Flatten(),
        Linear(8*4*4, 10)
    )
    path = str(tmp_path / "conv_nobias.onnx")
    export_onnx(model, input_shape=[1, 3, 4, 4], path=path)
    assert os.path.exists(path)
