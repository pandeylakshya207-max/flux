import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.data.dataset import Dataset, TensorDataset
from flux.data.dataloader import DataLoader
from flux.data.trainer import Trainer
from flux.nn.linear import Linear
from flux.nn.sequential import Sequential
from flux.nn.activations import ReLU
from flux.nn.losses import mse_loss, cross_entropy_loss
from flux.optim.adam import Adam
from flux.optim.sgd import SGD

# --- TensorDataset ---

def test_tensor_dataset_len():
    X = Tensor(np.random.randn(100, 4))
    y = Tensor(np.random.randn(100, 1))
    ds = TensorDataset(X, y)
    assert len(ds) == 100

def test_tensor_dataset_getitem():
    X = Tensor(np.arange(20).reshape(10, 2).astype(float))
    y = Tensor(np.arange(10).astype(float))
    ds = TensorDataset(X, y)
    x0, y0 = ds[0]
    assert np.allclose(x0, [0, 1])
    assert y0 == 0.0

def test_tensor_dataset_mismatched_raises():
    X = Tensor(np.random.randn(10, 4))
    y = Tensor(np.random.randn(8, 1))
    with pytest.raises(AssertionError):
        TensorDataset(X, y)

# --- DataLoader ---

def test_dataloader_batch_count():
    X = Tensor(np.random.randn(100, 4))
    y = Tensor(np.random.randn(100, 1))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=10, shuffle=False)
    assert len(dl) == 10
    batches = list(dl)
    assert len(batches) == 10

def test_dataloader_batch_shape():
    X = Tensor(np.random.randn(50, 4))
    y = Tensor(np.random.randn(50, 1))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    Xb, yb = next(iter(dl))
    assert Xb.shape == (8, 4)
    assert yb.shape == (8, 1)

def test_dataloader_drop_last():
    X = Tensor(np.random.randn(105, 4))
    y = Tensor(np.random.randn(105, 1))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True)
    batches = list(dl)
    assert len(batches) == 10
    for Xb, _ in batches:
        assert Xb.shape[0] == 10

def test_dataloader_shuffle_changes_order():
    X = Tensor(np.arange(100).reshape(100, 1).astype(float))
    y = Tensor(np.zeros(100))
    ds = TensorDataset(X, y)
    np.random.seed(0)
    dl1 = DataLoader(ds, batch_size=100, shuffle=True)
    Xb1, _ = next(iter(dl1))
    np.random.seed(99)
    dl2 = DataLoader(ds, batch_size=100, shuffle=True)
    Xb2, _ = next(iter(dl2))
    assert not np.allclose(Xb1.data, Xb2.data)

def test_dataloader_covers_all_samples():
    X = Tensor(np.arange(50).reshape(50, 1).astype(float))
    y = Tensor(np.zeros(50))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=7, shuffle=False, drop_last=False)
    seen = []
    for Xb, _ in dl:
        seen.extend(Xb.data.flatten().tolist())
    assert len(seen) == 50

# --- Trainer ---

def make_regression_problem(n=200):
    np.random.seed(1)
    X = Tensor(np.random.randn(n, 4))
    y = Tensor((X.data @ np.array([1.0, -2.0, 0.5, 3.0])).reshape(-1, 1))
    ds = TensorDataset(X, y)
    train_ds = TensorDataset(
        Tensor(X.data[:160]), Tensor(y.data[:160])
    )
    val_ds = TensorDataset(
        Tensor(X.data[160:]), Tensor(y.data[160:])
    )
    return train_ds, val_ds

def test_trainer_fit_reduces_loss():
    train_ds, val_ds = make_regression_problem()
    train_dl = DataLoader(train_ds, batch_size=32)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    model = Linear(4, 1)
    opt = Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, mse_loss)
    history = trainer.fit(train_dl, val_dl, epochs=20, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert len(history["val_loss"]) == 20

def test_trainer_history_length():
    train_ds, _ = make_regression_problem()
    dl = DataLoader(train_ds, batch_size=32)
    model = Linear(4, 1)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, mse_loss)
    history = trainer.fit(dl, epochs=5, verbose=False)
    assert len(history["train_loss"]) == 5

def test_trainer_save_load(tmp_path):
    train_ds, _ = make_regression_problem()
    dl = DataLoader(train_ds, batch_size=32)
    model = Linear(4, 1)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, mse_loss)
    trainer.fit(dl, epochs=3, verbose=False)
    path = str(tmp_path / "ckpt.npz")
    trainer.save(path)
    # load into fresh model
    model2 = Linear(4, 1)
    trainer2 = Trainer(model2, SGD(model2.parameters(), lr=0.01), mse_loss)
    trainer2.load(path)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert np.allclose(p1.data, p2.data)

def test_trainer_checkpoint_dir(tmp_path):
    train_ds, _ = make_regression_problem()
    dl = DataLoader(train_ds, batch_size=32)
    model = Linear(4, 1)
    opt = SGD(model.parameters(), lr=0.01)
    ckpt_dir = str(tmp_path / "checkpoints")
    trainer = Trainer(model, opt, mse_loss, checkpoint_dir=ckpt_dir)
    trainer.fit(dl, epochs=3, verbose=False)
    import os
    files = os.listdir(ckpt_dir)
    assert len(files) == 3
