import sys
import numpy as np
sys.path.insert(0, '.')

from flux.data.mnist import MNISTDataset
from flux.data.dataloader import DataLoader
from flux.data.dataset import TensorDataset
from flux.tensor.tensor import Tensor
from flux.nn.sequential import Sequential
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.nn.losses import cross_entropy_loss
from flux.optim.adam import Adam

print("Loading MNIST...")
train_ds = MNISTDataset(root="data/mnist", train=True,  download=True)
test_ds  = MNISTDataset(root="data/mnist", train=False, download=True)
print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False)

model = Sequential(
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10)
)

opt = Adam(model.parameters(), lr=1e-3)

def accuracy(dl):
    correct = total = 0
    for X, y in dl:
        logits = model(X)
        preds = logits.data.argmax(axis=1)
        correct += (preds == y.data.astype(int)).sum()
        total += len(y.data)
    return correct / total

print("Training...")
for epoch in range(1, 11):
    losses = []
    for X, y in train_dl:
        logits = model(X)
        loss = cross_entropy_loss(logits, y.data.astype(int).flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    train_loss = np.mean(losses)
    acc = accuracy(test_dl)
    print(f"Epoch {epoch}/10 - loss: {train_loss:.4f}  test_acc: {acc*100:.2f}%")

final_acc = accuracy(test_dl)
print(f"\nFinal test accuracy: {final_acc*100:.2f}%")
if final_acc >= 0.97:
    print("TARGET MET: >97%")
else:
    print(f"Below target - need {(0.97-final_acc)*100:.2f}% more")
