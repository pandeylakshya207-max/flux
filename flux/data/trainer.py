import os
import numpy as np
from flux.tensor.tensor import Tensor

class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler=None, checkpoint_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader):
        losses = []
        for batch in dataloader:
            X, y = batch
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.data))
        return float(np.mean(losses))

    def evaluate(self, dataloader):
        losses = []
        for batch in dataloader:
            X, y = batch
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            losses.append(float(loss.data))
        return float(np.mean(losses))

    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            msg = f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                msg += f"  val_loss: {val_loss:.4f}"
            if self.scheduler is not None:
                self.scheduler.step()
            if verbose:
                print(msg)
            if self.checkpoint_dir is not None:
                self.save(os.path.join(self.checkpoint_dir, f"epoch_{epoch}.npz"))
        return self.history

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        params = {f"param_{i}": p.data for i, p in enumerate(self.model.parameters())}
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self.model.parameters()):
            p.data = data[f"param_{i}"]
