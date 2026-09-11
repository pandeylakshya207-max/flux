import os
import struct
import gzip
import urllib.request
import numpy as np
from .dataset import Dataset

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def _download(root):
    os.makedirs(root, exist_ok=True)
    for name, fname in FILES.items():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(MNIST_URL + fname, path)

def _load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h*w).astype(np.float32) / 255.0

def _load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

class MNISTDataset(Dataset):
    def __init__(self, root="data/mnist", train=True, download=True):
        if download:
            _download(root)
        if train:
            self.images = _load_images(os.path.join(root, FILES["train_images"]))
            self.labels = _load_labels(os.path.join(root, FILES["train_labels"]))
        else:
            self.images = _load_images(os.path.join(root, FILES["test_images"]))
            self.labels = _load_labels(os.path.join(root, FILES["test_labels"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
