import sys, os, pickle, numpy as np, urllib.request, tarfile
sys.path.insert(0, '.')
from flux.tensor.tensor import Tensor
from flux.nn.conv import Conv2D, MaxPool2D, Flatten
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.nn.normalization import Dropout
from flux.nn.losses import cross_entropy_loss
from flux.optim.adam import Adam
from flux.optim.schedulers import StepLR

url  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
root = "data/cifar10"
tar_path = os.path.join(root, "cifar-10-python.tar.gz")
os.makedirs(root, exist_ok=True)
if not os.path.exists(tar_path):
    print("Downloading CIFAR-10...")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as t:
        t.extractall(root)

def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1,3,32,32).astype(np.float32)/255.0
    y = np.array(d[b"labels"], dtype=np.int32)
    return X, y

base = os.path.join(root, "cifar-10-batches-py")
batches = [load_batch(os.path.join(base, f"data_batch_{i}")) for i in range(1,6)]
Xtr = np.concatenate([b[0] for b in batches])
ytr = np.concatenate([b[1] for b in batches])
Xte, yte = load_batch(os.path.join(base, "test_batch"))

mean = Xtr.mean(axis=(0,2,3), keepdims=True)
std  = Xtr.std(axis=(0,2,3),  keepdims=True) + 1e-7
Xtr  = (Xtr - mean) / std
Xte  = (Xte - mean) / std
print("Train:", Xtr.shape, " Test:", Xte.shape)

def avgpool2x2(x):
    N,C,H,W = x.data.shape
    out_data = x.data.reshape(N,C,H//2,2,W//2,2).mean(axis=(3,5))
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="avgpool")
    def _backward():
        if x.requires_grad:
            x._init_grad()
            g = out.grad[:,:,:,np.newaxis,:,np.newaxis] * np.ones((1,1,1,2,1,2)) / 4.0
            x.grad += g.reshape(N,C,H,W)
    out._backward = _backward
    return out

class BetterCNN:
    def __init__(self):
        self.c1  = Conv2D(3,  32, 3, padding=1)
        self.c2  = Conv2D(32, 64, 3, padding=1)
        self.c3  = Conv2D(64, 64, 3, padding=1)
        self.flat = Flatten()
        self.fc1 = Linear(64*4*4, 256)
        self.fc2 = Linear(256, 10)
        self.relu = ReLU()
    def __call__(self, x):
        x = self.relu(self.c1(x))
        x = avgpool2x2(x)
        x = self.relu(self.c2(x))
        x = avgpool2x2(x)
        x = self.relu(self.c3(x))
        x = avgpool2x2(x)
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    def parameters(self):
        return (self.c1.parameters() + self.c2.parameters() +
                self.c3.parameters() + self.fc1.parameters() + self.fc2.parameters())
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

model = BetterCNN()
opt   = Adam(model.parameters(), lr=1e-3)
sched = StepLR(opt, step_size=5, gamma=0.5)
n_params = sum(p.data.size for p in model.parameters())
print("Parameters:", n_params)

def get_batches(X, y, bs=64):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), bs):
        b = idx[i:i+bs]
        yield Tensor(X[b]), Tensor(y[b].astype(np.int32))

def accuracy(X, y, bs=128):
    correct = total = 0
    for i in range(0, len(X), bs):
        logits = model(Tensor(X[i:i+bs]))
        preds  = logits.data.argmax(axis=1)
        correct += (preds == y[i:i+bs]).sum()
        total   += len(y[i:i+bs])
    return correct / total

print("Training full CIFAR-10 (50k)...")
for epoch in range(1, 21):
    losses = []
    for Xb, yb in get_batches(Xtr, ytr, 32):
        logits = model(Xb)
        loss   = cross_entropy_loss(logits, yb.data.flatten().astype(int))
        model.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    sched.step()
    acc = accuracy(Xte[:2000], yte[:2000])
    print("Epoch", epoch, "- loss:", round(float(np.mean(losses)),4),
          " test_acc:", round(acc*100,2), "%  lr:", round(opt.lr,6))

final = accuracy(Xte, yte)
print("Final full test accuracy:", round(final*100,2), "%")
