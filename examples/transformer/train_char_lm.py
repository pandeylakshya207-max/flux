import sys, os, numpy as np
sys.path.insert(0, '.')
from flux.tensor.tensor import Tensor
from flux.nn.module import Module
from flux.nn.linear import Linear
from flux.nn.losses import cross_entropy_loss
from flux.optim.adam import Adam

# --- data ---
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = "data/shakespeare/input.txt"
os.makedirs("data/shakespeare", exist_ok=True)
if not os.path.exists(path):
    import urllib.request
    print("Downloading Shakespeare...")
    urllib.request.urlretrieve(url, path)

with open(path, 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
print(f"Vocab: {vocab_size}  Data: {len(data):,} chars")

block_size = 32
batch_size = 64
n_embd     = 64
n_head     = 4
n_layer    = 2
head_size  = n_embd // n_head
lr         = 3e-3
epochs     = 2000
eval_every = 200

split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(d):
    ix = np.random.randint(0, len(d) - block_size, (batch_size,))
    X  = np.stack([d[i:i+block_size]   for i in ix])
    Y  = np.stack([d[i+1:i+block_size+1] for i in ix])
    return X, Y

# --- model ---

class Embedding(Module):
    def __init__(self, vocab, dim):
        self.W = Tensor(np.random.randn(vocab, dim) * 0.02, requires_grad=True)
    def forward(self, idx):
        data = self.W.data[idx]
        out  = Tensor(data, requires_grad=True, _children=(self.W,), _op='embed')
        def _backward():
            if self.W.requires_grad:
                self.W._init_grad()
                np.add.at(self.W.grad, idx, out.grad)
        out._backward = _backward
        return out
    def parameters(self): return [self.W]

class Head(Module):
    def __init__(self):
        self.q = Linear(n_embd, head_size, bias=False)
        self.k = Linear(n_embd, head_size, bias=False)
        self.v = Linear(n_embd, head_size, bias=False)
        self.mask = np.tril(np.ones((block_size, block_size)))
    def forward(self, x):
        T = x.data.shape[1]
        q = Tensor(x.data.reshape(-1, n_embd)) @ self.q.weight
        k = Tensor(x.data.reshape(-1, n_embd)) @ self.k.weight
        v = Tensor(x.data.reshape(-1, n_embd)) @ self.v.weight
        q = Tensor(q.data.reshape(-1, T, head_size))
        k = Tensor(k.data.reshape(-1, T, head_size))
        v = Tensor(v.data.reshape(-1, T, head_size))
        scale = head_size ** -0.5
        # attention scores (B, T, T)
        att = np.einsum('bth,bsh->bts', q.data, k.data) * scale
        att[:, self.mask[:T,:T]==0] = -1e9
        att = att - att.max(axis=-1, keepdims=True)
        exp_att = np.exp(att)
        att_w = exp_att / exp_att.sum(axis=-1, keepdims=True)
        out_data = np.einsum('bts,bsh->bth', att_w, v.data)
        return Tensor(out_data, requires_grad=True, _children=(q,k,v), _op='attn')
    def parameters(self):
        return self.q.parameters() + self.k.parameters() + self.v.parameters()

class MultiHead(Module):
    def __init__(self):
        self.heads = [Head() for _ in range(n_head)]
        self.proj  = Linear(n_embd, n_embd, bias=False)
    def forward(self, x):
        cats = np.concatenate([h(x).data for h in self.heads], axis=-1)
        out  = Tensor(cats, requires_grad=True, _children=tuple(x for x in []), _op='mh')
        return Tensor((Tensor(cats) @ self.proj.weight).data, requires_grad=True,
                      _children=(self.proj.weight,), _op='mhproj')
    def parameters(self):
        p = self.proj.parameters()
        for h in self.heads: p += h.parameters()
        return p

class FFN(Module):
    def __init__(self):
        self.fc1 = Linear(n_embd, 4*n_embd)
        self.fc2 = Linear(4*n_embd, n_embd)
    def forward(self, x):
        B, T, C = x.data.shape
        flat = Tensor(x.data.reshape(-1, C))
        h    = (flat @ self.fc1.weight + self.fc1.bias).relu()
        out  = (h    @ self.fc2.weight + self.fc2.bias)
        return Tensor(out.data.reshape(B, T, C), requires_grad=True,
                      _children=(self.fc1.weight, self.fc2.weight), _op='ffn')
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

class Block(Module):
    def __init__(self):
        self.attn = MultiHead()
        self.ffn  = FFN()
    def forward(self, x):
        a = self.attn(x)
        x2 = Tensor(x.data + a.data, requires_grad=True, _children=(x, a), _op='res1')
        f  = self.ffn(x2)
        return Tensor(x2.data + f.data, requires_grad=True, _children=(x2, f), _op='res2')
    def parameters(self):
        return self.attn.parameters() + self.ffn.parameters()

class GPT(Module):
    def __init__(self):
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Tensor(np.random.randn(block_size, n_embd) * 0.01, requires_grad=True)
        self.blocks  = [Block() for _ in range(n_layer)]
        self.head    = Linear(n_embd, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        x = Tensor(self.tok_emb(idx).data + self.pos_emb.data[:T], requires_grad=True,
                   _children=(self.tok_emb.W, self.pos_emb), _op='emb')
        for blk in self.blocks:
            x = blk(x)
        B2, T2, C = x.data.shape
        logits = Tensor(x.data.reshape(-1, C)) @ self.head.weight + self.head.bias
        return logits
    def parameters(self):
        p = [self.pos_emb] + self.tok_emb.parameters() + self.head.parameters()
        for b in self.blocks: p += b.parameters()
        return p

model = GPT()
opt   = Adam(model.parameters(), lr=lr)
n_params = sum(p.data.size for p in model.parameters())
print(f"Parameters: {n_params:,}")

def estimate_loss(d, steps=20):
    losses = []
    for _ in range(steps):
        X, Y = get_batch(d)
        logits = model(X)
        loss = cross_entropy_loss(logits, Y.flatten())
        losses.append(float(loss.data))
    return np.mean(losses)

print("Training...")
for step in range(1, epochs+1):
    X, Y = get_batch(train_data)
    logits = model(X)
    loss = cross_entropy_loss(logits, Y.flatten())
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % eval_every == 0:
        tl = estimate_loss(train_data)
        vl = estimate_loss(val_data)
        print(f"Step {step:4d} - train_loss: {tl:.4f}  val_loss: {vl:.4f}")

# generate sample
print("\n--- Sample ---")
ctx = np.zeros((1, 1), dtype=np.int32)
out_chars = []
for _ in range(300):
    ctx_crop = ctx[:, -block_size:]
    logits = model(ctx_crop)
    last   = logits.data[-1]
    last  -= last.max()
    probs  = np.exp(last) / np.exp(last).sum()
    next_c = np.random.choice(vocab_size, p=probs)
    ctx    = np.concatenate([ctx, [[next_c]]], axis=1)
    out_chars.append(itos[next_c])
print("".join(out_chars))
