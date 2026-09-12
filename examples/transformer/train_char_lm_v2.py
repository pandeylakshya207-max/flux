import sys, os, numpy as np, urllib.request
sys.path.insert(0, '.')
from flux.tensor.tensor import Tensor
from flux.nn.module import Module
from flux.nn.linear import Linear
from flux.nn.normalization import LayerNorm, Dropout
from flux.nn.embedding import Embedding
from flux.nn.losses import cross_entropy_loss
from flux.optim.adam import Adam
from flux.optim.schedulers import CosineAnnealingLR

path = "data/shakespeare/input.txt"
os.makedirs("data/shakespeare", exist_ok=True)
if not os.path.exists(path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        path)

with open(path) as f: text = f.read()
chars      = sorted(set(text))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
print("Vocab:", vocab_size, " Data:", len(data))

block_size = 64
batch_size = 32
n_embd     = 128
n_head     = 4
n_layer    = 3
head_size  = n_embd // n_head
dropout_p  = 0.1
steps      = 5000
eval_every = 500

split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(d):
    ix = np.random.randint(0, len(d) - block_size, (batch_size,))
    X  = np.stack([d[i:i+block_size]     for i in ix])
    Y  = np.stack([d[i+1:i+block_size+1] for i in ix])
    return X, Y

class Head(Module):
    def __init__(self):
        self.q   = Linear(n_embd, head_size, bias=False)
        self.k   = Linear(n_embd, head_size, bias=False)
        self.v   = Linear(n_embd, head_size, bias=False)
        self.drop = Dropout(dropout_p)
    def forward(self, x):
        B, T, C = x.data.shape
        flat = x.data.reshape(-1, C)
        q = self.q(Tensor(flat)).data.reshape(B, T, head_size)
        k = self.k(Tensor(flat)).data.reshape(B, T, head_size)
        v = self.v(Tensor(flat)).data.reshape(B, T, head_size)
        scale = head_size ** -0.5
        att = np.einsum("bth,bsh->bts", q, k) * scale
        mask = np.tril(np.ones((T, T)))
        att = np.where(mask[None] == 1, att, -1e9)
        att = att - att.max(axis=-1, keepdims=True)
        exp_a = np.exp(att)
        att_w = exp_a / exp_a.sum(axis=-1, keepdims=True)
        out_data = np.einsum("bts,bsh->bth", att_w, v)
        return Tensor(out_data, requires_grad=True, _children=(x,), _op="head")
    def parameters(self):
        return self.q.parameters() + self.k.parameters() + self.v.parameters()

class MultiHead(Module):
    def __init__(self):
        self.heads = [Head() for _ in range(n_head)]
        self.proj  = Linear(n_embd, n_embd, bias=False)
        self.drop  = Dropout(dropout_p)
    def forward(self, x):
        cat = np.concatenate([h(x).data for h in self.heads], axis=-1)
        flat = Tensor(cat.reshape(-1, n_embd))
        out  = (flat @ self.proj.weight).data.reshape(x.data.shape)
        return Tensor(out, requires_grad=True, _children=(x,), _op="mha")
    def parameters(self):
        p = self.proj.parameters()
        for h in self.heads: p += h.parameters()
        return p

class FFN(Module):
    def __init__(self):
        self.fc1  = Linear(n_embd, 4*n_embd)
        self.fc2  = Linear(4*n_embd, n_embd)
        self.drop = Dropout(dropout_p)
    def forward(self, x):
        B, T, C = x.data.shape
        h = Tensor(x.data.reshape(-1, C))
        h = (h @ self.fc1.weight + self.fc1.bias).relu()
        h = (h @ self.fc2.weight + self.fc2.bias)
        return Tensor(h.data.reshape(B, T, C), requires_grad=True,
                      _children=(x,), _op="ffn")
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

class Block(Module):
    def __init__(self):
        self.ln1  = LayerNorm(n_embd)
        self.attn = MultiHead()
        self.ln2  = LayerNorm(n_embd)
        self.ffn  = FFN()
    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = Tensor(x.data + a.data, requires_grad=True, _children=(x, a), _op="res1")
        f = self.ffn(self.ln2(x))
        return Tensor(x.data + f.data, requires_grad=True, _children=(x, f), _op="res2")
    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters() +
                self.ln2.parameters() + self.ffn.parameters())

class GPT(Module):
    def __init__(self):
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Tensor(np.random.randn(block_size, n_embd)*0.01, requires_grad=True)
        self.blocks  = [Block() for _ in range(n_layer)]
        self.ln_f    = LayerNorm(n_embd)
        self.head    = Linear(n_embd, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        tok  = self.tok_emb(idx)
        x    = Tensor(tok.data + self.pos_emb.data[:T], requires_grad=True,
                      _children=(tok, self.pos_emb), _op="emb")
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        B2, T2, C = x.data.shape
        logits = self.head(Tensor(x.data.reshape(-1, C)))
        return logits
    def parameters(self):
        p = [self.pos_emb] + self.tok_emb.parameters() + self.head.parameters() + self.ln_f.parameters()
        for b in self.blocks: p += b.parameters()
        return p

model = GPT()
opt   = Adam(model.parameters(), lr=3e-3)
sched = CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
n_params = sum(p.data.size for p in model.parameters())
print("Parameters:", n_params)

def estimate_loss(d, n=30):
    losses = []
    for _ in range(n):
        X, Y = get_batch(d)
        logits = model(X)
        losses.append(float(cross_entropy_loss(logits, Y.flatten()).data))
    return float(np.mean(losses))

print("Training...")
for step in range(1, steps+1):
    X, Y = get_batch(train_data)
    logits = model(X)
    loss   = cross_entropy_loss(logits, Y.flatten())
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % eval_every == 0:
        tl = estimate_loss(train_data)
        vl = estimate_loss(val_data)
        print("Step", step, "- train:", round(tl,4), " val:", round(vl,4), " lr:", round(opt.lr,6))

print("\nGenerating sample...")
ctx = np.zeros((1,1), dtype=np.int32)
out_chars = []
for _ in range(400):
    ctx_crop = ctx[:, -block_size:]
    logits   = model(ctx_crop)
    last     = logits.data[-1]
    last    -= last.max()
    probs    = np.exp(last) / np.exp(last).sum()
    nxt      = np.random.choice(vocab_size, p=probs)
    ctx      = np.concatenate([ctx, [[nxt]]], axis=1)
    out_chars.append(itos[nxt])
print("".join(out_chars))
