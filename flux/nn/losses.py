import numpy as np
from flux.tensor.tensor import Tensor

def mse_loss(pred, target):
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = pred - target
    return (diff * diff).mean()

def cross_entropy_loss(logits, targets):
    # logits: (N, C), targets: (N,) int class indices
    N = logits.data.shape[0]
    # stable softmax
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    # NLL loss
    log_probs = np.log(probs[np.arange(N), targets] + 1e-12)
    loss_val = -log_probs.mean()
    out = Tensor(loss_val, requires_grad=logits.requires_grad, _children=(logits,), _op='cross_entropy')
    def _backward():
        if logits.requires_grad:
            logits._init_grad()
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1
            grad /= N
            logits.grad += grad
    out._backward = _backward
    return out
