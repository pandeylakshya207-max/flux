import numpy as np
from .optimizer import Optimizer

class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + self.eps)
