import numpy as np
from .optimizer import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        super().__init__(parameters, lr)
        self.eps = eps
        self.G = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.G[i] += p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.G[i]) + self.eps)
