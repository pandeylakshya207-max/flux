import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                g = self.velocity[i]
            p.data -= self.lr * g
