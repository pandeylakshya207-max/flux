import numpy as np

class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class ExponentialLR:
    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.t = 0

    def step(self):
        self.t += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.t / self.T_max)
        )

def clip_grad_norm(parameters, max_norm):
    total_norm = np.sqrt(sum(
        np.sum(p.grad ** 2) for p in parameters if p.grad is not None
    ))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale
    return total_norm
