from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        import numpy as np
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __repr__(self):
        layers_str = '\n  '.join(repr(l) for l in self.layers)
        return f'Sequential(\n  {layers_str}\n)'
