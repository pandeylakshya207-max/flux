import random
from .value import Value

class Neuron:
    def __init__(self, nin, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        return act  # linear

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation='relu'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, layer_sizes, activations=None):
        sizes = [nin] + layer_sizes
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 1) + ['sigmoid']
        self.layers = [Layer(sizes[i], sizes[i+1], activations[i])
                       for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
