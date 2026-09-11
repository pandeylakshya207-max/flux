import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- properties ---

    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def size(self): return self.data.size
    @property
    def dtype(self): return self.data.dtype

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

    # --- grad init helper ---

    def _init_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    # --- ops ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                g = out.grad
                # sum over broadcasted dims
                for _ in range(g.ndim - self.data.ndim):
                    g = g.sum(axis=0)
                for i, (s, og) in enumerate(zip(self.data.shape, g.shape)):
                    if s == 1 and og != 1:
                        g = g.sum(axis=i, keepdims=True)
                self.grad += g
            if other.requires_grad:
                other._init_grad()
                g = out.grad
                for _ in range(g.ndim - other.data.ndim):
                    g = g.sum(axis=0)
                for i, (s, og) in enumerate(zip(other.data.shape, g.shape)):
                    if s == 1 and og != 1:
                        g = g.sum(axis=i, keepdims=True)
                other.grad += g
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                g = other.data * out.grad
                for _ in range(g.ndim - self.data.ndim):
                    g = g.sum(axis=0)
                for i, (s, og) in enumerate(zip(self.data.shape, g.shape)):
                    if s == 1 and og != 1:
                        g = g.sum(axis=i, keepdims=True)
                self.grad += g
            if other.requires_grad:
                other._init_grad()
                g = self.data * out.grad
                for _ in range(g.ndim - other.data.ndim):
                    g = g.sum(axis=0)
                for i, (s, og) in enumerate(zip(other.data.shape, g.shape)):
                    if s == 1 and og != 1:
                        g = g.sum(axis=i, keepdims=True)
                other.grad += g
        out._backward = _backward
        return out

    def __neg__(self): return self * Tensor(-1.0)
    def __sub__(self, other): return self + (-other if isinstance(other, Tensor) else Tensor(-other))
    def __rsub__(self, other): return Tensor(other) - self
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad, _children=(self,), _op=f'**{exp}')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='matmul')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._init_grad()
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other): return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='sum')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='mean')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                n = self.data.shape[axis] if axis is not None else self.data.size
                self.grad += np.broadcast_to(grad, self.data.shape) / n
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op='transpose')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    inv = np.argsort(axes)
                    self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    @property
    def T(self): return self.transpose()

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def slice(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _children=(self,), _op='slice')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad[idx] += out.grad
        out._backward = _backward
        return out

    def __getitem__(self, idx): return self.slice(idx)

    # --- backward ---

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # --- static constructors ---

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, requires_grad=False):
        if stop is None: start, stop = 0, start
        return Tensor(np.arange(start, stop, step), requires_grad=requires_grad)
