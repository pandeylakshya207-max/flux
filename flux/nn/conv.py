import numpy as np
from .module import Module
from flux.tensor.tensor import Tensor

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        kH, kW = self.kernel_size
        k = 1.0 / np.sqrt(in_channels * kH * kW)
        self.weight = Tensor(np.random.uniform(-k, k, (out_channels, in_channels, kH, kW)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x):
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        s, p = self.stride, self.padding
        oH = (H + 2*p - kH) // s + 1
        oW = (W + 2*p - kW) // s + 1
        xp = np.pad(x.data, ((0,0),(0,0),(p,p),(p,p))) if p > 0 else x.data
        cols = np.zeros((N, C, kH, kW, oH, oW))
        for i in range(kH):
            for j in range(kW):
                cols[:, :, i, j, :, :] = xp[:, :, i:i+s*oH:s, j:j+s*oW:s]
        cols = cols.reshape(N, C*kH*kW, oH*oW)
        W_mat = self.weight.data.reshape(self.out_channels, -1)
        out_data = (W_mat @ cols).reshape(N, self.out_channels, oH, oW)
        if self.bias is not None:
            out_data += self.bias.data[None, :, None, None]
        children = (x, self.weight) + ((self.bias,) if self.bias else ())
        out = Tensor(out_data, requires_grad=True, _children=children, _op='conv2d')
        def _backward():
            if self.weight.requires_grad:
                self.weight._init_grad()
                dout = out.grad.reshape(N, self.out_channels, oH*oW)
                self.weight.grad += (dout @ cols.transpose(0,2,1)).sum(axis=0).reshape(self.weight.data.shape)
            if self.bias is not None and self.bias.requires_grad:
                self.bias._init_grad()
                self.bias.grad += out.grad.sum(axis=(0,2,3))
            if x.requires_grad:
                x._init_grad()
                dout = out.grad.reshape(N, self.out_channels, oH*oW)
                dcols = (W_mat.T @ dout).reshape(N, C, kH, kW, oH, oW)
                dxp = np.zeros_like(xp)
                for i in range(kH):
                    for j in range(kW):
                        dxp[:, :, i:i+s*oH:s, j:j+s*oW:s] += dcols[:, :, i, j, :, :]
                x.grad += dxp[:, :, p:-p, p:-p] if p > 0 else dxp
        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]

    def __repr__(self):
        return f'Conv2D({self.in_channels}, {self.out_channels}, kernel={self.kernel_size})'


class BatchNorm1D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x):
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean = self.running_mean
            var  = self.running_var
        xhat_data = (x.data - mean) / np.sqrt(var + self.eps)
        out_data  = self.gamma.data * xhat_data + self.beta.data
        out = Tensor(out_data, requires_grad=True, _children=(x, self.gamma, self.beta), _op='batchnorm1d')
        def _backward():
            N = x.data.shape[0]
            if self.gamma.requires_grad:
                self.gamma._init_grad()
                self.gamma.grad += (out.grad * xhat_data).sum(axis=0)
            if self.beta.requires_grad:
                self.beta._init_grad()
                self.beta.grad += out.grad.sum(axis=0)
            if x.requires_grad:
                x._init_grad()
                dxhat = out.grad * self.gamma.data
                dvar  = (-0.5 * dxhat * (x.data - mean) * (var + self.eps)**-1.5).sum(axis=0)
                dmean = (-dxhat / np.sqrt(var + self.eps)).sum(axis=0) + dvar*(-2*(x.data-mean)).mean(axis=0)
                x.grad += dxhat/np.sqrt(var+self.eps) + dvar*2*(x.data-mean)/N + dmean/N
        out._backward = _backward
        return out

    def parameters(self): return [self.gamma, self.beta]
    def train(self): self.training = True
    def eval(self):  self.training = False


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.data.shape[0], -1)
