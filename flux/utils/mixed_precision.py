import numpy as np
from flux.tensor.tensor import Tensor


class MixedPrecisionWrapper:
    def __init__(self, model, loss_scale=128.0):
        self.model      = model
        self.loss_scale = loss_scale
        self._overflow  = False

    def __call__(self, x):
        x16 = Tensor(x.data.astype(np.float16), requires_grad=x.requires_grad,
                     _children=(x,), _op="cast_fp16")
        def _backward():
            if x.requires_grad:
                x._init_grad()
                x.grad += x16.grad.astype(np.float64)
        x16._backward = _backward
        original_dtypes = {}
        for i, p in enumerate(self.model.parameters()):
            original_dtypes[i] = p.data.dtype
            p.data = p.data.astype(np.float16)
        out16 = self.model(x16)
        for i, p in enumerate(self.model.parameters()):
            p.data = p.data.astype(original_dtypes[i])
        out32 = Tensor(out16.data.astype(np.float64), requires_grad=out16.requires_grad,
                       _children=(out16,), _op="cast_fp32")
        def _backward32():
            if out16.requires_grad:
                out16._init_grad()
                out16.grad = out32.grad.astype(np.float16)
        out32._backward = _backward32
        return out32

    def scale_loss(self, loss):
        scaled = Tensor(loss.data * self.loss_scale,
                        requires_grad=loss.requires_grad,
                        _children=(loss,), _op="loss_scale")
        def _backward():
            if loss.requires_grad:
                loss._init_grad()
                loss.grad += scaled.grad * self.loss_scale
        scaled._backward = _backward
        return scaled

    def unscale_grads(self):
        self._overflow = False
        for p in self.model.parameters():
            if p.grad is not None:
                if np.any(~np.isfinite(p.grad)):
                    self._overflow = True
                    return
                p.grad = p.grad / self.loss_scale

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    @property
    def overflow(self):
        return self._overflow


def autocast(x):
    return Tensor(x.data.astype(np.float16)).data.astype(np.float64)


def fp16_safe(x):
    clipped = np.clip(x.data, -65504, 65504)
    return Tensor(clipped.astype(np.float16).astype(np.float64))
