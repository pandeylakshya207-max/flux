# flux

A deep learning training framework built from scratch in pure Python + NumPy.

**Zero ML framework dependencies.** No PyTorch, no TensorFlow, no JAX.

## Results

| Model | Dataset | Result |
|-------|---------|--------|
| MLP (784-256-128-10) | MNIST | 97.32% test accuracy (10 epochs, Adam) |
| Transformer char-LM (109K params) | Shakespeare | val_loss 2.79 (2000 steps) |
| CNN (Conv2D x2 + Linear) | CIFAR-10 | 55-57% (10k subset, CPU only) |

## Architecture

    flux/
    flux/engine/     Scalar autograd: Value, Neuron, Layer, MLP
    flux/tensor/     N-dim tensor engine with broadcasting + autograd
    flux/nn/         Linear, Conv2D, BatchNorm1D, LayerNorm, Dropout,
                     ReLU/Sigmoid/Tanh, Sequential, MSELoss, CrossEntropyLoss
    flux/optim/      SGD+momentum, Adam, AdaGrad, RMSProp,
                     StepLR, ExponentialLR, CosineAnnealingLR, grad clipping
    flux/data/       Dataset, TensorDataset, DataLoader (shuffle+drop_last),
                     Trainer (fit/save/load/checkpoint), MNISTDataset
    flux/utils/      Fused linear+relu/sigmoid, computation graph DOT viz,
                     memory profiler, ONNX export
    examples/
    examples/mnist/          MLP training script
    examples/transformer/    Transformer char-LM on Shakespeare
    examples/benchmarks/     CIFAR-10 CNN

## What I built (in order)

### Phase 1 - Scalar autograd engine
- Value class: tracks data, grad, and a _backward closure
- Implements +, -, *, /, **, exp, log, relu, sigmoid, tanh
- backward() does topological sort + reverse-mode autodiff
- Proof: MLP learns XOR to <0.01 loss in 2000 steps

### Phase 2 - N-dim tensor engine
- Tensor class backed by NumPy arrays
- Full broadcasting support in add/mul backward passes
- matmul, sum, mean, reshape, transpose, slice - all with backward
- Numerical gradient checker validates every op

### Phase 3 - NN layers
- Module base class with parameters() + zero_grad()
- Linear with Kaiming uniform init
- Conv2D implemented via im2col (no scipy)
- BatchNorm1D, LayerNorm with full backward derivation
- Dropout with inverted scaling
- Sequential container
- MSELoss and CrossEntropyLoss (numerically stable softmax)

### Phase 4 - Optimizers (from original papers)
- SGD with momentum and weight decay
- Adam with bias correction (Kingma & Ba 2014)
- AdaGrad with accumulated gradient squares
- RMSProp with exponential moving average
- StepLR, ExponentialLR, CosineAnnealingLR schedulers
- Global gradient norm clipping

### Phase 5 - DataLoader + Trainer
- Dataset / TensorDataset abstractions
- DataLoader with shuffle, batch_size, drop_last
- Trainer: fit(), evaluate(), save(), load(), checkpoint_dir
- MNISTDataset: downloads + parses IDX format

### Phase 6 - Real model training
- MLP on MNIST: 97.32% in 10 epochs
- Transformer char-LM: custom Embedding, multi-head attention, FFN, residuals
- CIFAR-10 CNN: Conv2D pipeline with avgpool

### Phase 7 - Production utilities
- Operator fusion: fused_linear_relu / fused_linear_sigmoid
  Single kernel for matmul+activation, one backward pass
- Computation graph visualizer: exports DOT format for any expression
- Memory profiler: per-layer output shape, MB, and forward time
- ONNX export: hand-written protobuf encoder, no onnx package needed

## Test suite

    148 tests across 8 test files
    pytest tests/ - all pass in ~1.3s

## Design decisions

**Why scalar autograd first?**
Forces you to understand that a neural net is just a DAG of scalar operations.
The tensor engine is then obviously the same idea, scaled up.

**Why im2col for Conv2D?**
Transforms convolution into a single matmul. Same approach used by cuDNN.
Makes the backward pass straightforward: dW = dout @ cols.T

**Why write ONNX protobuf by hand?**
The onnx package is a 100MB dependency. ONNX is just protobuf.
Writing the encoder from scratch means zero runtime dependencies.

**Why not use NumPy autograd (like autograd library)?**
The point is to understand what happens inside. Every backward closure
is written explicitly so there are no surprises.

## Install

    pip install numpy
    pip install -e .
    pytest

## Usage

    from flux.tensor import Tensor
    from flux.nn import Sequential, Linear, ReLU
    from flux.nn.losses import cross_entropy_loss
    from flux.optim import Adam

    model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
    opt   = Adam(model.parameters(), lr=1e-3)

    for X, y in dataloader:
        logits = model(X)
        loss   = cross_entropy_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
