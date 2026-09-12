# flux

A deep learning training framework built from scratch in Python.

**Only dependency: NumPy.** No PyTorch, no TensorFlow autograd.

## What this is

flux is a full training engine - autograd, tensors, layers, optimizers, and a real training loop - built ground-up to understand how frameworks like PyTorch actually work internally.

## Architecture

    flux/
    flux/engine/    - scalar autograd (Value, backward)
    flux/tensor/    - n-dim tensors with strides + broadcasting
    flux/nn/        - layers: Linear, Conv2D, BatchNorm, LayerNorm, Dropout, Embedding
    flux/optim/     - SGD, Adam, AdaGrad, RMSProp, LR schedulers
    flux/data/      - Dataset, DataLoader, MNIST loader
    flux/utils/     - graph viz, ONNX export, memory profiler
    examples/       - MNIST, Transformer char-LM, CIFAR-10 CNN
    tests/
    docs/

## Roadmap

- [x] Phase 0: Repo + CI
- [ ] Phase 1: Scalar autograd engine
- [ ] Phase 2: N-dim tensor engine
- [ ] Phase 3: NN layers
- [ ] Phase 4: Optimizers
- [ ] Phase 5: DataLoader + training loop
- [ ] Phase 6: Train real models (MNIST >97%, Transformer, CIFAR-10 >70%)
- [ ] Phase 7: Production signal (op fusion, ONNX export, memory profiler)

## Results

| Model | Dataset | Accuracy |
|-------|---------|----------|
| MLP | MNIST | 97.32% (10 epochs, Adam lr=1e-3) |
| CNN | CIFAR-10 | TBD |
| Transformer char-LM | Shakespeare | val_loss 2.79 (2000 steps, 109K params) |
