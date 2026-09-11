import random
from flux.engine.mlp import MLP

def test_xor_convergence():
    random.seed(42)
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,    1,    1,    0   ]

    model = MLP(2, [4, 1], activations=['tanh', 'sigmoid'])

    for epoch in range(2000):
        # forward
        preds = [model(x) for x in X]
        loss = sum((p - t)**2 for p, t in zip(preds, y))

        # backward
        model.zero_grad()
        loss.backward()

        # SGD step
        for p in model.parameters():
            p.data -= 0.1 * p.grad

    # final predictions
    preds = [model(x).data for x in X]
    # XOR: threshold at 0.5
    correct = sum((p > 0.5) == bool(t) for p, t in zip(preds, y))
    assert correct == 4, f"XOR failed: preds={[round(p,3) for p in preds]}"
    assert loss.data < 0.01, f"Loss too high: {loss.data:.4f}"
