import numpy as np
from flux.tensor.tensor import Tensor

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            batch = [self.dataset[int(i)] for i in batch_idx]
            # batch is list of tuples -> transpose to tuple of arrays
            if isinstance(batch[0], tuple):
                yield tuple(
                    Tensor(np.stack([item[i] for item in batch]))
                    for i in range(len(batch[0]))
                )
            else:
                yield Tensor(np.stack(batch))
