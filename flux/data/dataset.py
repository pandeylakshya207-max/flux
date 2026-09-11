class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(len(t.data) == len(tensors[0].data) for t in tensors)
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0].data)

    def __getitem__(self, idx):
        return tuple(t.data[idx] for t in self.tensors)
