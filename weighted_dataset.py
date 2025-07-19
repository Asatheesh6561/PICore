import torch


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, gamma):
        self.dataset = dataset
        self.indices = indices
        self.gamma = gamma

    def __getitem__(self, idx):
        data, target = (
            self.dataset[self.indices[idx]]["x"],
            self.dataset[self.indices[idx]]["y"],
        )
        weight = self.gamma[self.indices[idx]]
        return data, target, weight

    def __len__(self):
        return len(self.indices)
