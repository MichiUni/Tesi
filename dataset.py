import numpy as np
import torch
from torch.utils.data import Dataset

class FrequencyDataset(Dataset):
    def __init__(self, filepath):
        data = torch.tensor(np.loadtxt(filepath, delimiter=","), dtype=torch.float32)
        self.x = data[:, :2]
        self.y = data[:, 2:5]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
