import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of (input_tensor, target_index) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, target_index = self.data[idx]
        return input_tensor, target_index
