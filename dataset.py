import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class JSDataset(Dataset):
    def __init__(self, X, y, length, ID):
        self.data = X
        self.label = y
        self.length = length
        self.ID = ID
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.length[idx], self.ID[idx]
    
    def __len__(self):
        return len(self.data)