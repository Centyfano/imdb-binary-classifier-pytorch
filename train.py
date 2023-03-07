import torch
from torch.utils.data import Dataset
from torchaudio import datasets
import os

class ImdbDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        
