import os
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = np.load(path)
        self.file_number = self.files.shape[0]

    def __len__(self):
        return self.file_number

    def __getitem__(self, idx):
        input_seq = self.files[idx]/255.0
        return input_seq


class TestDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = np.load(path)
        self.file_number = self.files.shape[0]

    def __len__(self):
        return self.file_number

    def __getitem__(self, idx):
        input_seq = self.files[idx]/255.0
        return input_seq