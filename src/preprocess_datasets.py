import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from mne.filter import filter_data
from glob import glob

class PreprocessedThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "preprocessed_data", original_data_dir: str = "data"):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.original_data_dir = original_data_dir
        self.file_paths = glob(os.path.join(data_dir, f"{split}_X", "*.npy"))
        
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, i):
        X_path = self.file_paths[i]
        X = np.load(X_path).astype(np.float32)
        X = torch.from_numpy(X).float()

        subject_idx_path = os.path.join(self.original_data_dir, f"{self.split}_subject_idxs", os.path.basename(X_path))
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.original_data_dir, f"{self.split}_y", os.path.basename(X_path))
            y = torch.from_numpy(np.load(y_path))
            return X, y, subject_idx
        else:
            return X, subject_idx
