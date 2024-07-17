import os
import numpy as np
import torch
from torch.utils.data import Dataset #追加
from sklearn.preprocessing import StandardScaler #追加
from mne.filter import resample, filter_data #追加
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data",sfreq: float = 200.0, l_freq: float = 1.0, h_freq: float = 50.0) -> None:
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        
        super().__init__()
        
        # データのスケーリング
        self.scaler = StandardScaler()
        self._fit_scaler()
        
    def _fit_scaler(self):
        # 初期スケーリングを実行
        sample_data = []
        for i in range(min(100, self.num_samples)):  # データセットの最初の1000サンプルを使用してスケーラーをフィット
            X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
            X = np.load(X_path).astype(np.float64)  # データを float64 に変換
            X = filter_data(X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, filter_length='auto')  # フィルタリング
            sample_data.append(X.astype(np.float32))  # データを float32 に変換  
        
        sample_data = np.concatenate(sample_data, axis=1)  # チャネルを維持したままサンプルを連結
        self.scaler.fit(sample_data.T)  # サンプルごとにスケーリング     

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path).astype(np.float64)  # データを float64 に変換
            
        # フィルタリング
        X = filter_data(X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, filter_length='auto')
        
        # スケーリング
        X = self.scaler.transform(X.T).T  

        X = torch.from_numpy(X.astype(np.float32)).float()  # データを float32 に変換して PyTorch のテンソルに変換      
        #X = torch.from_numpy(np.load(X_path))
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]