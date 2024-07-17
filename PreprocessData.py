import os
import numpy as np
import mne
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler
from glob import glob

# ロギングレベルを抑制
mne.set_log_level('WARNING')

def preprocess_and_save(data_dir, output_dir, sfreq=200.0, l_freq=1.0, h_freq=50.0, batch_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    
    scaler = StandardScaler()
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, f"{split}_X")
        output_split_dir = os.path.join(output_dir, f"{split}_X")
        os.makedirs(output_split_dir, exist_ok=True)
        
        file_paths = glob(os.path.join(split_dir, "*.npy"))
        sample_data = []
        
        # バッチ処理でスケーラーをフィット
        for start in range(0, len(file_paths), batch_size):
            end = start + batch_size
            batch_paths = file_paths[start:end]
            for file_path in batch_paths:
                X = np.load(file_path).astype(np.float64)
                X = filter_data(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, filter_length='auto')
                sample_data.append(X.astype(np.float32))
            sample_data = np.concatenate(sample_data, axis=1)
            scaler.partial_fit(sample_data.T)
            sample_data = []

        # フィルタリングとスケーリングを行い、前処理済みデータを保存
        for file_path in file_paths:
            X = np.load(file_path).astype(np.float64)
            X = filter_data(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, filter_length='auto')
            X = scaler.transform(X.T).T
            file_name = os.path.basename(file_path)
            np.save(os.path.join(output_split_dir, file_name), X.astype(np.float32))

# 使用例
data_dir = "data"
output_dir = "preprocessed_data"
batch_size = 1000

preprocess_and_save(data_dir, output_dir, batch_size=batch_size)
