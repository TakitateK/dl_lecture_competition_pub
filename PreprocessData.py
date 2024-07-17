import os
import numpy as np
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler
from glob import glob

def preprocess_and_save(data_dir, output_dir, sfreq=200.0, l_freq=1.0, h_freq=50.0):
    os.makedirs(output_dir, exist_ok=True)
    
    scaler = StandardScaler()
    sample_data = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, f"{split}_X")
        output_split_dir = os.path.join(output_dir, f"{split}_X")
        os.makedirs(output_split_dir, exist_ok=True)
        
        file_paths = glob(os.path.join(split_dir, "*.npy"))
        for file_path in file_paths:
            X = np.load(file_path).astype(np.float64)
            X = filter_data(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, filter_length='auto')
            sample_data.append(X.astype(np.float32))
            
            file_name = os.path.basename(file_path)
            np.save(os.path.join(output_split_dir, file_name), X.astype(np.float32))

    sample_data = np.concatenate(sample_data, axis=1)
    scaler.fit(sample_data.T)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, f"{split}_X")
        file_paths = glob(os.path.join(split_dir, "*.npy"))
        for file_path in file_paths:
            X = np.load(file_path)
            X = scaler.transform(X.T).T
            np.save(file_path, X)

# 使用例
data_dir = "data"
output_dir = "preprocessed_data"
preprocess_and_save(data_dir, output_dir)
