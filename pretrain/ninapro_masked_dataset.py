import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class NinaproMaskedDataset(Dataset):
    def __init__(self, folders, window_len, step):
        self.window_len = window_len
        self.step = step
        self.emg_data = []
        self.window_info = []

        for folder in folders:
            files = sorted(glob(os.path.join(folder, '*.npy')))
            print(f"Found {len(files)} .npy files in {folder}", flush=True)
            for path in files:
                # Load EMG data as memory-mapped array
                emg = np.load(path, mmap_mode='r')
                self.emg_data.append(emg)
                file_idx = len(self.emg_data) - 1
                
                T, _ = emg.shape  # T: number of time steps, _: number of channels
                cnt = 0
                # Create window information without loading all data into memory
                for st in range(0, T - self.window_len + 1, self.step):
                    self.window_info.append((file_idx, st))
                    cnt += 1
                print(f"  {os.path.basename(path)} → {cnt} windows", flush=True)
        
        if len(self.window_info) > 0:
            # Get shape from one sample to display
            sample_shape = self[0].shape
            print(f"Total windows: {len(self)} with a shape of {sample_shape}\n", flush=True)
        else:
            print("Total windows: 0\n", flush=True)

    def __len__(self):
        return len(self.window_info)

    def __getitem__(self, idx):
        file_idx, st = self.window_info[idx]
        emg = self.emg_data[file_idx]
        
        # Slice the window from the memory-mapped array
        seg = emg[st:st + self.window_len]
        # Normalize per-channel (zero mean, unit variance)
        seg = (seg - seg.mean(0)) / (seg.std(0) + 1e-6)
        # Return as torch tensor with shape (C, T) for PyTorch models
        return torch.from_numpy(seg.T.astype(np.float32))

if __name__ == '__main__':
    # Parameters from train.py
    DATA_FOLDERS = ['./autodl-tmp/data/DB2_all/']
    WINDOW_LEN = 400
    STEP = 100

    # Create dataset instance
    dataset = NinaproMaskedDataset(folders=DATA_FOLDERS, window_len=WINDOW_LEN, step=STEP)

    # Check if the dataset is empty
    if len(dataset) > 0:
        # Get one sample
        sample = dataset[0]
        # Print shape and data type
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
    else:
        print("Dataset is empty. Could not retrieve a sample.")