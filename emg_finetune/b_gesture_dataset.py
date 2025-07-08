import torch
from torch.utils.data import Dataset
import numpy as np

class BGestureDataset(Dataset):
    def __init__(self, df_sub):
        self.features = np.stack(df_sub['windowed_ts_data'].to_numpy(), axis=0)  # (N, T, C)
        self.labels = df_sub["Gesture_ID"].to_numpy()
        
        self.label2idx = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Get the string label, map it to an integer index, then create the tensor
        label_str = self.labels[idx]
        label_idx = self.label2idx[label_str]
        y = torch.tensor(label_idx, dtype=torch.long)
        
        # Instance-wise normalization
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        
        return x, y