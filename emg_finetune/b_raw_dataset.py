
import json
from enum import Enum

import torch
from torch.utils.data import Dataset
from scipy.signal import resample

MODALITY = 'EMG'

class GestureEncoder(Enum):
    PAN = 0
    DUPLICATE = 1
    ZOOM_OUT = 2
    ZOOM_IN = 3
    MOVE = 4
    ROTATE = 5
    SELECT_SINGLE = 6
    DELETE = 7
    CLOSE = 8
    OPEN = 9

    @classmethod
    def from_string(cls, gesture_str: str):
        return cls[gesture_str.replace('-', '_').upper()]

class BRawDataset(Dataset):
    def __init__(self, json_path: str, subjects: list):
        super().__init__()
        self.subjects = subjects
        
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        
        self.feature = []
        self.label = []

        for s in subjects:
            gestures = all_data[s].keys()
            for g in gestures:
                indices = all_data[s][g].keys()
                for idx in indices:
                    sample_data = all_data[s][g][idx][MODALITY]
                    
                    resampled_data = resample(sample_data, num=8192, axis=1)
                    self.feature.append(torch.tensor(resampled_data, dtype=torch.float32))
                    self.label.append(GestureEncoder.from_string(g).value)
            
        self.feature = torch.stack(self.feature)  # [N, C, T]
        self.label = torch.tensor(self.label, dtype=torch.long)  # [N,]
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]