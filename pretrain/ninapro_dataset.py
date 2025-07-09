import json

import numpy as np
import torch
from torch.utils.data import Dataset

class NinaproDataset(Dataset):
    def __init__(self, json_path: str = 'data/Ninapro/DB2_emg_only_all_subjects.json'):
        self.data = []
        
        with open(json_path, 'r') as f:
            all_subject_data = json.load(f)
            
        for s in all_subject_data:
            all_gesture_segments = s['data']
            for g in all_gesture_segments:
                emg_segment = np.array(g['emg'])
                label = g['label']
                self.data.append({
                    'feature': emg_segment,
                    'label': label
                })
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        feature = torch.tensor(item['feature'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return feature, label
    
if __name__ == '__main__':
    # Example usage
    dataset = NinaproDataset(json_path='data/Ninapro/DB2_emg_only_s1_to_s2.json')
    print(f"Total samples: {len(dataset)}")
    
    # Get a sample
    sample_feature, sample_label = dataset[0]
    print(f"Sample feature shape: {sample_feature.shape}, Sample label: {sample_label.item()}")