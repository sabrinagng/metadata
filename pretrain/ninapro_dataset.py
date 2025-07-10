import os
import torch
from torch.utils.data import Dataset

class NinaproDataset(Dataset):
    def __init__(self, processed_data_dir: str):
        """
        Initializes the dataset by finding all preprocessed .pt files.
        
        Args:
            processed_data_dir (str): Path to the directory containing processed .pt files.
        """
        self.data_dir = processed_data_dir
        self.file_list = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        if not self.file_list:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}. Did you run the preprocessing script?")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads a single data sample from its .pt file on demand.
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, weights_only=True)
        return data['feature'], data['label']

if __name__ == '__main__':
    # Example usage with the new processed data directory
    # Make sure you have run the preprocess_data.py script first
    try:
        dataset = NinaproDataset(processed_data_dir='data/processed/DB2_emg_only_all_subjects')
        print(f"Total samples: {len(dataset)}")
        
        # Get a sample
        sample_feature, sample_label = dataset[0]
        print(f"Sample feature shape: {sample_feature.shape}, Sample label: {sample_label.item()}")
    except FileNotFoundError as e:
        print(e)