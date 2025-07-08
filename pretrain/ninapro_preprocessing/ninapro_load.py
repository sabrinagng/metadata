import os
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

class NinaproLoader:
    """
    A class to load Ninapro dataset files.
    """
    def __init__(self, resample_length: int = 8192):
        self.resample_length = resample_length
        self.data = {}

    def inplace_unzip(self, subject_code: str, zip_dir: str = './data/Ninapro_original/DB2/'):
        """
        Extracts all files from a zip archive to a specified directory.
        
        Args:
            subject_code (str): The code of the subject whose data is to be extracted.
            zip_dir (str): The directory where the zip files are located.
        """
        zip_path = os.path.join(zip_dir, f'DB2_{subject_code}.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
        print(f'Extracted {zip_path} to {zip_dir}')
        
    def load(self, subject_code: str) -> dict:
        """
        Loads the Ninapro dataset files.
        
        Args:
            subject_code (str): The code of the subject whose data is to be loaded.
        
        Returns:
            dict: A dictionary containing the loaded data.
        """
        load_dir = f'./data/Ninapro_original/DB2/DB2_{subject_code}/'
        file_list = [f for f in os.listdir(load_dir) if os.path.isfile(os.path.join(load_dir, f))]
        
        all_emg = []
        all_restimulus = []

        for file_name in file_list:
            if file_name.endswith('.mat'):
                mat_data = loadmat(os.path.join(load_dir, file_name))
                if 'emg' in mat_data and 'restimulus' in mat_data:
                    all_emg.append(mat_data['emg'])
                    all_restimulus.append(mat_data['restimulus'].flatten())

        if not all_emg:
            self.data = {}

        self.data = {
            'emg': np.concatenate(all_emg, axis=0), 
            'restimulus': np.concatenate(all_restimulus, axis=0)
        }
        
    def segment(self):
        restimulus = self.data['restimulus'].flatten()
        emg = self.data['emg']
        
        non_zero_indices = np.where(restimulus != 0)[0]

        # Find contiguous blocks of non-zero indices
        if non_zero_indices.size > 0:
            splits = np.where(np.diff(non_zero_indices) > 1)[0] + 1
            contiguous_blocks = np.split(non_zero_indices, splits)
        else:
            contiguous_blocks = []

        emg_sections = [emg[block] for block in contiguous_blocks if block.size > 0]
        labels = [restimulus[block[0]] for block in contiguous_blocks if block.size > 0]

        return emg_sections, labels
    
    def align_section(self, emg_section):
        timestamp = np.arange(len(emg_section))
        resampled_data, resampled_timestamp = resample(emg_section, t=timestamp, num=self.resample_length)
        
        return resampled_data, resampled_timestamp
    
if __name__ == "__main__":
    loader = NinaproLoader()
    subject_code = 's2'
    zip_dir = './data/Ninapro_original/DB2/'
    
    loader.inplace_unzip(subject_code, zip_dir)
    loader.load(subject_code)
    
    emg_sections, labels = loader.segment()
    
    for i, (section, label) in enumerate(zip(emg_sections, labels)):
        resampled_emg, _ = loader.align_section(section)
        print(f'Section {i+1}: Label {label}, Resampled EMG shape: {resampled_emg.shape}')