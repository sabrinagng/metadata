import json
import os
from tqdm import tqdm
import torch
import numpy as np

def json2pt(json_path):
    """
    Reads a large JSON file containing Ninapro data and saves each EMG 
    segment as an individual .pt file.
    """
    if not os.path.exists(json_path):
        print(f"Error: Input file not found at {json_path}")
        return

    json_filename = os.path.basename(json_path)
    dir_name = os.path.splitext(json_filename)[0]
    output_dir = os.path.join('data/processed', dir_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    with open(json_path, 'r') as f:
        all_subject_data = json.load(f)

    sample_idx = 0
    for s in tqdm(all_subject_data, desc="Processing subjects"):
        all_gesture_segments = s['data']
        for g in tqdm(all_gesture_segments, desc="Processing gestures", leave=False):
            emg_segment = np.array(g['emg'])
            label = g['label']
            
            feature_tensor = torch.tensor(emg_segment, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Save each sample as a dictionary in a .pt file
            sample_path = os.path.join(output_dir, f'sample_{sample_idx}.pt')
            torch.save({'feature': feature_tensor, 'label': label_tensor}, sample_path)
            
            sample_idx += 1
            
    print(f"\nSuccessfully converted {sample_idx} samples.")
    print(f"Data saved in: {output_dir}")

if __name__ == '__main__':
    # --- Configuration ---
    # Make sure these paths are correct
    TRAIN_JSON_PATH = 'data/Ninapro/DB2_emg_only_all_subjects.json'
    TEST_JSON_PATH = 'data/Ninapro/DB3_emg_only_all_subjects.json'

    print("--- Processing Training Data ---")
    # json2pt(TRAIN_JSON_PATH)
    
    print("\n--- Processing Test Data ---")
    json2pt(TEST_JSON_PATH)