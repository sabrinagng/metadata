import pickle
import pandas as pd
import os

# Adjust the path to point to the correct location of the pickle file
PICKLE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "noFE_windowed_segraw_allEMG.pkl")

try:
    # Load the DataFrame from the pickle file
    df = pd.read_pickle(PICKLE_PATH)
    print("Pickle file loaded successfully.")

    # Check if the data column exists
    if "windowed_ts_data" in df.columns:
        # Get the first data sample (the numpy array)
        first_sample = df["windowed_ts_data"].iloc[0]
        
        # The shape is usually (channels, timesteps)
        num_timesteps, num_channels = first_sample.shape
        
        print(f"\nData shape of the first sample: {first_sample.shape}")
        print(f"Number of Channels: {num_channels}")
        print(f"Number of Timesteps: {num_timesteps}")
    else:
        print("Column 'windowed_ts_data' not found in the DataFrame.")

except FileNotFoundError:
    print(f"ERROR: File not found at '{PICKLE_PATH}'")
except Exception as e:
    print(f"An error occurred: {e}")