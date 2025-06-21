import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Assumes the data is in the parent directory's 'data' folder
INPUT_PICKLE_PATH = os.path.join("..", "data", "noFE_windowed_segraw_allEMG.pkl")
OUTPUT_PICKLE_PATH = os.path.join("..", "data", "noFE_windowed_segraw_allEMG_64ch_12ts.pkl")

# Timesteps to remove (using 0-based indexing for arrays)
# This will remove the 9th, 11th, 13th, and 15th timestep
TIMESTEPS_TO_REMOVE = [8, 10, 12, 14]

def modify_pickle_timesteps():
    """
    Loads a pickle file, removes specified timesteps from the data,
    and saves it to a new file.
    """
    print(f"Attempting to load data from: {INPUT_PICKLE_PATH}")
    try:
        df = pd.read_pickle(INPUT_PICKLE_PATH)
        print("Data loaded successfully.")

        data_column = "windowed_ts_data"
        if data_column not in df.columns:
            raise ValueError(f"Error: Column '{data_column}' not found in the DataFrame.")

        # --- Timestep Removal ---
        original_shape = df[data_column].iloc[0].shape
        print(f"Original data shape (first sample): {original_shape}")
        
        # Check the original shape
        if original_shape != (64, 16):
             print(f"Warning: Expected shape (64, 16), but found {original_shape}. Aborting modification.")
             return

        print(f"Removing timesteps at indices: {TIMESTEPS_TO_REMOVE}")
        # Use .apply() with a lambda function to remove columns (axis=1) from each numpy array
        df[data_column] = df[data_column].apply(
            lambda x: np.delete(x, TIMESTEPS_TO_REMOVE, axis=1)
        )

        # --- Verification and Saving ---
        new_shape = df[data_column].iloc[0].shape
        print(f"New data shape (first sample): {new_shape}")

        # Check the new shape
        if new_shape == (64, 12):
            print("Timestep removal successful. New shape is (64, 12).")
            print(f"Saving modified data to: {OUTPUT_PICKLE_PATH}")
            df.to_pickle(OUTPUT_PICKLE_PATH)
            print("New pickle file saved successfully.")
        else:
            print(f"Error: New shape is {new_shape}, which is not the expected (64, 12). File not saved.")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_PICKLE_PATH}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    modify_pickle_timesteps()