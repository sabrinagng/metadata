import argparse
import numpy as np

def load_and_print_shape(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded {file_path}, shape: {data.shape}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load an npy file and print its shape.")
    parser.add_argument("file", help="Path to the .npy file")
    args = parser.parse_args()
    load_and_print_shape(args.file)