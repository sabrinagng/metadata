import sys
import numpy as np
import os


def inspect_npy_file(file_path):
    if file_path.endswith('.npy'):
        arr = np.load(file_path, allow_pickle=True)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Type: npy (single array)")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
    elif file_path.endswith('.npz'):
        data = np.load(file_path, allow_pickle=True)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Type: npz (multiple arrays)")
        for name in data.files:
            arr = data[name]
            print(f"Array name: {name}, Shape: {arr.shape}, Dtype: {arr.dtype}")
    else:
        print("Unsupported file type. Please provide a .npy or .npz file.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python npy_inspector.py <file.npy or file.npz>")
        sys.exit(1)
    file_path = sys.argv[1]
    inspect_npy_file(file_path)


if __name__ == "__main__":
    main()
