import pickle
import sys
import time
from pprint import pprint

def inspect_pickle(file_path):
    print(f"Attempting to open: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            print("File opened. Loading pickle data... (This may take a while)")
            start_time = time.time()
            data = pickle.load(f)
            end_time = time.time()
            print(f"Pickle loaded in {end_time - start_time:.2f} seconds.")
        
        print("\n--- Pickle Content ---")
        pprint(data)
        print("--- End of Content ---")

    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pickle_file>")
        sys.exit(1)
    inspect_pickle(sys.argv[1])
    # Save the loaded pickle content to a file
    try:
        with open(sys.argv[1], 'rb') as f:
            data = pickle.load(f)
        output_file = sys.argv[1] + ".txt"
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(str(data))
        print(f"Pickle content saved to {output_file}")
    except Exception as e:
        print(f"Failed to save pickle content: {e}")