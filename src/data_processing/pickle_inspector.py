import pandas as pd
import sys
import os

def inspect_dataframe_pickle(file_path):
    """
    Loads a pickle file expected to contain a pandas DataFrame,
    then prints its dimensions and column names (keys).
    """
    print("-" * 50) # Separator for multiple files
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Inspecting pickle file: {file_path}\n")
    
    try:
        # Load the data directly into a pandas DataFrame
        df = pd.read_pickle(file_path)

        if isinstance(df, pd.DataFrame):
            # Get dimensions
            num_rows, num_cols = df.shape
            
            # Get column names (keys)
            column_names = df.columns.tolist()
            
            print(f"Successfully loaded a pandas DataFrame.")
            print(f" - Number of rows: {num_rows}")
            print(f" - Number of columns: {num_cols}")
            print(f" - Column Names (Keys): {column_names}")
            
        else:
            print(f"The pickle file does not contain a pandas DataFrame.")
            print(f"Data type: {type(df)}")

    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_pickle_file_1> [path_to_pickle_file_2] ...")
        sys.exit(1)
    
    # Loop through all file paths provided as arguments
    for pickle_file_path in sys.argv[1:]:
        inspect_dataframe_pickle(pickle_file_path)