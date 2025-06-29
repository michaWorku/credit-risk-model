from pathlib import Path
import pandas as pd

def load_data(file_path: Path, delimiter: str = ',', file_type: str = 'csv') -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame.

    This function supports loading data from CSV and TXT files, handling different
    delimiters. It performs basic checks for file existence and readability.

    Args:
        file_path (Path): The full path to the data file.
        delimiter (str): The delimiter to use for parsing the file (e.g., ',', '|', '\t').
                         Defaults to ','.
        file_type (str): The type of the file ('csv' or 'txt'). Defaults to 'csv'.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, delimiter=delimiter)
        elif file_type == 'txt':
            df = pd.read_csv(file_path, delimiter=delimiter) # pd.read_csv can handle txt with delimiter
        else:
            print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported.")
            return pd.DataFrame()

        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

# Example usage (for independent testing of this script)
if __name__ == "__main__":
    # Define the path to the raw Kaggle data file
    # Adjust this path if your project structure differs or if the file name is different
    project_root = Path(__file__).parent.parent # Navigates to credit-risk-model/
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df = load_data(raw_data_file_path, delimiter=',')

    if not df.empty:
        print("\nDataFrame Head:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
    else:
        print("Failed to load data for example usage.")
