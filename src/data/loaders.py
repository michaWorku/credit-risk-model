from pathlib import Path
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(file_path)

    try:
        if file_type in {"csv", "txt"}:
            return pd.read_csv(file_path, delimiter=delimiter)
        raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.exception("Failed loading data: %s", e)
        raise

# Example usage (for independent testing of this script)
if __name__ == "__main__":
    # Define the path to the raw Kaggle data file
    # Adjust this path if your project structure differs or if the file name is different
    project_root = Path(__file__).parent.parent.parent # Navigates to credit-risk-model/
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df = load_data(raw_data_file_path, delimiter=',')

    if not df.empty:
        logger.info("Data loaded successfully.")
        logger.info("\nDataFrame Head:")
        logger.info(df.head())
        logger.info("\nDataFrame Info:")
        df.info()
    else:
        logger.error("Failed to load data for example usage.")
