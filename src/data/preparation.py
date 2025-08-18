import pandas as pd
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loaders import load_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial preprocessing steps on the transaction data.
    This includes converting columns to appropriate data types.

    Args:
        df (pd.DataFrame): The raw transaction DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with initial preprocessing applied.
    """
    if df.empty:
        logger.error("Input DataFrame is empty, returning empty DataFrame after preprocessing.")
        return df

    df_processed = df.copy()

    # Convert 'Amount' and 'Value' to numeric, coercing errors to NaN
    # These are crucial for numerical analysis and aggregation.
    for col in ['Amount', 'Value']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            logger.warning(f"Warning: Column '{col}' not found for numeric conversion.")

    # Convert 'TransactionStartTime' to datetime objects
    # Essential for temporal analysis.
    if 'TransactionStartTime' in df_processed.columns:
        df_processed['TransactionStartTime'] = pd.to_datetime(df_processed['TransactionStartTime'], errors='coerce')
    else:
        logger.warning("Warning: 'TransactionStartTime' column not found for datetime conversion.")

    # Convert 'FraudResult' to integer (0 or 1)
    if 'FraudResult' in df_processed.columns:
        df_processed['FraudResult'] = pd.to_numeric(df_processed['FraudResult'], errors='coerce').fillna(0).astype(int)
    else:
        logger.warning("Warning: 'FraudResult' column not found for integer conversion.")

    logger.info("Initial data preprocessing (type conversions) completed.")
    return df_processed

def save_processed_data(df: pd.DataFrame, output_dir: Path, filename: str = "processed_data.csv"):
    """
    Saves the processed DataFrame to a specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_dir (Path): The directory where the file should be saved.
        filename (str): The name of the output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    except Exception as e:
        logger.exception(f"Error saving processed data to {output_path}: {e}")

# Example usage for independent testing
if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / "data" / "raw" / "data.csv"
    processed_output_dir = project_root / "data" / "processed"

    # Load raw data
    raw_df = load_data(raw_data_path, delimiter=',')

    if not raw_df.empty:
        # Preprocess data
        processed_df = preprocess_data(raw_df)

        # Display info on processed data
        logger.info("\nProcessed DataFrame Info:")
        processed_df.info()
        logger.info("\nProcessed DataFrame Head:")
        logger.info(processed_df.head())

        # Save processed data (optional for EDA, but good practice for later steps)
        save_processed_data(processed_df, processed_output_dir)
    else:
        logger.error("Raw DataFrame is empty, skipping preprocessing and saving.")
