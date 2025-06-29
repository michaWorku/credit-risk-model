import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data

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
        print("Input DataFrame is empty, returning empty DataFrame after preprocessing.")
        return pd.DataFrame()

    df_processed = df.copy()

    # Convert 'Amount' and 'Value' to numeric, coercing errors to NaN
    # These are crucial for numerical analysis and aggregation.
    for col in ['Amount', 'Value']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found for numeric conversion.")

    # Convert 'TransactionStartTime' to datetime objects
    # Essential for temporal analysis.
    if 'TransactionStartTime' in df_processed.columns:
        df_processed['TransactionStartTime'] = pd.to_datetime(df_processed['TransactionStartTime'], errors='coerce')
    else:
        print("Warning: 'TransactionStartTime' column not found for datetime conversion.")

    # Convert 'FraudResult' to integer (0 or 1)
    if 'FraudResult' in df_processed.columns:
        df_processed['FraudResult'] = pd.to_numeric(df_processed['FraudResult'], errors='coerce').fillna(0).astype(int)
    else:
        print("Warning: 'FraudResult' column not found for integer conversion.")

    print("Initial data preprocessing (type conversions) completed.")
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
        print(f"Processed data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")

# Example usage for independent testing
if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw" / "data.csv"
    processed_output_dir = project_root / "data" / "processed"

    # Load raw data
    raw_df = load_data(raw_data_path, delimiter=',')

    if not raw_df.empty:
        # Preprocess data
        processed_df = preprocess_data(raw_df)

        # Display info on processed data
        print("\nProcessed DataFrame Info:")
        processed_df.info()
        print("\nProcessed DataFrame Head:")
        print(processed_df.head())

        # Save processed data (optional for EDA, but good practice for later steps)
        save_processed_data(processed_df, processed_output_dir)
    else:
        print("Raw DataFrame is empty, skipping preprocessing and saving.")
