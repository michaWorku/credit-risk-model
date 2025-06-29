from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import sys
# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent.parent)) # Adjust path to project root
from src.data_loader import load_data


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        if df.empty:
            print("DataFrame is empty. No data types or non-null counts to inspect.")
            return
        print("\nData Types and Non-null Counts:")
        # df.info() already provides this, so we can directly call it
        df.info()


# Concrete Strategy for Summary Statistics Inspection
# ----------------------------------------------------
# This strategy provides descriptive statistics for numerical columns and value counts for categorical ones.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints summary statistics for numerical columns and value counts
        for categorical columns in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the summary statistics to the console.
        """
        if df.empty:
            print("DataFrame is empty. No summary statistics to inspect.")
            return
        print("\nSummary Statistics for Numerical Features:")
        print(df.describe())

        print("\nValue Counts for Categorical Features (Top 10 for each):")
        for column in df.select_dtypes(include=['object', 'category']).columns:
            print(f"--- {column} ---")
            print(df[column].value_counts().head(10))
            print("\n")


# Context Class for Data Inspection
# ---------------------------------
# This class uses the Strategy pattern to perform various data inspections.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific strategy.

        Parameters:
        strategy (DataInspectionStrategy): The initial strategy to be used for data inspection.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Kaggle data file
    project_root = Path(__file__).parent.parent.parent # Navigates to credit-risk-model/
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df = load_data(raw_data_file_path, delimiter=',')

    if not df.empty:
        # Convert 'Amount' and 'Value' to numeric, coercing errors
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        print("\n--- Basic Data Inspection Examples (Xente Data) ---")

        # Initialize the Data Inspector with a specific strategy
        inspector = DataInspector(DataTypesInspectionStrategy())
        inspector.execute_inspection(df)

        # Change strategy to Summary Statistics and execute
        inspector.set_strategy(SummaryStatisticsInspectionStrategy())
        inspector.execute_inspection(df)

        print("\nBasic data inspection examples complete.")
    else:
        print("DataFrame is empty. Cannot run basic data inspection examples.")
