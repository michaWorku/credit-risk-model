from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations
import os # For dummy data creation

# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent.parent)) # Adjust path to project root
from src.data_loader import load_data # Corrected import path
from src.data_processing import preprocess_data # New import for preprocessing

# Abstract Base Class for Outlier Analysis Strategy
# -------------------------------------------------
# This class defines a common interface for outlier detection strategies.
# Subclasses must implement the analyze method.
class OutlierAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform outlier analysis on a specific numerical feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed for outliers.

        Returns:
        None: This method visualizes and/or prints outlier information.
        """
        pass


# Concrete Strategy for IQR-based Outlier Analysis
# -------------------------------------------------
# This strategy detects outliers in a numerical feature using the Interquartile Range (IQR) method.
class IQRBasedOutlierAnalysis(OutlierAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Detects and visualizes outliers in a numerical feature using the IQR method.
        Outliers are defined as values falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed for outliers.

        Returns:
        None: Displays a box plot and prints outlier statistics.
        """
        if feature not in df.columns:
            print(f"Error: Feature '{feature}' not found in DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"Error: Feature '{feature}' is not numerical. Cannot perform outlier analysis.")
            return

        # Drop NA values for IQR calculation to avoid errors
        data = df[feature].dropna()

        if data.empty:
            print(f"No valid data in '{feature}' for outlier analysis after dropping NaNs.")
            return

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        print(f"--- Outlier Analysis for {feature} (IQR Method) ---")
        print(f"Q1 (25th percentile): {Q1:.2f}")
        print(f"Q3 (75th percentile): {Q3:.2f}")
        print(f"IQR (Interquartile Range): {IQR:.2f}")
        print(f"Lower Bound (Q1 - 1.5*IQR): {lower_bound:.2f}")
        print(f"Upper Bound (Q3 + 1.5*IQR): {upper_bound:.2f}")
        print(f"Number of outliers detected: {len(outliers)}")
        if len(outliers) > 0:
            print(f"Percentage of outliers: {(len(outliers) / len(data) * 100):.2f}%")
            print(f"Outlier values (first 10 if many):\\n{outliers.head(10)}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[feature].dropna()) # Boxplot handles NaNs by default, but we ensure data is clean for stats
        plt.title(f'Box Plot of {feature} for Outlier Detection')
        plt.ylabel(feature)
        plt.show()


# Context Class for Outlier Analysis
# ----------------------------------
# This class uses the Strategy pattern to perform outlier analysis.
class OutlierAnalyzer:
    def __init__(self, strategy: OutlierAnalysisStrategy):
        """
        Initializes the OutlierAnalyzer with a specific strategy.

        Parameters:
        strategy (OutlierAnalysisStrategy): The strategy to be used for analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierAnalysisStrategy):
        """
        Sets a new strategy for the analyzer.

        Parameters:
        strategy (OutlierAnalysisStrategy): The new strategy to be used.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the outlier analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature (str): The name of the feature to be analyzed for outliers.
        """
        self._strategy.analyze(df, feature)


# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Xente data file (assuming 'data.csv' is the Xente data)
    project_root = Path(__file__).parent.parent.parent
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df_raw = load_data(raw_data_file_path)

    if not df_raw.empty:
        # Preprocess the data using the dedicated data_processing script
        df = preprocess_data(df_raw)

        print("\n--- Outlier Analysis Examples (Xente Data) ---")

        # Analyze outliers in numerical features
        numerical_cols_for_outliers = ['Amount', 'Value'] # 'TotalClaims' not in Xente data fields
        for col in numerical_cols_for_outliers:
            outlier_analyzer = OutlierAnalyzer(IQRBasedOutlierAnalysis())
            outlier_analyzer.execute_analysis(df, col)

        # Example: Test with a non-existent column
        outlier_analyzer.execute_analysis(df, 'NonExistentColumn')

        # Example: Test with a non-numerical column (should be handled by the strategy)
        outlier_analyzer.execute_analysis(df, 'ProductCategory')

        print("\nOutlier analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run outlier analysis examples.")
