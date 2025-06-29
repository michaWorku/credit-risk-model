from abc import ABC, abstractmethod
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data # Corrected import path
from src.data_processing import preprocess_data # New import for preprocessing


# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform missing values analysis.")
            return
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method should display a heatmap of missing values.
        """
        pass


# Concrete Implementation for Simple Missing Values Analysis
# ----------------------------------------------------------
# This class provides concrete implementations for identifying and visualizing missing values.
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        # Filter to show only columns with missing values
        missing_values_exist = missing_values[missing_values > 0]
        if not missing_values_exist.empty:
            print(missing_values_exist)
        else:
            print("No missing values found in the DataFrame.")


    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        # Use a different cmap for better visibility if data is sparse
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Xente data file (assuming 'data.csv' is the Xente data)
    project_root = Path(__file__).parent.parent.parent
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df_raw = load_data(raw_data_file_path, delimiter=',')

    if not df_raw.empty:
        # Preprocess the data using the dedicated data_processing script
        df = preprocess_data(df_raw)

        print("\n--- Missing Values Analysis Examples (Xente Data) ---")

        # Perform Missing Values Analysis
        missing_values_analyzer = SimpleMissingValuesAnalysis()
        missing_values_analyzer.analyze(df)

        print("\nMissing values analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run missing values analysis examples.")
