from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data # Corrected import path
from src.data_processing import preprocess_data # New import for preprocessing


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, features: list = None):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform multivariate analysis.")
            return

        print("\n--- Performing Multivariate Analysis ---")
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df, features) # Pass features to pairplot

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates a correlation heatmap for numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.

        Returns:
        None: Displays a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame, features: list = None):
        """
        Generates a pair plot for selected numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: Displays a pair plot.
        """
        pass


# Concrete Implementation for Simple Multivariate Analysis
# --------------------------------------------------------
# This class provides concrete implementations for generating correlation heatmaps and pair plots.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates a correlation heatmap for numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.

        Returns:
        None: Displays a heatmap.
        """
        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=np.number)
        if numerical_df.empty:
            print("No numerical columns found for correlation heatmap.")
            return

        print("\nGenerating Correlation Heatmap...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame, features: list = None):
        """
        Generates a pair plot for selected numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: Displays a pair plot.
        """
        if features:
            # Filter features to ensure they are numerical and exist in the DataFrame
            numerical_features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not numerical_features:
                print(f"No valid numerical features found among {features} for pair plot.")
                return
            plot_df = df[numerical_features]
        else:
            plot_df = df.select_dtypes(include=np.number) # Use all numerical columns
            if plot_df.empty:
                print("No numerical columns found for pair plot.")
                return

        print("\nGenerating Pair Plot...")
        # Pairplot can be very slow for many features. Limit to a reasonable number.
        if plot_df.shape[1] > 10:
            print("Warning: Too many numerical features for a practical pair plot. Consider selecting a subset.")
            # For demonstration, we'll proceed, but in real scenarios, subsetting is advised.

        # Drop NaNs for pairplot to avoid errors, only from the columns being plotted
        sns.pairplot(plot_df.dropna())
        plt.suptitle('Pair Plot of Selected Numerical Features', y=1.02) # Adjust title position
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

        print("\n--- Multivariate Analysis Examples (Xente Data) ---")

        # Example 1: Correlation Heatmap (using all relevant numerical columns)
        multivariate_analyzer = SimpleMultivariateAnalysis()
        multivariate_analyzer.analyze(df)

        # Example 2: Pair Plot of selected important numerical features
        # Based on Xente data fields, 'Amount' and 'Value' are the primary numerical ones.
        selected_features_for_pairplot = ['Amount', 'Value']

        multivariate_analyzer.analyze(df, features=selected_features_for_pairplot)

        print("\nMultivariate analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run multivariate analysis examples.")
