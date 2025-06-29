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


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram and KDE plot.
        """
        if feature not in df.columns:
            print(f"Error: Numerical feature '{feature}' not found in DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"Error: Feature '{feature}' is not numerical. Cannot perform numerical univariate analysis.")
            return

        print(f"--- Analyzing Numerical Feature: {feature} ---")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature].dropna(), kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

        print(f"Descriptive Statistics for {feature}:\n{df[feature].describe()}")


# Concrete Strategy for Categorical Features
# --------------------------------------------
# This strategy analyzes categorical features by plotting their count distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the count distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot.
        """
        if feature not in df.columns:
            print(f"Error: Categorical feature '{feature}' not found in DataFrame.")
            return

        print(f"--- Analyzing Categorical Feature: {feature} ---")
        plt.figure(figsize=(12, 7))
        # Use value_counts() to get counts and sort for better visualization
        # Convert to string to handle mixed types or non-string categoricals gracefully
        sns.countplot(y=df[feature].astype(str), order=df[feature].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Count')
        plt.ylabel(feature)
        plt.tight_layout()
        plt.show()

        print(f"Value Counts for {feature}:\n{df[feature].value_counts()}")


# Context Class for Univariate Analysis
# -------------------------------------
# This class uses the Strategy pattern to perform univariate analysis.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the analyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature (str): The name of the feature to be analyzed.
        """
        self._strategy.analyze(df, feature)


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

        print("\n--- Univariate Analysis Examples (Xente Data) ---")

        # Define columns based on Xente Challenge Data Fields
        numerical_cols = ['Amount', 'Value']
        categorical_cols = [
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
            'ChannelId', 'PricingStrategy', 'FraudResult'
        ]
        # Note: Identifiers are typically not analyzed for distribution in this way.

        # Analyzing numerical features
        for col in numerical_cols:
            analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
            analyzer.execute_analysis(df, col)

        # Analyzing categorical features
        for col in categorical_cols:
            analyzer = UnivariateAnalyzer(CategoricalUnivariateAnalysis())
            analyzer.execute_analysis(df, col)

        print("\nUnivariate analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run univariate analysis examples.")
