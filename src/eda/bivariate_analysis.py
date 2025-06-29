from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations like np.finfo
import os # For dummy data creation

# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_loader import load_data
from src.data_preparation import preprocess_data # New import for preprocessing



# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature.
        feature2 (str): The name of the second numerical feature.

        Returns:
        None: Displays a scatter plot.
        """
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature1]) or not pd.api.types.is_numeric_dtype(df[feature2]):
            print(f"Error: Both features ('{feature1}', '{feature2}') must be numerical for this analysis.")
            return

        print(f"--- Analyzing Numerical vs Numerical: {feature1} vs {feature2} ---")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature1], y=df[feature2], alpha=0.6)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

        print(f"Correlation between {feature1} and {feature2}: {df[[feature1, feature2]].corr().iloc[0, 1]:.2f}")


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical and a numerical feature.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature
        using a box plot or violin plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature.
        feature2 (str): The name of the numerical feature.

        Returns:
        None: Displays a box plot or violin plot.
        """
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return
        if not pd.api.types.is_categorical_dtype(df[feature1].dtype) and not pd.api.types.is_object_dtype(df[feature1].dtype):
            print(f"Error: Feature '{feature1}' must be categorical for this analysis.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature2]):
            print(f"Error: Feature '{feature2}' must be numerical for this analysis.")
            return

        print(f"--- Analyzing Categorical vs Numerical: {feature1} vs {feature2} ---")
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=df[feature1], y=df[feature2])
        plt.title(f'Box Plot of {feature2} by {feature1}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
        plt.tight_layout()
        plt.show()

        # Print group-wise statistics
        print(f"Mean of {feature2} grouped by {feature1}:\n{df.groupby(feature1)[feature2].mean()}")


# Concrete Strategy for Categorical vs Categorical Analysis
# ----------------------------------------------------------
# This strategy analyzes the relationship between two categorical features using a stacked bar plot or heatmap.
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two categorical features using a heatmap of counts.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first categorical feature.
        feature2 (str): The name of the second categorical feature.

        Returns:
        None: Displays a heatmap of counts.
        """
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in DataFrame.")
            return
        if (not pd.api.types.is_categorical_dtype(df[feature1].dtype) and not pd.api.types.is_object_dtype(df[feature1].dtype)) or \
           (not pd.api.types.is_categorical_dtype(df[feature2].dtype) and not pd.api.types.is_object_dtype(df[feature2].dtype)):
            print(f"Error: Both features ('{feature1}', '{feature2}') must be categorical for this analysis.")
            return

        print(f"--- Analyzing Categorical vs Categorical: {feature1} vs {feature2} ---")
        # Create a cross-tabulation (contingency table)
        crosstab = pd.crosstab(df[feature1], df[feature2])

        if crosstab.empty:
            print(f"No data to display for {feature1} vs {feature2} crosstab.")
            return

        plt.figure(figsize=(12, 8))
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
        plt.title(f'Cross-Tabulation of {feature1} vs {feature2}')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.tight_layout()
        plt.show()

        print(f"Cross-tabulation for {feature1} vs {feature2}:\n{crosstab}")


# Context Class for Bivariate Analysis
# ------------------------------------
# This class uses the Strategy pattern to perform bivariate analysis.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the analyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        feature1 (str): The name of the first feature.
        feature2 (str): The name of the second feature.
        """
        self._strategy.analyze(df, feature1, feature2)


# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Kaggle data file
    project_root = Path(__file__).parent.parent.parent # Navigates to credit-risk-model/
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df_raw = load_data(raw_data_file_path, delimiter=',')

    if not df_raw.empty:
        # Preprocess the data using the dedicated data_processing script
        df = preprocess_data(df_raw)

        print("\n--- Bivariate Analysis Examples (Xente Data) ---")

        # Example 1: Numerical vs Numerical (Amount vs Value)
        bivariate_analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
        bivariate_analyzer.execute_analysis(df, 'Amount', 'Value')

        # Example 2: Categorical vs Numerical (ProductCategory vs Amount)
        bivariate_analyzer.set_strategy(CategoricalVsNumericalAnalysis())
        bivariate_analyzer.execute_analysis(df, 'ProductCategory', 'Amount')

        # Example 3: Categorical vs Numerical (ChannelId vs Amount)
        bivariate_analyzer.execute_analysis(df, 'ChannelId', 'Amount')

        # Example 4: Categorical vs Categorical (ProductCategory vs ChannelId)
        bivariate_analyzer.set_strategy(CategoricalVsCategoricalAnalysis())
        bivariate_analyzer.execute_analysis(df, 'ProductCategory', 'ChannelId')

        # Example 5: Categorical vs Categorical (PricingStrategy vs FraudResult)
        bivariate_analyzer.execute_analysis(df, 'PricingStrategy', 'FraudResult')

        # Example 6: Test with non-existent columns
        print("\n--- Example 6: Testing with Non-Existent Columns ---")
        bivariate_analyzer.execute_analysis(df, 'NonExistentFeature1', 'Amount')

        # Example 7: Test with incorrect data types (e.g., passing numerical to categorical strategy)
        print("\n--- Example 7: Testing with Incorrect Data Types ---")
        bivariate_analyzer.set_strategy(CategoricalVsNumericalAnalysis()) # Expects categorical, numerical
        bivariate_analyzer.execute_analysis(df, 'Amount', 'ProductCategory') # Amount is numerical, ProductCategory is categorical

        print("\nBivariate analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run bivariate analysis examples.")
