from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data # Corrected import path
from src.data_processing import preprocess_data # New import for preprocessing

# Abstract Base Class for Temporal Analysis Strategy
# --------------------------------------------------
# This class defines a common interface for temporal analysis strategies.
# Subclasses must implement the analyze method.
class TemporalAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Perform temporal analysis on a dataframe, visualizing trends of specified metrics
        over time.

        Parameters:
        df (pd.DataFrame): The dataframe containing the time-series data.
        time_column (str): The name of the datetime column to use for temporal analysis.
        metrics (list): A list of numerical columns to aggregate and plot as trends.

        Returns:
        None: This method visualizes temporal trends.
        """
        pass


# Concrete Strategy for Monthly Trend Analysis
# ---------------------------------------------
# This strategy aggregates data monthly and plots trends for specified numerical metrics.
class MonthlyTrendAnalysis(TemporalAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Aggregates data monthly and plots trends for specified numerical metrics.
        It also calculates and plots 'TransactionCount' (number of unique transactions)
        and 'AverageTransactionAmount' (mean of 'Amount').

        Parameters:
        df (pd.DataFrame): The dataframe containing the time-series data.
        time_column (str): The name of the datetime column to use for temporal analysis.
        metrics (list): A list of numerical columns to aggregate and plot as trends.
                        Expected to include 'Amount' and potentially 'Value'.

        Returns:
        None: Displays line plots of temporal trends.
        """
        if time_column not in df.columns:
            print(f"Error: Time column '{time_column}' not found in DataFrame.")
            return

        # Ensure time_column is datetime
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
            print(f"Warning: '{time_column}' is not datetime type. Attempting conversion.")
            df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors='coerce')

        df_copy.dropna(subset=[time_column], inplace=True) # Drop rows where date conversion failed

        if df_copy.empty:
            print("DataFrame is empty after processing time column. No temporal analysis can be performed.")
            return

        df_copy['YearMonth'] = df_copy[time_column].dt.to_period('M')

        # Ensure metrics are numerical
        valid_metrics = []
        for metric in metrics:
            if metric in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[metric]):
                valid_metrics.append(metric)
            else:
                print(f"Warning: Metric '{metric}' is not numerical or not found. Skipping.")

        if not valid_metrics:
            print("No valid numerical metrics to analyze.")
            return

        print(f"--- Analyzing Monthly Trends for {', '.join(valid_metrics)} ---")

        # Define aggregations with explicit output column names
        agg_dict = {}
        if 'Amount' in valid_metrics:
            agg_dict['TotalAmount'] = ('Amount', 'sum')
            agg_dict['AverageAmount'] = ('Amount', 'mean')
        if 'Value' in valid_metrics:
            agg_dict['TotalValue'] = ('Value', 'sum')
        if 'FraudResult' in valid_metrics: # Assuming FraudResult is 0/1 and we want sum of fraud incidents
            agg_dict['TotalFraudIncidents'] = ('FraudResult', 'sum')

        if 'TransactionId' in df_copy.columns:
            agg_dict['TransactionCount'] = ('TransactionId', 'count')


        if not agg_dict:
            print("No valid aggregations to perform.")
            return

        # Perform aggregation
        monthly_data = df_copy.groupby('YearMonth').agg(**agg_dict).reset_index()
        monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str) # Convert Period to string for plotting


        # Prepare list of metrics to plot based on what was actually aggregated
        plot_metrics = []
        if 'TotalAmount' in monthly_data.columns:
            plot_metrics.append('TotalAmount')
        if 'AverageAmount' in monthly_data.columns:
            plot_metrics.append('AverageAmount')
        if 'TotalValue' in monthly_data.columns:
            plot_metrics.append('TotalValue')
        if 'TotalFraudIncidents' in monthly_data.columns:
            plot_metrics.append('TotalFraudIncidents')
        if 'TransactionCount' in monthly_data.columns:
            plot_metrics.append('TransactionCount')


        if not plot_metrics:
            print("No metrics to plot after aggregation.")
            return

        fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(14, 5 * len(plot_metrics)), sharex=True)
        if len(plot_metrics) == 1: # Handle case where only one plot is generated
            axes = [axes]

        for idx, metric in enumerate(plot_metrics):
            sns.lineplot(ax=axes[idx], x='YearMonth', y=metric, data=monthly_data, marker='o')
            axes[idx].set_title(f'Monthly {metric} Over Time')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)

        plt.xlabel('Year-Month')
        plt.tight_layout()
        plt.show()


# Context Class for Temporal Analysis
# -----------------------------------
# This class uses the Strategy pattern to perform temporal analysis.
class TemporalAnalyzer:
    def __init__(self, strategy: TemporalAnalysisStrategy):
        """
        Initializes the TemporalAnalyzer with a specific strategy.

        Parameters:
        strategy (TemporalAnalysisStrategy): The strategy to be used for analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: TemporalAnalysisStrategy):
        """
        Sets a new strategy for the analyzer.

        Parameters:
        strategy (TemporalAnalysisStrategy): The new strategy to be used.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Executes the temporal analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        time_column (str): The name of the datetime column.
        metrics (list): A list of numerical columns to aggregate and plot.
        """
        self._strategy.analyze(df, time_column, metrics)


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

        print("\n--- Temporal Analysis Examples (Xente Data) ---")

        # Example 1: Analyze monthly trends for 'Amount', 'Value', and 'FraudResult'
        # 'FraudResult' is included as a metric to see fraud trends over time
        temporal_analyzer = TemporalAnalyzer(MonthlyTrendAnalysis())
        temporal_analyzer.execute_analysis(df, 'TransactionStartTime', ['Amount', 'Value', 'FraudResult'])

        # Example 2: Test with a non-existent time column
        print("\n--- Example 2: Testing with Non-Existent Time Column ---")
        temporal_analyzer.execute_analysis(df, 'NonExistentTimeColumn', ['Amount'])

        # Example 3: Test with a non-numerical metric (should be skipped or warned)
        print("\n--- Example 3: Testing with Non-Numerical Metric ---")
        temporal_analyzer.execute_analysis(df, 'TransactionStartTime', ['ProductCategory'])

        print("\nTemporal analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run temporal analysis examples.")
