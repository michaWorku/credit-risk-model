import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
import numpy as np
import warnings
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data

# Suppress all warnings for cleaner output, especially from KMeans
warnings.filterwarnings('ignore')

class RFMEngineer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to engineer a proxy target variable ('is_high_risk')
    based on RFM analysis and K-Means clustering.

    RFM (Recency, Frequency, Monetary) metrics are calculated for each customer,
    customers are clustered, and the least engaged cluster is labeled as high-risk.
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42,
                 customer_id_col: str = 'CustomerId',
                 transaction_id_col: str = 'TransactionId',
                 transaction_time_col: str = 'TransactionStartTime',
                 amount_col: str = 'Amount'):
        """
        Initializes the RFMEngineer with clustering parameters and column names.

        Args:
            n_clusters (int): The number of clusters for K-Means. Defaults to 3.
            random_state (int): The random state for K-Means reproducibility. Defaults to 42.
            customer_id_col (str): Name of the customer ID column.
            transaction_id_col (str): Name of the transaction ID column.
            transaction_time_col (str): Name of the transaction timestamp column.
            amount_col (str): Name of the transaction amount column (expected to be positive after prior steps).
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.customer_id_col = customer_id_col
        self.transaction_id_col = transaction_id_col
        self.transaction_time_col = transaction_time_col
        self.amount_col = amount_col

        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10) # n_init for robust centroid init
        
        self.high_risk_cluster_label = None
        self.snapshot_date = None
        self.rfm_columns = ['Recency', 'Frequency', 'Monetary']
        self.rfm_scaled_columns = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
        self.fitted_ = False

    def _calculate_rfm(self, df: pd.DataFrame, current_snapshot_date: datetime.datetime) -> pd.DataFrame:
        """
        Internal method to calculate Recency, Frequency, and Monetary (RFM) metrics.
        """
        required_cols = [self.customer_id_col, self.transaction_time_col, self.amount_col, self.transaction_id_col]
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing one or more required columns for RFM calculation: {required_cols}. Returning empty RFM DataFrame.")
            return pd.DataFrame(columns=[self.customer_id_col] + self.rfm_columns)
        
        # Ensure transaction_time_col is datetime and amount_col is numeric
        df[self.transaction_time_col] = pd.to_datetime(df[self.transaction_time_col], errors='coerce', utc=True)
        df[self.amount_col] = pd.to_numeric(df[self.amount_col], errors='coerce')
        df[self.transaction_id_col] = df[self.transaction_id_col].astype(str) # Ensure TransactionId is string for nunique

        # Drop rows with NaNs in critical columns for RFM calculation
        df_cleaned = df.dropna(subset=[self.customer_id_col, self.transaction_time_col, self.amount_col, self.transaction_id_col]).copy()
        
        if df_cleaned.empty:
            print("Warning: DataFrame is empty after cleaning for RFM calculation. Returning empty RFM DataFrame.")
            return pd.DataFrame(columns=[self.customer_id_col] + self.rfm_columns)

        rfm_df = df_cleaned.groupby(self.customer_id_col).agg(
            Recency=(self.transaction_time_col, lambda date: (current_snapshot_date - date.max()).days),
            Frequency=(self.transaction_id_col, 'nunique'), # Use nunique for distinct transactions
            Monetary=(self.amount_col, 'sum') # Total amount spent per customer
        ).reset_index()

        # Handle potential zero or negative Monetary values (e.g., if all transactions were refunds)
        # For RFM, Monetary is typically positive spending. Set to a small positive if <= 0.
        rfm_df['Monetary'] = rfm_df['Monetary'].apply(lambda x: x if x > 0 else 1e-6)
        
        # Recency should not be negative, if it is, set to 0
        rfm_df['Recency'] = rfm_df['Recency'].apply(lambda x: x if x >= 0 else 0)
        
        # Add 1 to Recency and Frequency for log transformation to handle 0 values
        rfm_df['Recency'] = rfm_df['Recency'] + 1
        rfm_df['Frequency'] = rfm_df['Frequency'] + 1

        return rfm_df

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the RFM engineer by calculating RFM metrics, scaling them,
        and clustering customers to identify the high-risk segment.

        Args:
            X (pd.DataFrame): The input DataFrame containing transaction data.
            y: Ignored.

        Returns:
            self: Fitted transformer.
        """
        print("RFMEngineer.fit called.")
        if X.empty:
            print("Warning: Input DataFrame is empty for RFMEngineer.fit. Skipping fit.")
            self.fitted_ = False
            return self

        # Determine snapshot date from the training data's latest transaction
        # This snapshot date will be used consistently for both fit and transform
        X_temp_for_date = X.copy()
        # Ensure TransactionStartTime is datetime before finding max
        if self.transaction_time_col in X_temp_for_date.columns:
            X_temp_for_date[self.transaction_time_col] = pd.to_datetime(X_temp_for_date[self.transaction_time_col], errors='coerce', utc=True)
            X_temp_for_date.dropna(subset=[self.transaction_time_col], inplace=True)
        
        if X_temp_for_date.empty:
            print("Warning: No valid transaction times to determine snapshot date. Skipping fit.")
            self.fitted_ = False
            return self
        self.snapshot_date = X_temp_for_date[self.transaction_time_col].max() + datetime.timedelta(days=1)


        # 1. Calculate RFM Metrics
        rfm_data = self._calculate_rfm(X.copy(), self.snapshot_date)
        if rfm_data.empty:
            print("RFM calculation resulted in empty DataFrame during fit. Cannot proceed with clustering.")
            self.fitted_ = False
            return self

        # 2. Pre-process RFM Features (log transform and scale)
        # Apply log transformation to handle skewness
        rfm_log = rfm_data[self.rfm_columns].apply(np.log) # Use np.log directly after adding 1
        
        # Fit StandardScaler
        self.scaler.fit(rfm_log)
        
        # 3. Cluster Customers
        rfm_scaled = self.scaler.transform(rfm_log)
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=self.rfm_scaled_columns, index=rfm_data.index)
        
        self.kmeans.fit(rfm_scaled_df[self.rfm_scaled_columns])
        self.cluster_centroids = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.rfm_scaled_columns)

        # 4. Define and Assign "High-Risk" Label
        # Analyze cluster centroids to find the "least engaged" cluster
        # The cluster with the highest scaled Recency, lowest scaled Frequency,
        # and lowest scaled Monetary value is considered least engaged/high-risk.
        # We sum the ranks to find the "worst" cluster.
        self.cluster_centroids['Recency_rank'] = self.cluster_centroids['Recency_scaled'].rank(ascending=False)
        self.cluster_centroids['Frequency_rank'] = self.cluster_centroids['Frequency_scaled'].rank(ascending=True)
        self.cluster_centroids['Monetary_rank'] = self.cluster_centroids['Monetary_scaled'].rank(ascending=True)

        self.cluster_centroids['Engagement_Score'] = (
            self.cluster_centroids['Recency_rank'] +
            self.cluster_centroids['Frequency_rank'] +
            self.cluster_centroids['Monetary_rank']
        )
        
        # The cluster with the highest Engagement_Score is the least engaged (high-risk)
        self.high_risk_cluster_label = self.cluster_centroids['Engagement_Score'].idxmax()
        
        print(f"RFMEngineer fitted. High-risk cluster identified: {self.high_risk_cluster_label}")
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by calculating RFM metrics, assigning clusters,
        and adding the 'is_high_risk' target variable.

        Args:
            X (pd.DataFrame): The input DataFrame containing transaction data.

        Returns:
            pd.DataFrame: The DataFrame with added 'Recency', 'Frequency', 'Monetary',
                          and 'is_high_risk' columns.
                          Returns original DataFrame if not fitted or empty.
        """
        print("RFMEngineer.transform called.")
        if not self.fitted_:
            print("Warning: RFMEngineer not fitted. Returning original DataFrame without RFM or 'is_high_risk'.")
            # Ensure 'is_high_risk' is present and defaulted to 0 if not fitted
            if 'is_high_risk' not in X.columns:
                X['is_high_risk'] = 0
            # Ensure RFM columns are present and defaulted to 0 if not fitted
            for col in self.rfm_columns:
                if col not in X.columns:
                    X[col] = 0.0 # Default to 0.0 instead of NaN for numerical columns
            return X.copy()
        
        if X.empty:
            print("Warning: Input DataFrame is empty for RFMEngineer.transform. Returning empty DataFrame.")
            # Ensure expected columns are present in the empty DataFrame
            empty_df = X.copy()
            empty_df['is_high_risk'] = []
            for col in self.rfm_columns:
                empty_df[col] = []
            return empty_df


        X_copy = X.copy()

        # 1. Calculate RFM Metrics for the current batch using the fitted snapshot_date
        rfm_data = self._calculate_rfm(X_copy, self.snapshot_date)
        
        if rfm_data.empty:
            print("RFM calculation resulted in empty DataFrame during transform. Adding 'is_high_risk' as 0 and returning.")
            if 'is_high_risk' not in X_copy.columns:
                X_copy['is_high_risk'] = 0
            for col in self.rfm_columns:
                if col not in X_copy.columns:
                    X_copy[col] = 0.0 # Default to 0.0 for numerical columns if RFM calculation fails
            return X_copy

        # 2. Pre-process RFM Features (log transform and scale) using fitted scaler
        rfm_log = rfm_data[self.rfm_columns].apply(np.log)
        
        # Transform using the fitted scaler
        rfm_scaled = self.scaler.transform(rfm_log)
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=self.rfm_scaled_columns, index=rfm_data.index)
        rfm_scaled_df[self.customer_id_col] = rfm_data[self.customer_id_col]

        # 3. Predict Clusters using fitted KMeans
        rfm_scaled_df['Cluster'] = self.kmeans.predict(rfm_scaled_df[self.rfm_scaled_columns])

        # 4. Assign 'is_high_risk' label using the high_risk_cluster_label determined during fit
        rfm_scaled_df['is_high_risk'] = (rfm_scaled_df['Cluster'] == self.high_risk_cluster_label).astype(int)

        # Merge 'is_high_risk' and original RFM features back to the original DataFrame X
        # Create a combined DataFrame of CustomerId, RFM, and is_high_risk for merging
        customer_features_to_merge = rfm_data[[self.customer_id_col] + self.rfm_columns].copy()
        customer_features_to_merge['is_high_risk'] = rfm_scaled_df['is_high_risk']

        # Perform a left merge to keep all original transactions and add the RFM and is_high_risk flags
        df_transformed = pd.merge(X_copy, customer_features_to_merge, on=self.customer_id_col, how='left')

        # Fill any NaNs that might result from customers in X_copy not being present in rfm_data
        # (e.g., if they had no valid transactions after cleaning)
        if 'is_high_risk' in df_transformed.columns:
            df_transformed['is_high_risk'] = df_transformed['is_high_risk'].fillna(0).astype(int)
        else: # Should not happen if merge is correct, but for safety
            df_transformed['is_high_risk'] = 0

        for col in self.rfm_columns:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(0) # Fill RFM NaNs with 0 for non-transacting customers
            else:
                df_transformed[col] = 0 # Add if missing entirely (shouldn't happen with left merge)


        return df_transformed


class ProxyTargetEngineer(BaseEstimator, TransformerMixin):
    """
    A strategy to engineer a proxy target variable, 'is_high_risk',
    based on the 'FraudResult' column. This is a direct mapping.
    """
    def __init__(self, fraud_result_col='FraudResult'):
        self.fraud_result_col = fraud_result_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_copy = X.copy()
        if self.fraud_result_col not in df_copy.columns:
            print(f"Warning: '{self.fraud_result_col}' column not found. Cannot create proxy target 'is_high_risk'.")
            df_copy['is_high_risk'] = 0 # Default to low risk if fraud result is unknown
            return df_copy

        df_copy['is_high_risk'] = df_copy[self.fraud_result_col].apply(lambda x: 1 if x == 1 else 0)
        return df_copy


# Main execution for demonstration
if __name__ == "__main__":
    # Define the path to the raw Xente data file
    project_root = Path(__file__).parent.parent
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df_raw = load_data(raw_data_file_path, delimiter=',')

    if df_raw is not None and not df_raw.empty:
        print("Successfully loaded data for target engineering.")

        # --- Prepare data for RFM and Proxy Target Engineering ---
        # Ensure 'Amount' is numeric and positive
        df_raw['Amount'] = pd.to_numeric(df_raw['Amount'], errors='coerce').abs()
        # Ensure 'TransactionStartTime' is datetime
        df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'], errors='coerce', utc=True)
        # Ensure 'TransactionId' is string for nunique in RFMEngineer
        df_raw['TransactionId'] = df_raw['TransactionId'].astype(str)

        # Example 1: Using RFMEngineer
        print("\n--- Applying RFM Engineering and Risk Labeling ---")
        rfm_engineer = RFMEngineer(
            customer_id_col='CustomerId',
            transaction_id_col='TransactionId',
            transaction_time_col='TransactionStartTime',
            amount_col='Amount',
            n_clusters=3 # You can tune this
        )
        # Use fit_transform directly for demonstration
        df_with_risk_label = rfm_engineer.fit_transform(df_raw.copy())

        print("\nDataFrame head after RFM and Risk Labeling:")
        # Select relevant columns for clear output
        display_cols_rfm = ['CustomerId', 'TransactionStartTime', 'Amount', 'Recency', 'Frequency', 'Monetary', 'is_high_risk']
        # Filter for existing columns to avoid KeyError if some are not generated due to data issues
        existing_display_cols_rfm = [col for col in display_cols_rfm if col in df_with_risk_label.columns]
        print(df_with_risk_label[existing_display_cols_rfm].head())
        
        if 'Recency' in df_with_risk_label.columns:
            print(f"\nRecency head:\n{df_with_risk_label['Recency'].head()}")
        if 'Frequency' in df_with_risk_label.columns:
            print(f"Frequency head:\n{df_with_risk_label['Frequency'].head()}")
        if 'Monetary' in df_with_risk_label.columns:
            print(f"Monetary head:\n{df_with_risk_label['Monetary'].head()}")
        
        if 'is_high_risk' in df_with_risk_label.columns:
            print(f"\n'is_high_risk' value counts after RFM:\n{df_with_risk_label['is_high_risk'].value_counts()}")
            print(f"\n'is_high_risk' dtype: {df_with_risk_label['is_high_risk'].dtype}")
        print("\nInfo on DataFrame with 'is_high_risk' (checking dtype and non-null count):")
        df_with_risk_label.info()


        # Example 2: Using ProxyTargetEngineer (if FraudResult is available)
        if 'FraudResult' in df_raw.columns:
            print("\n--- Applying Proxy Target Engineering (FraudResult based) ---")
            proxy_engineer = ProxyTargetEngineer(fraud_result_col='FraudResult')
            df_with_proxy_target = proxy_engineer.fit_transform(df_raw.copy())
            print("\nDataFrame head after Proxy Target Labeling:")
            # Select relevant columns for clear output
            display_cols_proxy = ['TransactionId', 'FraudResult', 'is_high_risk']
            existing_display_cols_proxy = [col for col in display_cols_proxy if col in df_with_proxy_target.columns]
            print(df_with_proxy_target[existing_display_cols_proxy].head())
            
            if 'is_high_risk' in df_with_proxy_target.columns:
                print(f"\n'is_high_risk' value counts after Proxy Target:\n{df_with_proxy_target['is_high_risk'].value_counts()}")
        else:
            print("\nSkipping Proxy Target Engineering: 'FraudResult' column not found in raw data.")

    else:
        print("DataFrame is empty or could not be loaded. Cannot run target engineering examples.")

