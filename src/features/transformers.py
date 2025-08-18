import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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

# Conditional import for scorecardpy and definition of WOETransformer
try:
    import scorecardpy as sc
    logger.info("scorecardpy imported successfully. WOETransformer will use scorecardpy.")

    class WOETransformer(BaseEstimator, TransformerMixin):
        """
        Custom transformer to apply Weight of Evidence (WoE) transformation using scorecardpy.
        This transformer fits the binning process on the training data (when fit is called)
        and then applies the transformation.
        """
        def __init__(self, categorical_cols, target_col='FraudResult'):
            self.categorical_cols = categorical_cols
            self.target_col = target_col
            self.bins = None # To store the binning object from scorecardpy
            self.fitted_cols = [] # Store the actual columns that were fitted for transformation

        def fit(self, X, y=None):
            logger.info("WOETransformer.fit called (scorecardpy).")
            # Handle empty DataFrame gracefully
            if X.empty:
                logger.warning("WOETransformer.fit: Input DataFrame is empty. Skipping fit.")
                self.bins = None
                self.fitted_cols = []
                return self

            if y is None:
                raise ValueError("Target variable 'y' must be provided for WOETransformer.fit().")
            
            # Combine X and y for scorecardpy.woebin
            df_for_woebin = X.copy()
            df_for_woebin[self.target_col] = y.values # Ensure y is aligned

            # Filter for categorical columns that actually exist in the dataframe
            existing_categorical_cols = [col for col in self.categorical_cols if col in df_for_woebin.columns]
            logger.info(f"WOETransformer.fit: Categorical columns to fit: {existing_categorical_cols}")

            if not existing_categorical_cols:
                logger.warning("WOETransformer.fit: No valid categorical columns for WoE transformation. Initializing empty bins.")
                self.bins = None
                self.fitted_cols = []
                return self

            # Generate bins using scorecardpy.woebin
            self.bins = sc.woebin(df_for_woebin, y=self.target_col, x=existing_categorical_cols, print_info=False)
            logger.info("WOETransformer.fit: woebin completed.")

            # Store the names of the columns that were actually fitted
            self.fitted_cols = existing_categorical_cols
            return self

        def transform(self, X):
            logger.info("WOETransformer.transform called (scorecardpy).")
            # Handle empty DataFrame gracefully
            if X.empty:
                logger.warning("WOETransformer.transform: Input DataFrame is empty. Returning empty DataFrame.")
                return X.copy()

            if self.bins is None or not self.fitted_cols:
                logger.warning("WOETransformer.transform: Not fitted or no valid categorical columns were fitted. Returning original DataFrame.")
                return X.copy()

            X_copy = X.copy()
            logger.info(f"WOETransformer.transform: Input X_copy columns: {X_copy.columns.tolist()}")

            # Apply WoE replacement using scorecardpy.woe_replace
            X_transformed_woe = sc.woebin_ply(X_copy, self.bins)
            logger.info(f"WOETransformer.transform: After woe_replace, columns: {X_transformed_woe.columns.tolist()}")

            # Rename columns to add '_woe' suffix for consistency with test expectations
            rename_map = {col: f'{col}_woe' for col in self.fitted_cols if col in X_transformed_woe.columns}
            X_transformed_woe.rename(columns=rename_map, inplace=True)
            logger.info(f"WOETransformer.transform: After renaming, columns: {X_transformed_woe.columns.tolist()}")

            return X_transformed_woe

except ImportError as e:
    logger.error(f"Error importing scorecardpy: {e}. WOETransformer will be a placeholder and will not perform transformations.")
    class WOETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, categorical_cols, target_col='FraudResult', woe_maps=None):
            self.categorical_cols = categorical_cols
            self.target_col = target_col
            self.woe_maps = woe_maps
        def fit(self, X, y=None):
            logger.info("WOETransformer (placeholder).fit called.")
            if X.empty:
                return self
            return self
        def transform(self, X):
            logger.info("WOETransformer (placeholder).transform called.")
            if X.empty:
                return X.copy()
            return X.copy()


class HandleNegativeAmount(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle negative 'Amount' values by converting them to positive
    and creating an 'IsRefund' indicator.
    """
    def __init__(self, amount_col='Amount'):
        self.amount_col = amount_col
        self.is_refund_col = 'IsRefund'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.amount_col in X_copy.columns:
            # Ensure Amount is numeric before comparison
            X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')
            # Create 'IsRefund' flag: 1 if Amount is negative, 0 otherwise
            X_copy[self.is_refund_col] = (X_copy[self.amount_col] < 0).astype(int)
            # Convert negative amounts to their absolute value
            X_copy[self.amount_col] = X_copy[self.amount_col].abs()
        else:
            logger.warning(f"Warning: '{self.amount_col}' column not found for HandleNegativeAmount. Skipping transformation and adding default 'IsRefund'.")
            # Always add 'IsRefund' column to maintain consistent schema
            X_copy[self.is_refund_col] = 0
        return X_copy


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create aggregate features per customer.
    Requires 'CustomerId', 'Amount', 'TransactionId' columns.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', transaction_id_col='TransactionId'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.transaction_id_col = transaction_id_col
        self.agg_features_names = [
            f'Total_{self.amount_col}',
            f'Avg_{self.amount_col}',
            f'Count_{self.transaction_id_col}',
            f'Std_{self.amount_col}'
        ]

    def fit(self, X, y=None):
        # No fitting required for this transformation
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ensure all aggregate columns exist, initialized to NaN if not present
        for feat_name in self.agg_features_names:
            if feat_name not in X_copy.columns:
                X_copy[feat_name] = np.nan

        required_cols = [self.customer_id_col, self.amount_col, self.transaction_id_col]
        
        # Check if all critical columns are present for actual aggregation
        if not all(col in X_copy.columns for col in required_cols):
            logger.warning(f"Warning: Missing one or more critical columns for aggregation: {required_cols}. Aggregate features will be NaN.")
            # If critical columns are missing, aggregate features are already NaN-filled above
            return X_copy 

        # Ensure 'Amount' is numeric for aggregation
        X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')

        # Perform aggregation only if 'Amount' is not entirely NaN after conversion and DataFrame is not empty
        if X_copy.empty or X_copy[self.amount_col].isnull().all():
            logger.warning(f"Warning: DataFrame is empty or '{self.amount_col}' column is entirely NaN. Skipping aggregation calculations.")
            return X_copy # Return with NaN-filled aggregate columns (already done above)
        
        customer_agg = X_copy.groupby(self.customer_id_col).agg(
            total_amount=(self.amount_col, 'sum'),
            avg_amount=(self.amount_col, 'mean'),
            transaction_count=(self.transaction_id_col, 'count'),
            std_amount=(self.amount_col, lambda x: x.std(skipna=True)) # Ensure std handles NaNs gracefully
        ).reset_index()

        # Rename columns for clarity
        customer_agg.columns = [self.customer_id_col] + self.agg_features_names

        # Merge aggregated features back to the original DataFrame
        # Drop the placeholder NaN columns before merging the calculated ones
        X_processed = pd.merge(X_copy.drop(columns=self.agg_features_names, errors='ignore'),
                               customer_agg, on=self.customer_id_col, how='left')

        # Fill NaN standard deviations (e.g., for customers with only one transaction) with 0
        X_processed[f'Std_{self.amount_col}'] = X_processed[f'Std_{self.amount_col}'].fillna(0)

        return X_processed

# Custom Transformer for Time-Based Feature Extraction
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract time-based features from 'TransactionStartTime'.
    """
    def __init__(self, time_column='TransactionStartTime'):
        self.time_column = time_column
        self.extracted_time_features = [
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
            'TransactionHour_sin', 'TransactionHour_cos',
            'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.time_column in X_copy.columns:
            # Ensure the column is datetime
            X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column], errors='coerce', utc=True)
            # Drop rows where datetime conversion failed for critical time features
            # If a row has NaN for TransactionStartTime, it cannot have time features
            # This dropna can change the number of rows, which is acceptable for this transformer
            X_copy.dropna(subset=[self.time_column], inplace=True)

            if X_copy.empty:
                logger.warning(f"Warning: DataFrame became empty after dropping NaNs in '{self.time_column}'. Returning empty DataFrame.")
                # Add placeholder columns to an empty DataFrame to maintain schema consistency
                for feat in self.extracted_time_features:
                    X_copy[feat] = np.nan
                return X_copy

            X_copy['TransactionHour'] = X_copy[self.time_column].dt.hour
            X_copy['TransactionDay'] = X_copy[self.time_column].dt.day
            X_copy['TransactionMonth'] = X_copy[self.time_column].dt.month
            X_copy['TransactionYear'] = X_copy[self.time_column].dt.year
            X_copy['TransactionDayOfWeek'] = X_copy[self.time_column].dt.dayofweek # Monday=0, Sunday=6
            # .dt.isocalendar().week returns a Series with UInt32Dtype, convert to int
            X_copy['TransactionWeekOfYear'] = X_copy[self.time_column].dt.isocalendar().week.astype(int)
            X_copy['TransactionQuarter'] = X_copy[self.time_column].dt.quarter

            # Cyclical features for hour and day of week
            X_copy['TransactionHour_sin'] = np.sin(2 * np.pi * X_copy['TransactionHour'] / 24)
            X_copy['TransactionHour_cos'] = np.cos(2 * np.pi * X_copy['TransactionHour'] / 24)
            X_copy['TransactionDayOfWeek_sin'] = np.sin(2 * np.pi * X_copy['TransactionDayOfWeek'] / 7)
            X_copy['TransactionDayOfWeek_cos'] = np.cos(2 * np.pi * X_copy['TransactionDayOfWeek'] / 7)

            # Drop original datetime column after extraction
            X_copy = X_copy.drop(columns=[self.time_column])
        else:
            logger.warning(f"Warning: Time column '{self.time_column}' not found. Skipping time feature extraction and adding placeholder columns.")
            # Add placeholder columns if time_column is missing
            for feat in self.extracted_time_features:
                X_copy[feat] = np.nan # Or 0, depending on desired default behavior

        return X_copy
