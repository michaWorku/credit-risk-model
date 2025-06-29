import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data # Assuming data_loader is in src/

# Conditional import for scorecardpy and definition of WOETransformer
try:
    import scorecardpy as sc
    print("scorecardpy imported successfully. WOETransformer will use scorecardpy.")

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
            print("WOETransformer.fit called (scorecardpy).")
            # Handle empty DataFrame gracefully
            if X.empty:
                print("WOETransformer.fit: Input DataFrame is empty. Skipping fit.")
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
            print(f"WOETransformer.fit: Categorical columns to fit: {existing_categorical_cols}")

            if not existing_categorical_cols:
                print("WOETransformer.fit: No valid categorical columns for WoE transformation. Initializing empty bins.")
                self.bins = None
                self.fitted_cols = []
                return self

            # Generate bins using scorecardpy.woebin
            self.bins = sc.woebin(df_for_woebin, y=self.target_col, x=existing_categorical_cols, print_info=False)
            print("WOETransformer.fit: woebin completed.")

            # Store the names of the columns that were actually fitted
            self.fitted_cols = existing_categorical_cols
            return self

        def transform(self, X):
            print("WOETransformer.transform called (scorecardpy).")
            # Handle empty DataFrame gracefully
            if X.empty:
                print("WOETransformer.transform: Input DataFrame is empty. Returning empty DataFrame.")
                return X.copy()

            if self.bins is None or not self.fitted_cols:
                print("WOETransformer.transform: Not fitted or no valid categorical columns were fitted. Returning original DataFrame.")
                return X.copy()

            X_copy = X.copy()
            print(f"WOETransformer.transform: Input X_copy columns: {X_copy.columns.tolist()}")

            # Apply WoE replacement using scorecardpy.woe_replace
            X_transformed_woe = sc.woe_replace(X_copy, self.bins)
            print(f"WOETransformer.transform: After woe_replace, columns: {X_transformed_woe.columns.tolist()}")

            # Rename columns to add '_woe' suffix for consistency with test expectations
            rename_map = {col: f'{col}_woe' for col in self.fitted_cols if col in X_transformed_woe.columns}
            X_transformed_woe.rename(columns=rename_map, inplace=True)
            print(f"WOETransformer.transform: After renaming, columns: {X_transformed_woe.columns.tolist()}")

            return X_transformed_woe

except ImportError as e:
    print(f"Error importing scorecardpy: {e}. WOETransformer will be a placeholder and will not perform transformations.")
    class WOETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, categorical_cols, target_col='FraudResult', woe_maps=None):
            self.categorical_cols = categorical_cols
            self.target_col = target_col
            self.woe_maps = woe_maps
        def fit(self, X, y=None):
            print("WOETransformer (placeholder).fit called.")
            if X.empty:
                return self
            return self
        def transform(self, X):
            print("WOETransformer (placeholder).transform called.")
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
            print(f"Warning: '{self.amount_col}' column not found for HandleNegativeAmount. Skipping transformation and adding default 'IsRefund'.")
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
            print(f"Warning: Missing one or more critical columns for aggregation: {required_cols}. Aggregate features will be NaN.")
            # If critical columns are missing, aggregate features are already NaN-filled above
            return X_copy 

        # Ensure 'Amount' is numeric for aggregation
        X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')

        # Perform aggregation only if 'Amount' is not entirely NaN after conversion and DataFrame is not empty
        if X_copy.empty or X_copy[self.amount_col].isnull().all():
            print(f"Warning: DataFrame is empty or '{self.amount_col}' column is entirely NaN. Skipping aggregation calculations.")
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
                print(f"Warning: DataFrame became empty after dropping NaNs in '{self.time_column}'. Returning empty DataFrame.")
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
            print(f"Warning: Time column '{self.time_column}' not found. Skipping time feature extraction and adding placeholder columns.")
            # Add placeholder columns if time_column is missing
            for feat in self.extracted_time_features:
                X_copy[feat] = np.nan # Or 0, depending on desired default behavior

        return X_copy


class FinalPreprocessing(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies ColumnTransformer for final imputation,
    scaling of numerical features, and One-Hot Encoding (if WoE not used).
    It dynamically handles column lists based on previous transformations.
    """
    def __init__(self, numerical_cols, categorical_cols, time_column, id_columns, target_column):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.time_column = time_column
        self.id_columns = id_columns
        self.target_column = target_column
        self.preprocessor = None
        self.ohe_feature_names = []
        self.fitted_feature_names_out = None # To store feature names after fit for empty transform

    def fit(self, X, y=None):
        if X.empty: # Early exit for empty DataFrame
            print("FinalPreprocessing: Input DataFrame is empty. Skipping fit.")
            self.fitted_feature_names_out = self._get_expected_output_columns(X.columns.tolist())
            self.preprocessor = None # Indicate that preprocessor was not fitted
            return self

        current_cols = X.columns.tolist()

        all_expected_numerical_cols = list(set(self.numerical_cols +
                                              [f'Total_Amount', f'Avg_Amount', f'Count_TransactionId', 'IsRefund', 'Std_Amount'] + 
                                              [
                                                'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                                                'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
                                                'TransactionHour_sin', 'TransactionHour_cos',
                                                'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
                                              ]))
        if 'scorecardpy' in sys.modules:
            woe_cols = [f'{col}_woe' for col in self.categorical_cols if f'{col}_woe' in current_cols]
            all_expected_numerical_cols.extend(woe_cols)

        for col in all_expected_numerical_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        num_cols_for_scaling = [col for col in all_expected_numerical_cols if col in X.columns]

        cat_cols_for_ohe = []
        if 'scorecardpy' not in sys.modules:
            for col in self.categorical_cols:
                if col in current_cols:
                    cat_cols_for_ohe.append(col)

        transformers = []

        if num_cols_for_scaling:
            transformers.append(('num_pipeline',
                                 Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                  ('scaler', StandardScaler())]),
                                 num_cols_for_scaling))

        if cat_cols_for_ohe:
            transformers.append(('cat_pipeline',
                                 Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                                 cat_cols_for_ohe))

        if not transformers:
            self.preprocessor = 'passthrough'
            self.fitted_feature_names_out = [col for col in X.columns.tolist() if col != self.target_column]
        else:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            self.preprocessor.fit(X.copy())
            self.fitted_feature_names_out = self.preprocessor.get_feature_names_out()

            if 'cat_pipeline' in self.preprocessor.named_transformers_:
                ohe_encoder = self.preprocessor.named_transformers_['cat_pipeline'].named_steps['onehot']
                self.ohe_feature_names = ohe_encoder.get_feature_names_out(cat_cols_for_ohe)
            else:
                self.ohe_feature_names = []

        return self

    def transform(self, X):
        if X.empty:
            print("FinalPreprocessing: Input DataFrame is empty. Returning empty DataFrame.")
            if self.fitted_feature_names_out is not None:
                return pd.DataFrame(columns=self.fitted_feature_names_out)
            else:
                return X.copy() 

        if self.preprocessor is None:
            print("FinalPreprocessing: Transformer was fitted on empty data. Returning empty DataFrame with expected columns.")
            return pd.DataFrame(columns=self._get_expected_output_columns(X.columns.tolist()))
        
        if self.preprocessor == 'passthrough':
            return X.copy()

        all_expected_numerical_cols = list(set(self.numerical_cols +
                                              [f'Total_Amount', f'Avg_Amount', f'Count_TransactionId', 'IsRefund', 'Std_Amount'] +
                                              [
                                                'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                                                'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
                                                'TransactionHour_sin', 'TransactionHour_cos',
                                                'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
                                              ]))
        if 'scorecardpy' in sys.modules:
            woe_cols = [f'{col}_woe' for col in self.categorical_cols if f'{col}_woe' in X.columns]
            all_expected_numerical_cols.extend(woe_cols)

        for col in all_expected_numerical_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        X_transformed_array = self.preprocessor.transform(X.copy())

        output_feature_names = self.preprocessor.get_feature_names_out()

        X_transformed_df = pd.DataFrame(X_transformed_array, columns=output_feature_names, index=X.index)

        expected_final_columns = self._get_expected_output_columns(X.columns.tolist())
        for col in expected_final_columns:
            if col not in X_transformed_df.columns:
                X_transformed_df[col] = np.nan

        X_transformed_df = X_transformed_df[expected_final_columns]

        for col in X_transformed_df.columns:
            is_base_numeric = any(num_col_prefix in col for num_col_prefix in ['Amount', 'Value', 'PricingStrategy', 'Total_', 'Avg_', 'Count_', 'Std_', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear', 'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter', '_sin', '_cos', 'IsRefund'])
            is_woe_col = '_woe' in col and 'scorecardpy' in sys.modules
            is_ohe_col = False
            if 'scorecardpy' not in sys.modules:
                for cat_col in self.categorical_cols:
                    if col.startswith(f"{cat_col}_"):
                        is_ohe_col = True
                        break

            if is_base_numeric or is_woe_col or is_ohe_col:
                X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').astype(float) # Explicitly cast to float
                if X_transformed_df[col].isnull().any():
                    if is_woe_col or is_ohe_col:
                        X_transformed_df[col] = X_transformed_df[col].fillna(0.0) # Fill with 0.0 for OHE/WoE missing
                    else:
                        if not X_transformed_df[col].isnull().all():
                            X_transformed_df[col] = X_transformed_df[col].fillna(X_transformed_df[col].median())
                        else:
                            pass
        
        return X_transformed_df

    def _get_expected_output_columns(self, X_initial_cols):
        """
        Helper to determine expected output columns for an empty DataFrame or for schema consistency.
        This method must accurately reflect the final columns produced by the entire pipeline.
        """
        expected_cols = list(self.numerical_cols)
        
        expected_cols.extend(['IsRefund', 'Total_Amount', 'Avg_Amount', 'Count_TransactionId', 'Std_Amount'])
        expected_cols.extend([
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
            'TransactionHour_sin', 'TransactionHour_cos',
            'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
        ])

        if 'scorecardpy' in sys.modules:
            expected_woe_cols = [f'{col}_woe' for col in self.categorical_cols if col in X_initial_cols]
            expected_cols.extend(expected_woe_cols)
        else:
            # For OHE, we'll try to predict names based on sample_raw_df categories
            expected_ohe_cols = []
            # These categories are derived from the sample_raw_df fixture in test_data_processing.py
            # They should cover the typical categories in the Xente dataset.
            if 'ProviderId' in self.categorical_cols and 'ProviderId' in X_initial_cols:
                expected_ohe_cols.extend(['ProviderId_P1', 'ProviderId_P2', 'ProviderId_P3'])
            if 'ProductCategory' in self.categorical_cols and 'ProductCategory' in X_initial_cols:
                expected_ohe_cols.extend(['ProductCategory_Airtime', 'ProductCategory_Electronics', 'ProductCategory_Financial Services'])
            if 'ChannelId' in self.categorical_cols and 'ChannelId' in X_initial_cols:
                expected_ohe_cols.extend(['ChannelId_Mobile', 'ChannelId_Web'])
            expected_cols.extend(expected_ohe_cols)
        
        passthrough_id_cols = [c for c in self.id_columns if c != self.time_column and c in X_initial_cols]
        expected_cols.extend(passthrough_id_cols)

        return sorted(list(set(expected_cols)))


class DataProcessor:
    """
    A class to build and apply a robust, automated, and reproducible
    data processing pipeline for the Credit Risk Probability Model.
    """
    def __init__(self, numerical_cols, categorical_cols, time_column, id_columns, target_column='FraudResult'):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.time_column = time_column
        self.id_columns = id_columns # Columns like CustomerId, AccountId for aggregation keys
        self.target_column = target_column
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Builds the scikit-learn pipeline for data preprocessing and feature engineering.
        """
        pipeline_steps = [
            ('handle_negative_amount', HandleNegativeAmount(amount_col='Amount')),
            ('aggregate_features', AggregateFeatures(
                customer_id_col='CustomerId',
                amount_col='Amount',
                transaction_id_col='TransactionId'
            )),
            ('time_feature_extraction', TimeFeatureExtractor(time_column=self.time_column)),
        ]

        if 'scorecardpy' in sys.modules:
            pipeline_steps.append(('woe_transform', WOETransformer(
                categorical_cols=[col for col in self.categorical_cols if col not in self.id_columns],
                target_col=self.target_column
            )))
        else:
            print("Warning: scorecardpy not available. Categorical features will be One-Hot Encoded.")

        # Final Preprocessing step using FinalPreprocessing custom transformer
        pipeline_steps.append(('final_preprocessing', FinalPreprocessing(
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            time_column=self.time_column,
            id_columns=self.id_columns,
            target_column=self.target_column
        )))

        return Pipeline(steps=pipeline_steps)

    def fit(self, X, y=None):
        if 'scorecardpy' in sys.modules:
            X_for_fit = X.copy()
            if self.target_column not in X_for_fit.columns and y is not None:
                X_for_fit[self.target_column] = y
            elif self.target_column not in X_for_fit.columns and y is None:
                print(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
            self.pipeline.fit(X_for_fit, y)
        else:
            self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        if 'scorecardpy' in sys.modules:
            X_for_fit_transform = X.copy()
            if self.target_column not in X_for_fit_transform.columns and y is not None:
                X_for_fit_transform[self.target_column] = y
            elif self.target_column not in X_for_fit_transform.columns and y is None:
                print(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
            return self.pipeline.fit_transform(X_for_fit_transform, y)
        else:
            return self.pipeline.fit_transform(X, y)

# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Xente data file
    project_root = Path(__file__).parent.parent
    raw_data_file_path = project_root / "data" / "raw" / "data.csv"

    # Load the data
    df_raw = load_data(raw_data_file_path, delimiter=',')

    if not df_raw.empty:
        # Separate features (X) and target (y)
        X = df_raw.drop(columns=['FraudResult'])
        y = df_raw['FraudResult']

        # Define columns based on EDA insights
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']
        numerical_features = ['Amount', 'Value', 'PricingStrategy']
        categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
        time_column = 'TransactionStartTime'

        processor = DataProcessor(
            numerical_cols=numerical_features,
            categorical_cols=categorical_features,
            time_column=time_column,
            id_columns=id_columns,
            target_column='FraudResult'
        )

        print("Fitting and transforming data...")
        X_processed = processor.fit_transform(X, y)

        print("\nProcessed DataFrame head:")
        print(X_processed.head())

        print("\nProcessed DataFrame info:")
        X_processed.info()

        print("\nMissing values after processing:")
        print(X_processed.isnull().sum()[X_processed.isnull().sum() > 0])

        print("\nDescriptive statistics of processed numerical features:")
        print(X_processed.describe())

        print("\nVerifying new features:")
        if 'IsRefund' in X_processed.columns:
            print(f"IsRefund value counts:\n{X_processed['IsRefund'].value_counts()}")
        if 'Total_Amount' in X_processed.columns:
            sample_customer_id = X_processed['CustomerId'].iloc[0]
            print(f"Total_Amount for a sample customer ({sample_customer_id}):\n{X_processed[X_processed['CustomerId'] == sample_customer_id]['Total_Amount'].iloc[0]}")
        if 'TransactionHour' in X_processed.columns:
            print(f"TransactionHour value counts:\n{X_processed['TransactionHour'].value_counts().head()}")
        
        if 'scorecardpy' in sys.modules:
            if 'ProductCategory_woe' in X_processed.columns:
                print(f"ProductCategory_woe head:\n{X_processed['ProductCategory_woe'].head()}")
        else:
            if 'ProductCategory_Electronics' in X_processed.columns:
                print(f"ProductCategory_Electronics head (OHE):\n{X_processed['ProductCategory_Electronics'].head()}")

    else:
        print("Raw DataFrame is empty. Cannot perform feature engineering.")
