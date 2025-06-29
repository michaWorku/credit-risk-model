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

# Conditional import for optbinning and definition of WOETransformer
try:
    from optbinning import OptimalBinning, BinningProcess
    print("optbinning imported successfully. WOETransformer will use optbinning.")

    class WOETransformer(BaseEstimator, TransformerMixin):
        """
        Custom transformer to apply Weight of Evidence (WoE) transformation using optbinning.
        This transformer fits the binning process on the training data (when fit is called)
        and then applies the transformation.
        """
        def __init__(self, categorical_cols, target_col='FraudResult'):
            self.categorical_cols = categorical_cols
            self.target_col = target_col
            self.binning_process = None
            self.woe_transformed_cols = []

        def fit(self, X, y=None):
            # optbinning's BinningProcess expects X and y separately for fitting
            # Ensure target column is provided
            if y is None:
                raise ValueError("Target variable 'y' must be provided for WOETransformer.fit().")

            # Filter for categorical columns that actually exist in the dataframe
            existing_categorical_cols = [col for col in self.categorical_cols if col in X.columns]

            if not existing_categorical_cols:
                print("No valid categorical columns for WoE transformation. Initializing empty binning_process.")
                self.binning_process = None
                return self

            # Initialize BinningProcess with just the variable names.
            # The target variable is passed during the .fit() call of BinningProcess.
            self.binning_process = BinningProcess(
                variable_names=existing_categorical_cols,
                # No target_variable here!
            )
            
            # Create a temporary DataFrame for fitting, including only relevant features and target
            # optbinning.BinningProcess.fit expects X (features) and y (target) as separate args
            self.binning_process.fit(X[existing_categorical_cols], y)

            # Store the names of the columns that will be created after transformation
            self.woe_transformed_cols = [f'{col}_woe' for col in existing_categorical_cols]
            return self

        def transform(self, X):
            if self.binning_process is None:
                print("WOETransformer not fitted. Returning original DataFrame.")
                return X.copy()

            # Create a copy to avoid modifying original DataFrame
            X_copy = X.copy()

            # Ensure only the columns that were used for fitting are passed to transform
            cols_to_transform = [col for col in self.categorical_cols if col in X_copy.columns]
            if not cols_to_transform:
                print("No categorical columns found in DataFrame to apply WoE transformation. Returning original DataFrame.")
                return X_copy

            # Apply the transformation. BinningProcess.transform returns a DataFrame
            # with original columns dropped and new _woe columns added.
            # It expects only the features, not the target.
            X_transformed_part = self.binning_process.transform(X_copy[cols_to_transform])

            # Drop original categorical columns from X_copy
            X_copy = X_copy.drop(columns=cols_to_transform, errors='ignore')

            # Merge the transformed WoE columns back into X_copy
            # Use X_copy.index to ensure alignment
            X_copy = X_copy.merge(X_transformed_part, left_index=True, right_index=True, how='left')

            return X_copy

except ImportError as e:
    print(f"Error importing optbinning: {e}. WOETransformer will be a placeholder and will not perform transformations.")
    class WOETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, categorical_cols, target_col='FraudResult', woe_maps=None):
            self.categorical_cols = categorical_cols
            self.target_col = target_col
            self.woe_maps = woe_maps
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X.copy()


# Custom Transformer for handling negative 'Amount' values
class HandleNegativeAmount(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle negative 'Amount' values.
    It creates a binary 'IsRefund' flag and takes the absolute value of 'Amount'.
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
            print(f"Warning: '{self.amount_col}' column not found for HandleNegativeAmount. Skipping transformation.")
            X_copy[self.is_refund_col] = 0 # Add column with default if Amount is missing
        return X_copy

# Custom Transformer for Aggregate Features
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

        # Ensure necessary columns are present
        required_cols = [self.customer_id_col, self.amount_col, self.transaction_id_col]
        if not all(col in X_copy.columns for col in required_cols):
            print(f"Warning: Missing one or more required columns for aggregation: {required_cols}. Skipping aggregation.")
            # Add placeholder columns filled with NaN if aggregation cannot be performed
            for feat_name in self.agg_features_names:
                if feat_name not in X_copy.columns:
                    X_copy[feat_name] = np.nan
            return X_copy

        # Group by CustomerId and aggregate
        # Ensure 'Amount' is numeric for aggregation
        X_copy[self.amount_col] = pd.to_numeric(X_copy[self.amount_col], errors='coerce')

        customer_agg = X_copy.groupby(self.customer_id_col).agg(
            total_amount=(self.amount_col, 'sum'),
            avg_amount=(self.amount_col, 'mean'),
            transaction_count=(self.transaction_id_col, 'count'),
            std_amount=(self.amount_col, lambda x: x.std(skipna=True)) # Ensure std handles NaNs gracefully
        ).reset_index()

        # Rename columns for clarity
        customer_agg.columns = [self.customer_id_col] + self.agg_features_names

        # Merge aggregated features back to the original DataFrame
        # Use a left merge to ensure all original rows are kept
        X_processed = pd.merge(X_copy, customer_agg, on=self.customer_id_col, how='left')

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
            X_copy.dropna(subset=[self.time_column], inplace=True)

            if X_copy.empty:
                print(f"Warning: DataFrame became empty after dropping NaNs in '{self.time_column}'.")
                return X_copy # Return empty if no valid time data

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
        self.target_column = target_column # Needed for WOETransformer context, though not directly processed here
        self.preprocessor = None
        self.ohe_feature_names = []

    def fit(self, X, y=None):
        # Identify numerical and categorical columns after previous transformations
        current_cols = X.columns.tolist()

        # Ensure all numerical-like columns are actually numeric before imputation/scaling
        # This is a crucial step to prevent 'object' dtypes from propagating
        # This loop should cover all columns that are expected to be numerical *at this stage*
        cols_to_convert_to_numeric = list(set(self.numerical_cols +
                                              [f'Total_Amount', f'Avg_Amount', f'Count_TransactionId', f'Std_Amount', 'IsRefund'] +
                                              [
                                                'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                                                'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
                                                'TransactionHour_sin', 'TransactionHour_cos',
                                                'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
                                              ]))
        if 'optbinning' in sys.modules:
            # If optbinning is used, WoE columns are numerical
            woe_cols = [f'{col}_woe' for col in self.categorical_cols if f'{col}_woe' in current_cols]
            cols_to_convert_to_numeric.extend(woe_cols)

        for col in cols_to_convert_to_numeric:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')


        # Identify numerical columns for scaling
        num_cols_for_scaling = [col for col in cols_to_convert_to_numeric if col in X.columns]


        # Identify categorical columns for OHE (if WoE was not used)
        cat_cols_for_ohe = []
        if 'optbinning' not in sys.modules: # Only if WoE was NOT used
            for col in self.categorical_cols:
                if col in current_cols:
                    cat_cols_for_ohe.append(col)
        # If optbinning IS used, then original categorical columns are dropped by WOETransformer
        # and new _woe columns are numerical, so no OHE is needed for them.


        # Define ColumnTransformer based on identified columns
        transformers = []

        if num_cols_for_scaling:
            transformers.append(('num_pipeline',
                                 Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                  ('scaler', StandardScaler())]),
                                 num_cols_for_scaling))

        if cat_cols_for_ohe: # Only if WoE was NOT used
            transformers.append(('cat_pipeline',
                                 Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                                 cat_cols_for_ohe))

        # Identify columns to pass through (IDs, target if present, and any other non-transformed columns)
        # Columns that are handled by the ColumnTransformer's named pipelines
        handled_by_columntransformer = set(num_cols_for_scaling + cat_cols_for_ohe)

        # Columns that were dropped by previous custom transformers
        dropped_by_custom_transformers = set([self.time_column]) # TimeFeatureExtractor drops this
        if 'optbinning' in sys.modules: # WOETransformer drops original categorical columns
            dropped_by_custom_transformers.update(self.categorical_cols)


        # Pass-through columns are those in the current X that are not handled by ColumnTransformer
        # and were not dropped by previous custom transformers.
        passthrough_cols = [
            col for col in X.columns
            if col not in handled_by_columntransformer and col not in dropped_by_custom_transformers
            and col != self.target_column # Ensure target is not passed through if it's in X
        ]

        if not transformers:
            self.preprocessor = 'passthrough'
        else:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough', # Pass through unlisted columns
                verbose_feature_names_out=False # Simpler names for OHE
            )
            # Fit on a copy to avoid modifying X in place if it's passed as a view
            self.preprocessor.fit(X.copy())

            # Store feature names for OneHotEncoder if used
            if 'cat_pipeline' in self.preprocessor.named_transformers_:
                ohe_encoder = self.preprocessor.named_transformers_['cat_pipeline'].named_steps['onehot']
                self.ohe_feature_names = ohe_encoder.get_feature_names_out(cat_cols_for_ohe)
            else:
                self.ohe_feature_names = []

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise RuntimeError("FinalPreprocessing transformer has not been fitted.")

        if self.preprocessor == 'passthrough':
            return X.copy()

        # Ensure numerical-like columns are numeric before transform as well
        # This is a copy of the logic from fit to ensure consistency
        cols_to_convert_to_numeric = list(set(self.numerical_cols +
                                              [f'Total_Amount', f'Avg_Amount', f'Count_TransactionId', f'Std_Amount', 'IsRefund'] +
                                              [
                                                'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                                                'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
                                                'TransactionHour_sin', 'TransactionHour_cos',
                                                'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos'
                                              ]))
        if 'optbinning' in sys.modules:
            woe_cols = [f'{col}_woe' for col in self.categorical_cols if f'{col}_woe' in X.columns] # Check for _woe suffix
            cols_to_convert_to_numeric.extend(woe_cols)

        for col in cols_to_convert_to_numeric:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')


        # Transform the data
        X_transformed_array = self.preprocessor.transform(X.copy()) # Pass a copy to avoid SettingWithCopyWarning

        # Get feature names after transformation using get_feature_names_out
        # This method correctly combines transformed and passthrough column names
        output_feature_names = self.preprocessor.get_feature_names_out()

        # Create a DataFrame from the transformed array
        try:
            X_transformed_df = pd.DataFrame(X_transformed_array, columns=output_feature_names, index=X.index)
        except ValueError as e:
            print(f"Error creating DataFrame after ColumnTransformer: {e}")
            print(f"Expected columns: {len(output_feature_names)}, Actual array columns: {X_transformed_array.shape[1]}")
            # Fallback to default column names if mismatch occurs (shouldn't with get_feature_names_out)
            X_transformed_df = pd.DataFrame(X_transformed_array, index=X.index)

        # Ensure numerical columns are float after final DataFrame creation
        # This is a final safeguard to ensure correct dtypes
        for col in output_feature_names:
            # Check if column name matches a numerical pattern or was expected to be numerical
            # This is a heuristic, adjust as needed
            if any(num_col_prefix in col for num_col_prefix in ['Amount', 'Value', 'PricingStrategy', 'Total_', 'Avg_', 'Count_', 'Std_', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear', 'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter', '_sin', '_cos', 'IsRefund', '_woe']): # Changed _WoE to _woe for optbinning
                X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce')
        
        return X_transformed_df


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

        # Add a step to handle WoE transformation if optbinning is available
        if 'optbinning' in sys.modules:
            # WOETransformer needs to be applied before the final ColumnTransformer
            # because it modifies categorical columns into numerical WoE columns
            pipeline_steps.append(('woe_transform', WOETransformer(
                categorical_cols=[col for col in self.categorical_cols if col not in self.id_columns], # Exclude IDs that are not truly categorical features
                target_col=self.target_column
            )))
        else:
            print("Warning: optbinning not available. Categorical features will be One-Hot Encoded.")


        # Final Preprocessing step using FinalPreprocessing custom transformer
        # This will handle imputation, scaling, and OHE/WoE output integration
        pipeline_steps.append(('final_preprocessing', FinalPreprocessing(
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            time_column=self.time_column,
            id_columns=self.id_columns,
            target_column=self.target_column
        )))

        return Pipeline(steps=pipeline_steps)

    def fit(self, X, y=None):
        """
        Fits the data processing pipeline.
        """
        # Create a copy of X and add y as 'FraudResult' if WoE is used, as WOETransformer expects it
        if 'optbinning' in sys.modules:
            X_for_fit = X.copy()
            if self.target_column not in X_for_fit.columns and y is not None:
                X_for_fit[self.target_column] = y
            elif self.target_column not in X_for_fit.columns and y is None:
                print(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
            self.pipeline.fit(X_for_fit, y) # Pass y to the pipeline fit
        else:
            self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Transforms the raw data using the fitted pipeline.
        """
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the raw data using the pipeline.
        """
        # Create a copy of X and add y as 'FraudResult' if WoE is used, as WOETransformer expects it
        if 'optbinning' in sys.modules:
            X_for_fit_transform = X.copy()
            if self.target_column not in X_for_fit_transform.columns and y is not None:
                X_for_fit_transform[self.target_column] = y
            elif self.target_column not in X_for_fit_transform.columns and y is None:
                print(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
            return self.pipeline.fit_transform(X_for_fit_transform, y) # Pass y to the pipeline fit
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
        # Drop 'CountryCode' and 'CurrencyCode' as they are constant
        # 'TransactionId' and 'BatchId' are identifiers, useful for aggregation but not direct features
        # 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId' are also IDs, 'CustomerId' is used for aggregation
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']

        # Columns that will be numerical after initial handling (e.g., negative Amount)
        numerical_features = ['Amount', 'Value', 'PricingStrategy']

        # Categorical features that need encoding
        categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']

        # Time column
        time_column = 'TransactionStartTime'

        # Initialize the DataProcessor
        processor = DataProcessor(
            numerical_cols=numerical_features,
            categorical_cols=categorical_features,
            time_column=time_column,
            id_columns=id_columns,
            target_column='FraudResult'
        )

        print("Fitting and transforming data...")
        # Fit and transform the data
        # Pass y to fit_transform for WoE calculation if optbinning is used
        X_processed = processor.fit_transform(X, y)

        print("\nProcessed DataFrame head:")
        print(X_processed.head())

        print("\nProcessed DataFrame info:")
        X_processed.info()

        print("\nMissing values after processing:")
        print(X_processed.isnull().sum()[X_processed.isnull().sum() > 0])

        print("\nDescriptive statistics of processed numerical features:")
        print(X_processed.describe())

        # Verify some new features
        print("\nVerifying new features:")
        if 'IsRefund' in X_processed.columns:
            print(f"IsRefund value counts:\n{X_processed['IsRefund'].value_counts()}")
        if 'Total_Amount' in X_processed.columns:
            # Need to get a sample CustomerId that exists in X_processed
            sample_customer_id = X_processed['CustomerId'].iloc[0]
            print(f"Total_Amount for a sample customer ({sample_customer_id}):\n{X_processed[X_processed['CustomerId'] == sample_customer_id]['Total_Amount'].iloc[0]}")
        if 'TransactionHour' in X_processed.columns:
            print(f"TransactionHour value counts:\n{X_processed['TransactionHour'].value_counts().head()}")
        if 'ProductCategory_woe' in X_processed.columns: # Check for WoE (optbinning suffix is _woe)
            print(f"ProductCategory_woe head:\n{X_processed['ProductCategory_woe'].head()}")
        elif 'ProductCategory_Electronics' in X_processed.columns: # Check for OHE (example category)
            print(f"ProductCategory_Electronics head (OHE):\n{X_processed['ProductCategory_Electronics'].head()}")

    else:
        print("Raw DataFrame is empty. Cannot perform feature engineering.")
