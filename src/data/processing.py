import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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

from src.data.loaders import load_data # Assuming data_loader is in src/
from src.features.target_engineering import RFMEngineer, ProxyTargetEngineer # Import both RFMEngineer and ProxyTargetEngineer

from src.features.transformers import WOETransformer, HandleNegativeAmount, AggregateFeatures, TimeFeatureExtractor


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
            logger.info("FinalPreprocessing: Input DataFrame is empty. Skipping fit.")
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
        # Add RFM features to numerical columns for scaling
        all_expected_numerical_cols.extend(['Recency', 'Frequency', 'Monetary'])

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
            logger.info("FinalPreprocessing: Input DataFrame is empty. Returning empty DataFrame.")
            if self.fitted_feature_names_out is not None:
                return pd.DataFrame(columns=self.fitted_feature_names_out)
            else:
                return X.copy() 

        if self.preprocessor is None:
            logger.info("FinalPreprocessing: Transformer was fitted on empty data. Returning empty DataFrame with expected columns.")
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
        # Add RFM features to numerical columns for scaling
        all_expected_numerical_cols.extend(['Recency', 'Frequency', 'Monetary'])

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
            is_base_numeric = any(num_col_prefix in col for num_col_prefix in ['Amount', 'Value', 'PricingStrategy', 'Total_', 'Avg_', 'Count_', 'Std_', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear', 'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter', '_sin', '_cos', 'IsRefund', 'Recency', 'Frequency', 'Monetary'])
            is_woe_col = '_woe' in col and 'scorecardpy' in sys.modules
            is_ohe_col = False
            if 'scorecardpy' not in sys.modules:
                for cat_col in self.categorical_cols:
                    if col.startswith(f"{cat_col}_"):
                        is_ohe_col = True
                        break

            if is_base_numeric or is_woe_col or is_ohe_col or col == 'is_high_risk': # Explicitly check for is_high_risk
                X_transformed_df[col] = pd.to_numeric(X_transformed_df[col], errors='coerce').astype(float) # Explicitly cast to float
                if X_transformed_df[col].isnull().any():
                    if is_woe_col or is_ohe_col or col == 'is_high_risk': # Fill with 0.0 for OHE/WoE/is_high_risk missing
                        X_transformed_df[col] = X_transformed_df[col].fillna(0.0)
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
        
        # Add RFM features and the new target variable
        expected_cols.extend(['Recency', 'Frequency', 'Monetary', 'is_high_risk'])

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
        
        # Include all ID columns in the expected output, regardless of whether they were in the initial input.
        # If they were missing, they will be added as NaNs by the transform method.
        passthrough_id_cols = [c for c in self.id_columns if c != self.time_column]
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
            # NEW ORDER: RFMEngineer before TimeFeatureExtractor
            ('rfm_engineering', RFMEngineer(
                n_clusters=3,
                random_state=42,
                customer_id_col='CustomerId',
                transaction_id_col='TransactionId',
                transaction_time_col='TransactionStartTime', # RFMEngineer needs this column
                amount_col='Amount'
            )),
            ('time_feature_extraction', TimeFeatureExtractor(time_column=self.time_column)), # This will drop TransactionStartTime
        ]

        if 'scorecardpy' in sys.modules:
            pipeline_steps.append(('woe_transform', WOETransformer(
                categorical_cols=[col for col in self.categorical_cols if col not in self.id_columns],
                target_col=self.target_column
            )))
        else:
            logger.warning("Warning: scorecardpy not available. Categorical features will be One-Hot Encoded.")

        # Final Preprocessing step using FinalPreprocessing custom transformer
        pipeline_steps.append(('final_preprocessing', FinalPreprocessing(
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            time_column=self.time_column, # This column will be dropped by TimeFeatureExtractor, but kept for schema consistency
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
                logger.warning(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
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
                logger.warning(f"Warning: '{self.target_column}' not provided for WoE fitting. WoE might not be correctly fitted.")
            return self.pipeline.fit_transform(X_for_fit_transform, y)
        else:
            return self.pipeline.fit_transform(X, y)

# Example usage for independent testing
if __name__ == "__main__":
    # Define the path to the raw Xente data file
    project_root = Path(__file__).parent.parent.parent
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
        time_column = 'TransactionStartTime' # This column will be dropped by TimeFeatureExtractor

        processor = DataProcessor(
            numerical_cols=numerical_features,
            categorical_cols=categorical_features,
            time_column=time_column,
            id_columns=id_columns,
            target_column='FraudResult'
        )

        logger.info("Fitting and transforming data (including RFM and proxy target engineering)...")
        X_processed = processor.fit_transform(X, y)

        logger.info("\nProcessed DataFrame head:")
        logger.info(X_processed.head())

        logger.info("\nProcessed DataFrame info:")
        X_processed.info()

        logger.info("\nMissing values after processing:")
        logger.info(X_processed.isnull().sum()[X_processed.isnull().sum() > 0])

        logger.info("\nDescriptive statistics of processed numerical features:")
        logger.info(X_processed.describe())

        logger.info("\nVerifying new features:")
        if 'IsRefund' in X_processed.columns:
            logger.info(f"IsRefund value counts:\n{X_processed['IsRefund'].value_counts()}")
        if 'Total_Amount' in X_processed.columns:
            sample_customer_id = X_processed['CustomerId'].iloc[0]
            logger.info(f"Total_Amount for a sample customer ({sample_customer_id}):\n{X_processed[X_processed['CustomerId'] == sample_customer_id]['Total_Amount'].iloc[0]}")
        if 'TransactionHour' in X_processed.columns:
            logger.info(f"TransactionHour value counts:\n{X_processed['TransactionHour'].value_counts().head()}")
        
        if 'scorecardpy' in sys.modules:
            if 'ProductCategory_woe' in X_processed.columns:
                logger.info(f"ProductCategory_woe head:\n{X_processed['ProductCategory_woe'].head()}")
        else:
            if 'ProductCategory_Electronics' in X_processed.columns:
                logger.info(f"ProductCategory_Electronics head (OHE):\n{X_processed['ProductCategory_Electronics'].head()}")
        
        if 'is_high_risk' in X_processed.columns:
            logger.info(f"\n'is_high_risk' value counts:\n{X_processed['is_high_risk'].value_counts()}")
            logger.info(f"'is_high_risk' dtype: {X_processed['is_high_risk'].dtype}")
        
        if 'Recency' in X_processed.columns:
            logger.info(f"\nRecency head:\n{X_processed['Recency'].head()}")
            logger.info(f"Frequency head:\n{X_processed['Frequency'].head()}")
            logger.info(f"Monetary head:\n{X_processed['Monetary'].head()}")

    else:
        logger.warning("Raw DataFrame is empty. Cannot perform feature engineering.")
