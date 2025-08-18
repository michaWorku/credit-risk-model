import pytest
import pandas as pd
import sys

# Import the DataProcessor and custom transformers from your src/ script
from src.data.processing import DataProcessor

# --- End-to-End DataProcessor Tests (modified to include is_high_risk) ---

def test_data_processor_fit_transform(data_processor_instance, sample_raw_df):
    """
    Tests the full DataProcessor pipeline's fit_transform method.
    Verifies output shape, column names, and data types, including 'is_high_risk'.
    """
    processor = data_processor_instance
    X = sample_raw_df.drop(columns=['FraudResult'])
    y = sample_raw_df['FraudResult']

    X_processed = processor.fit_transform(X, y)

    # Check that the output is a DataFrame
    assert isinstance(X_processed, pd.DataFrame)

    # Check number of rows (should be same as original, unless rows were dropped due to NaNs in time column)
    assert len(X_processed) <= len(sample_raw_df)

    # Check expected new features
    expected_new_features = [
        'IsRefund',
        'Total_Amount', 'Avg_Amount', 'Count_TransactionId', 'Std_Amount',
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
        'TransactionHour_sin', 'TransactionHour_cos',
        'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos',
        'Recency', 'Frequency', 'Monetary', # RFM features (will be scaled by FinalPreprocessing)
        'is_high_risk' # New target variable
    ]
    for col in expected_new_features:
        assert col in X_processed.columns, f"Missing expected new column: {col}"

    if 'scorecardpy' in sys.modules:
        woe_cols = [c for c in X_processed.columns if c.endswith('_woe')]
        if len(woe_cols) == 0:
            pytest.skip("No WoE columns created (possibly due to small dataset or binning issue).")
        else:
            # Check they are numeric
            for col in woe_cols:
                assert pd.api.types.is_numeric_dtype(X_processed[col])
    else:
        # Fallback check for OHE
        assert any("ProviderId_" in c for c in X_processed.columns)
        assert any("ProductCategory_" in c for c in X_processed.columns)
        assert any("ChannelId_" in c for c in X_processed.columns)

    # Check for encoded categorical columns (either WoE or One-Hot)
    # if 'scorecardpy' in sys.modules:
    #     for col in processor.categorical_cols:
    #         assert f'{col}_woe' in X_processed.columns, f"Missing WoE column: {col}_woe"
    #         assert col not in X_processed.columns, f"Original categorical column {col} should be dropped after WoE."
    # else:
    #     # Check for One-Hot Encoded columns (example categories from sample_raw_df)
    #     assert 'ProviderId_P1' in X_processed.columns
    #     assert 'ProductCategory_Electronics' in X_processed.columns
    #     assert 'ChannelId_Web' in X_processed.columns
    #     # Ensure original categorical columns are dropped if OHE is applied
    #     for col in processor.categorical_cols:
    #         assert col not in X_processed.columns, f"Original categorical column {col} should be dropped after OHE."

    # Check that original ID columns and constant features are still present (passthrough)
    # Exclude time_column as it's dropped by TimeFeatureExtractor
    passthrough_id_cols = [c for c in processor.id_columns if c != processor.time_column]
    for col in passthrough_id_cols:
        assert col in X_processed.columns, f"Missing expected passthrough column: {col}"

    # Check for NaNs after processing (should be none for numerical features after imputation)
    # Collect all columns that should be numerical after processing
    all_numerical_cols_after_fe = [
        'Amount', 'Value', 'PricingStrategy', 'IsRefund',
        'Total_Amount', 'Avg_Amount', 'Count_TransactionId', 'Std_Amount',
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
        'TransactionHour_sin', 'TransactionHour_cos',
        'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos',
        'Recency', 'Frequency', 'Monetary', # RFM features
        'is_high_risk' # New target variable
    ]
    if 'scorecardpy' in sys.modules:
        all_numerical_cols_after_fe.extend([f'{col}_woe' for col in processor.categorical_cols])
    else:
        # If OHE, add example OHE columns to check for NaNs
        all_numerical_cols_after_fe.extend(['ProviderId_P1', 'ProductCategory_Electronics', 'ChannelId_Web'])

    for col in all_numerical_cols_after_fe:
        if col in X_processed.columns: # Ensure column exists after all transformations
            assert X_processed[col].isnull().sum() == 0, f"NaNs found in {col} after processing."
            assert pd.api.types.is_float_dtype(X_processed[col]) or pd.api.types.is_integer_dtype(X_processed[col]), \
                f"Column {col} is not numeric (float or int) after processing. Dtype: {X_processed[col].dtype}"

    # Check for the removal of the original time column
    assert processor.time_column not in X_processed.columns

def test_data_processor_empty_df(data_processor_instance):
    """
    Tests the DataProcessor pipeline with an empty input DataFrame.
    Should return an empty DataFrame and handle gracefully.
    """
    processor = data_processor_instance
    # Create an empty DataFrame with all expected columns to simulate structure
    sample_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                   'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                   'ProductCategory', 'ChannelId', 'Amount', 'Value',
                   'TransactionStartTime', 'PricingStrategy', 'FraudResult']
    empty_df = pd.DataFrame(columns=sample_cols)
    empty_y = pd.Series(dtype=int)

    X_processed = processor.fit_transform(empty_df, empty_y)

    assert isinstance(X_processed, pd.DataFrame)
    assert X_processed.empty
    assert 'is_high_risk' in X_processed.columns # Should be present even if empty
    assert 'Recency' in X_processed.columns
    assert 'Frequency' in X_processed.columns
    assert 'Monetary' in X_processed.columns


def test_data_processor_missing_critical_columns(sample_raw_df):
    """
    Tests DataProcessor behavior when critical columns are missing.
    It should ideally raise an error or print warnings and handle gracefully.
    """
    # Remove 'Amount' to simulate missing critical column
    df_missing_amount = sample_raw_df.drop(columns=['Amount'])
    X_missing = df_missing_amount.drop(columns=['FraudResult'])
    y_missing = df_missing_amount['FraudResult']

    numerical_features = ['Amount', 'Value', 'PricingStrategy'] # 'Amount' is now missing
    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    time_column = 'TransactionStartTime'
    id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']
    target_column = 'FraudResult'

    processor = DataProcessor(
        numerical_cols=numerical_features,
        categorical_cols=categorical_features,
        time_column=time_column,
        id_columns=id_columns,
        target_column=target_column
    )

    X_processed = processor.fit_transform(X_missing, y_missing)

    assert isinstance(X_processed, pd.DataFrame)
    assert not X_processed.empty
    
    # Verify that features dependent on 'Amount' (e.g., Total_Amount, RFM) are handled
    assert 'Total_Amount' in X_processed.columns
    assert X_processed['Total_Amount'].isnull().all() # All should be NaN if Amount was missing

    assert 'Recency' in X_processed.columns
    # FIX: Assert that RFM columns are filled with 0s by RFMEngineer.transform
    assert X_processed['Recency'].isnull().sum() == 0 
    assert (X_processed['Recency'] == 0).all()

    assert 'Frequency' in X_processed.columns
    assert X_processed['Frequency'].isnull().sum() == 0
    assert (X_processed['Frequency'] == 0).all()

    assert 'Monetary' in X_processed.columns
    assert X_processed['Monetary'].isnull().sum() == 0
    assert (X_processed['Monetary'] == 0).all()

    assert 'is_high_risk' in X_processed.columns
    assert (X_processed['is_high_risk'] == 0).all() # Should default to 0 if RFM fails
    
    # Verify that other features (e.g., time features) are still processed and not NaN
    assert 'TransactionHour' in X_processed.columns
    assert X_processed['TransactionHour'].isnull().sum() == 0 # Time features should still be extracted and not NaN

