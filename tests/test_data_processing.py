import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
import os
import datetime

# Add the project root to sys.path to allow imports from src/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the DataProcessor and custom transformers from your src/ script
from src.data_processing import (
    DataProcessor,
    HandleNegativeAmount,
    AggregateFeatures,
    TimeFeatureExtractor,
    WOETransformer # This will now be the scorecardpy-based class or its placeholder
)
from src.target_engineering import RFMEngineer # Import the new RFMEngineer

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_raw_df():
    """
    Provides a sample raw DataFrame mimicking the Xente data structure.
    Includes various scenarios for testing:
    - Negative amount
    - Multiple transactions for one customer
    - Single transaction customer
    - Different product categories, channels, providers
    - Different transaction times
    - Fraud and non-fraud cases
    """
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
        'BatchId': ['B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'B4', 'B4', 'B5', 'B5'],
        'AccountId': ['A1', 'A1', 'A2', 'A3', 'A1', 'A4', 'A2', 'A5', 'A6', 'A6'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S3', 'S1', 'S4', 'S2', 'S5', 'S6', 'S6'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C1', 'C4', 'C2', 'C5', 'C6', 'C6'],
        'CurrencyCode': ['UGX'] * 10, # Constant feature
        'CountryCode': [256] * 10,    # Constant feature
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1', 'P2', 'P1', 'P3', 'P1', 'P2'],
        'ProductId': ['ProdA', 'ProdB', 'ProdC', 'ProdA', 'ProdB', 'ProdC', 'ProdA', 'ProdB', 'ProdC', 'ProdA'],
        'ProductCategory': ['Electronics', 'Airtime', 'Financial Services', 'Electronics', 'Airtime', 'Financial Services', 'Electronics', 'Airtime', 'Financial Services', 'Electronics'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web', 'Mobile', 'Web', 'Mobile', 'Web', 'Mobile'],
        'Amount': [100.0, -50.0, 200.0, 150.0, 75.0, 300.0, 120.0, 80.0, 1000.0, 50.0], # Negative amount for T2
        'Value': [100.0, 50.0, 200.0, 150.0, 75.0, 300.0, 120.0, 80.0, 1000.0, 50.0], # Value is abs(Amount)
        'TransactionStartTime': [
            '2023-01-01 10:00:00', # Sunday, Jan 1, 2023 is Week 52 of 2022
            '2023-01-01 11:30:00',
            '2023-01-02 14:00:00',
            '2023-02-15 08:00:00',
            '2023-01-01 12:00:00', # C1 has 3 transactions
            '2023-03-10 20:00:00',
            '2023-01-02 15:00:00', # C2 has 2 transactions
            '2023-04-05 09:00:00',
            '2023-05-20 16:00:00',
            '2023-05-20 17:00:00' # C6 has 2 transactions
        ],
        'PricingStrategy': [0, 1, 0, 2, 1, 0, 1, 2, 0, 1],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] # T4 and T9 are fraud
    }
    df = pd.DataFrame(data)
    # Ensure correct dtypes for initial processing
    df['Amount'] = df['Amount'].astype(float)
    df['Value'] = df['Value'].astype(float)
    df['FraudResult'] = df['FraudResult'].astype(int)
    return df

@pytest.fixture
def data_processor_instance():
    """
    Provides a configured DataProcessor instance for testing.
    """
    numerical_features = ['Amount', 'Value', 'PricingStrategy']
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
    return processor

# --- Tests for Individual Custom Transformers ---

def test_handle_negative_amount(sample_raw_df):
    """
    Tests the HandleNegativeAmount transformer.
    Verifies 'IsRefund' flag and absolute 'Amount' values.
    """
    transformer = HandleNegativeAmount(amount_col='Amount')
    df_transformed = transformer.transform(sample_raw_df.copy())

    assert 'IsRefund' in df_transformed.columns
    assert df_transformed.loc[df_transformed['TransactionId'] == 'T1', 'IsRefund'].iloc[0] == 0
    assert df_transformed.loc[df_transformed['TransactionId'] == 'T2', 'IsRefund'].iloc[0] == 1
    assert df_transformed.loc[df_transformed['TransactionId'] == 'T5', 'IsRefund'].iloc[0] == 0

    assert df_transformed.loc[df_transformed['TransactionId'] == 'T1', 'Amount'].iloc[0] == 100.0
    assert df_transformed.loc[df_transformed['TransactionId'] == 'T2', 'Amount'].iloc[0] == 50.0 # Was -50.0
    assert df_transformed.loc[df_transformed['TransactionId'] == 'T5', 'Amount'].iloc[0] == 75.0

    assert df_transformed['Amount'].min() >= 0
    assert df_transformed['IsRefund'].dtype == np.int64 # Ensure it's integer

def test_aggregate_features(sample_raw_df):
    """
    Tests the AggregateFeatures transformer.
    Verifies customer-level aggregations.
    """
    transformer = AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount', transaction_id_col='TransactionId')
    # Pre-process negative amounts first as AggregateFeatures expects positive amounts for sum/std
    df_preprocessed_amount = HandleNegativeAmount().transform(sample_raw_df.copy())
    df_transformed = transformer.transform(df_preprocessed_amount)

    assert 'Total_Amount' in df_transformed.columns
    assert 'Avg_Amount' in df_transformed.columns
    assert 'Count_TransactionId' in df_transformed.columns
    assert 'Std_Amount' in df_transformed.columns

    # Verify for Customer C1 (T1, T5, T2 - original amounts 100, 75, -50 -> abs: 100, 75, 50)
    # Total_Amount = 100 + 75 + 50 = 225
    # Avg_Amount = 225 / 3 = 75
    # Count_TransactionId = 3
    # Std_Amount = np.std([100, 75, 50], ddof=1) (sample std)
    c1_data = df_transformed[df_transformed['CustomerId'] == 'C1']
    assert c1_data['Total_Amount'].iloc[0] == pytest.approx(225.0)
    assert c1_data['Avg_Amount'].iloc[0] == pytest.approx(75.0)
    assert c1_data['Count_TransactionId'].iloc[0] == 3
    # Corrected assertion: standard deviation of [100, 75, 50] is 25.0
    assert c1_data['Std_Amount'].iloc[0] == pytest.approx(np.std([100, 75, 50], ddof=1)) # Should be 25.0

    # Verify for Customer C3 (T4 - original amount 150) - single transaction
    # Std_Amount should be 0 for single transaction
    c3_data = df_transformed[df_transformed['CustomerId'] == 'C3']
    assert c3_data['Total_Amount'].iloc[0] == pytest.approx(150.0)
    assert c3_data['Avg_Amount'].iloc[0] == pytest.approx(150.0)
    assert c3_data['Count_TransactionId'].iloc[0] == 1
    assert c3_data['Std_Amount'].iloc[0] == pytest.approx(0.0) # Standard deviation of a single value is 0

    # Ensure output columns are numeric
    assert df_transformed['Total_Amount'].dtype == float
    assert df_transformed['Avg_Amount'].dtype == float
    assert df_transformed['Count_TransactionId'].dtype == np.int64 # Count is int
    assert df_transformed['Std_Amount'].dtype == float


def test_time_feature_extractor(sample_raw_df):
    """
    Tests the TimeFeatureExtractor transformer.
    Verifies extraction of various time components and cyclical features.
    """
    transformer = TimeFeatureExtractor(time_column='TransactionStartTime')
    df_transformed = transformer.transform(sample_raw_df.copy())

    assert 'TransactionStartTime' not in df_transformed.columns # Original column dropped
    assert 'TransactionHour' in df_transformed.columns
    assert 'TransactionDay' in df_transformed.columns
    assert 'TransactionMonth' in df_transformed.columns
    assert 'TransactionYear' in df_transformed.columns
    assert 'TransactionDayOfWeek' in df_transformed.columns
    assert 'TransactionWeekOfYear' in df_transformed.columns
    assert 'TransactionQuarter' in df_transformed.columns
    assert 'TransactionHour_sin' in df_transformed.columns
    assert 'TransactionHour_cos' in df_transformed.columns
    assert 'TransactionDayOfWeek_sin' in df_transformed.columns
    assert 'TransactionDayOfWeek_cos' in df_transformed.columns

    # Verify values for a specific transaction (T1: 2023-01-01 10:00:00)
    t1_data = df_transformed[df_transformed['TransactionId'] == 'T1']
    assert t1_data['TransactionHour'].iloc[0] == 10
    assert t1_data['TransactionDay'].iloc[0] == 1
    assert t1_data['TransactionMonth'].iloc[0] == 1
    assert t1_data['TransactionYear'].iloc[0] == 2023
    assert t1_data['TransactionDayOfWeek'].iloc[0] == 6 # Sunday (0=Monday, 6=Sunday)
    # Corrected assertion for TransactionWeekOfYear based on ISO 8601 for 2023-01-01 (Sunday)
    assert t1_data['TransactionWeekOfYear'].iloc[0] == 52 # 2023-01-01 is in week 52 of 2022 (ISO)
    assert t1_data['TransactionQuarter'].iloc[0] == 1

    # Verify cyclical features (approximate due to float precision)
    assert t1_data['TransactionHour_sin'].iloc[0] == pytest.approx(np.sin(2 * np.pi * 10 / 24))
    assert t1_data['TransactionHour_cos'].iloc[0] == pytest.approx(np.cos(2 * np.pi * 10 / 24))
    assert t1_data['TransactionDayOfWeek_sin'].iloc[0] == pytest.approx(np.sin(2 * np.pi * 6 / 7))
    assert t1_data['TransactionDayOfWeek_cos'].iloc[0] == pytest.approx(np.cos(2 * np.pi * 6 / 7))

    # Ensure new time features are numeric
    for col in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                 'TransactionDayOfWeek', 'TransactionWeekOfYear', 'TransactionQuarter',
                 'TransactionHour_sin', 'TransactionHour_cos',
                 'TransactionDayOfWeek_sin', 'TransactionDayOfWeek_cos']:
        assert pd.api.types.is_numeric_dtype(df_transformed[col]), f"Column {col} is not numeric."


def test_woe_transformer_with_scorecardpy(sample_raw_df): # Renamed function
    """
    Tests the WOETransformer when scorecardpy is installed.
    Verifies that WoE columns are created and original categorical columns are dropped.
    """
    try:
        import scorecardpy as sc
    except ImportError:
        pytest.skip("scorecardpy not installed, skipping WOETransformer test with scorecardpy.")

    categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
    target_column = 'FraudResult'

    X = sample_raw_df.drop(columns=[target_column])
    y = sample_raw_df[target_column]

    transformer = WOETransformer(categorical_features, target_column)
    transformer.fit(X, y) # Fit with X and y

    df_transformed = transformer.transform(X.copy())

    # Verify WoE columns exist (scorecardpy replaces original, then we rename to '_woe' suffix)
    for col in categorical_features:
        assert f'{col}_woe' in df_transformed.columns, f"Missing WoE column: {col}_woe"
        # Verify original categorical columns are dropped
        assert col not in df_transformed.columns, f"Original categorical column {col} should be dropped after WoE."

    # Verify WoE columns are numerical
    for col in categorical_features:
        assert pd.api.types.is_numeric_dtype(df_transformed[f'{col}_woe']), f"WoE column {col}_woe is not numeric."

    # Check if WoE values are not all identical (unless expected for very small dataset)
    # This is a weak check, but better than nothing for a unit test
    for col in categorical_features:
        # Check if there's more than one unique value, or if the column is constant and has only one row (edge case)
        assert df_transformed[f'{col}_woe'].nunique() > 1 or len(df_transformed) <= 1, \
            f"WoE column {col}_woe has no variance, check binning or data."


def test_woe_transformer_without_scorecardpy(sample_raw_df, monkeypatch): # Renamed function
    """
    Tests the WOETransformer when scorecardpy is NOT installed (mocked).
    Verifies that it acts as a placeholder and doesn't transform.
    """
    # Temporarily remove scorecardpy from sys.modules to simulate it not being installed
    with monkeypatch.context() as m:
        if 'scorecardpy' in sys.modules:
            m.delitem(sys.modules, 'scorecardpy')
        # Re-import WOETransformer to get the placeholder version
        from src.data_processing import WOETransformer as PlaceholderWOETransformer

        categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
        target_column = 'FraudResult'

        X = sample_raw_df.drop(columns=[target_column])
        y = sample_raw_df[target_column]

        transformer = PlaceholderWOETransformer(categorical_features, target_column)
        df_transformed = transformer.transform(X.copy())

        # Verify no WoE columns are added
        for col in categorical_features:
            assert f'{col}_woe' not in df_transformed.columns
            assert col in df_transformed.columns # Original columns should remain untouched

        # Verify it returns a copy of the original DataFrame
        pd.testing.assert_frame_equal(df_transformed, X.copy())


# --- Tests for RFMEngineer ---

@pytest.fixture
def rfm_engineer_instance():
    """Provides a configured RFMEngineer instance for testing."""
    return RFMEngineer(n_clusters=3, random_state=42,
                       customer_id_col='CustomerId',
                       transaction_id_col='TransactionId', # Added for RFMEngineer
                       transaction_time_col='TransactionStartTime',
                       amount_col='Amount')

def test_rfm_engineer_calculate_rfm(rfm_engineer_instance, sample_raw_df):
    """Tests the RFMEngineer's internal RFM calculation."""
    # Ensure Amount is positive for RFM calculation
    df_for_rfm = HandleNegativeAmount().transform(sample_raw_df.copy())
    
    # FIX: Ensure TransactionStartTime is datetime before calculating max
    df_for_rfm['TransactionStartTime'] = pd.to_datetime(df_for_rfm['TransactionStartTime'], utc=True)

    # Need a snapshot date for this internal method test
    snapshot_date = df_for_rfm['TransactionStartTime'].max() + datetime.timedelta(days=1)
    
    rfm_df = rfm_engineer_instance._calculate_rfm(df_for_rfm, snapshot_date)

    assert not rfm_df.empty
    assert 'CustomerId' in rfm_df.columns
    assert 'Recency' in rfm_df.columns
    assert 'Frequency' in rfm_df.columns
    assert 'Monetary' in rfm_df.columns

    # Verify RFM values for a specific customer (e.g., C1)
    # C1: T1 (2023-01-01 10:00:00), T5 (2023-01-01 12:00:00), T2 (2023-01-01 11:30:00)
    # Max date for C1 is 2023-01-01 12:00:00
    # Snapshot date from fixture is 2023-05-20 17:00:00 + 1 day = 2023-05-21 17:00:00
    # Recency for C1: (2023-05-21 17:00:00 - 2023-01-01 12:00:00).days + 1 for log transform = 140 + 1 = 141
    # Frequency for C1: 3 + 1 = 4
    # Monetary for C1: (100 + 75 + 50) = 225 (after abs transform)
    
    c1_rfm = rfm_df[rfm_df['CustomerId'] == 'C1'].iloc[0]
    assert c1_rfm['Recency'] == pytest.approx(141) # (Snapshot_date - C1_latest_tx_date).days + 1
    assert c1_rfm['Frequency'] == 4 # 3 transactions + 1
    assert c1_rfm['Monetary'] == pytest.approx(225.0) # Sum of abs amounts

    # Check dtypes
    assert pd.api.types.is_numeric_dtype(rfm_df['Recency'])
    assert pd.api.types.is_numeric_dtype(rfm_df['Frequency'])
    assert pd.api.types.is_numeric_dtype(rfm_df['Monetary'])


def test_rfm_engineer_fit_transform(rfm_engineer_instance, sample_raw_df):
    """
    Tests the full RFMEngineer fit_transform pipeline.
    """
    # Simulate the data coming out of earlier processing steps
    # Ensure Amount is positive and TransactionStartTime is datetime
    df_processed_temp = HandleNegativeAmount().transform(sample_raw_df.copy())
    df_processed_temp['TransactionStartTime'] = pd.to_datetime(sample_raw_df['TransactionStartTime'], utc=True)
    df_processed_temp['TransactionId'] = df_processed_temp['TransactionId'].astype(str) # Ensure string for nunique

    rfm_engineer = rfm_engineer_instance
    df_transformed = rfm_engineer.fit_transform(df_processed_temp.copy())

    assert 'is_high_risk' in df_transformed.columns
    assert df_transformed['is_high_risk'].dtype == np.int64 # Should be int after .astype(int)

    # Check that both 0s and 1s are present (assuming diverse enough sample data)
    assert df_transformed['is_high_risk'].nunique() >= 2

    # Verify RFM features are added as well (unscaled, before FinalPreprocessing)
    assert 'Recency' in df_transformed.columns
    assert 'Frequency' in df_transformed.columns
    assert 'Monetary' in df_transformed.columns

    # Verify that the number of rows is consistent (RFM adds per-customer data, then maps back)
    assert len(df_transformed) == len(sample_raw_df)

    # Check for NaNs in RFM features (should be none after fillna(0))
    assert df_transformed['Recency'].isnull().sum() == 0
    assert df_transformed['Frequency'].isnull().sum() == 0
    assert df_transformed['Monetary'].isnull().sum() == 0


def test_rfm_engineer_empty_df(rfm_engineer_instance):
    """Tests RFMEngineer with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=['CustomerId', 'TransactionId', 'TransactionStartTime', 'Amount'])
    
    # Fit and transform
    rfm_engineer = rfm_engineer_instance
    df_transformed = rfm_engineer.fit_transform(empty_df.copy())

    assert df_transformed.empty
    assert 'is_high_risk' in df_transformed.columns # Column should still be present even if empty
    assert 'Recency' in df_transformed.columns
    assert 'Frequency' in df_transformed.columns
    assert 'Monetary' in df_transformed.columns


def test_rfm_engineer_missing_critical_cols(rfm_engineer_instance, sample_raw_df):
    """Tests RFMEngineer with missing critical columns."""
    df_missing_amount = sample_raw_df.drop(columns=['Amount'])
    # Ensure TransactionStartTime is datetime for the test
    df_missing_amount['TransactionStartTime'] = pd.to_datetime(sample_raw_df['TransactionStartTime'], utc=True)
    df_missing_amount['TransactionId'] = df_missing_amount['TransactionId'].astype(str)

    rfm_engineer = rfm_engineer_instance
    df_transformed = rfm_engineer.fit_transform(df_missing_amount.copy())

    assert 'is_high_risk' in df_transformed.columns
    # If Amount is missing, RFM calculation will result in NaNs, and is_high_risk should default to 0
    assert (df_transformed['is_high_risk'] == 0).all() # All should be 0 if RFM calculation fails

    # Original columns should still be present
    assert 'CustomerId' in df_transformed.columns
    assert 'TransactionStartTime' in df_transformed.columns
    assert 'Recency' in df_transformed.columns
    assert 'Frequency' in df_transformed.columns
    assert 'Monetary' in df_transformed.columns
    
    # FIX: Assert that RFM columns are filled with 0s (as per RFMEngineer.transform logic)
    assert (df_transformed['Recency'] == 0).all()
    assert (df_transformed['Frequency'] == 0).all()
    assert (df_transformed['Monetary'] == 0).all()


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

    # Check for encoded categorical columns (either WoE or One-Hot)
    if 'scorecardpy' in sys.modules:
        for col in processor.categorical_cols:
            assert f'{col}_woe' in X_processed.columns, f"Missing WoE column: {col}_woe"
            assert col not in X_processed.columns, f"Original categorical column {col} should be dropped after WoE."
    else:
        # Check for One-Hot Encoded columns (example categories from sample_raw_df)
        assert 'ProviderId_P1' in X_processed.columns
        assert 'ProductCategory_Electronics' in X_processed.columns
        assert 'ChannelId_Web' in X_processed.columns
        # Ensure original categorical columns are dropped if OHE is applied
        for col in processor.categorical_cols:
            assert col not in X_processed.columns, f"Original categorical column {col} should be dropped after OHE."

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

