import numpy as np
import pandas as pd
import datetime
import pytest
from src.data.processing import HandleNegativeAmount

# --- Tests for RFMEngineer ---

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
