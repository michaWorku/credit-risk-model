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
from src.data.processing import (
    DataProcessor,
    HandleNegativeAmount,
    AggregateFeatures,
    TimeFeatureExtractor,
    WOETransformer # This will now be the scorecardpy-based class or its placeholder
)
from src.features.target_engineering import RFMEngineer # Import the new RFMEngineer

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

@pytest.fixture
def rfm_engineer_instance():
    """Provides a configured RFMEngineer instance for testing."""
    return RFMEngineer(n_clusters=3, random_state=42,
                       customer_id_col='CustomerId',
                       transaction_id_col='TransactionId', # Added for RFMEngineer
                       transaction_time_col='TransactionStartTime',
                       amount_col='Amount')
