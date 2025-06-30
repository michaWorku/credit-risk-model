import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
from tqdm import tqdm 

# Add project root to sys.path to allow absolute imports
# Assuming scripts/run_predict.py is one level down from project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data
from src.data_processing import DataProcessor

# Define paths
RAW_DATA_PATH_FOR_PROCESSOR_FIT = project_root / "data" / "raw" / "data.csv" # Used to fit processor
NEW_DATA_PATH = project_root / "data" / "raw" / "new_transactions.csv" # Placeholder for new data to predict on
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory

# Ensure MLflow tracking URI is set
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

def predict_risk(data_path: Path, registered_model_name: str = "CreditRiskClassifier", model_version: str = "latest"):
    """
    Loads a registered model from MLflow and uses it to predict risk on new data.

    Args:
        data_path (Path): Path to the new data CSV file.
        registered_model_name (str): The name of the registered model in MLflow.
        model_version (str): The version of the model to load (e.g., '1', 'latest').
    """
    print(f"Starting prediction process using model '{registered_model_name}' version '{model_version}'...")

    # --- 1. Load and Process New Data ---
    print(f"Loading new data from {data_path}...")
    new_df_raw = load_data(data_path, delimiter=',')
    if new_df_raw is None or new_df_raw.empty:
        print("Error: New data could not be loaded or is empty. Exiting prediction.")
        return

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

    print(f"Fitting data processor on training data from {RAW_DATA_PATH_FOR_PROCESSOR_FIT} (for consistent transformation)...")
    df_raw_for_processor_fit = load_data(RAW_DATA_PATH_FOR_PROCESSOR_FIT, delimiter=',')
    if df_raw_for_processor_fit is None or df_raw_for_processor_fit.empty:
        print("Error: Training data for processor fit could not be loaded. Cannot ensure consistent transformation. Exiting.")
        return
    
    X_raw_for_processor_fit = df_raw_for_processor_fit.drop(columns=['FraudResult'])
    y_raw_for_processor_fit = df_raw_for_processor_fit['FraudResult']
    
    processor.fit(X_raw_for_processor_fit.copy(), y_raw_for_processor_fit.copy())
    
    print("Processing new data using the fitted preprocessor...")
    X_new_processed = processor.transform(new_df_raw.copy())

    feature_cols = [col for col in X_new_processed.columns if col not in id_columns and col != 'is_high_risk' and col != 'FraudResult']
    X_predict = X_new_processed[feature_cols]

    if X_predict.isnull().any().any():
        print("Warning: NaNs detected in features for prediction. Imputing with median (using 0 for all-NaN columns).")
        for col in X_predict.columns:
            if X_predict[col].isnull().any():
                if not X_predict[col].isnull().all():
                    X_predict[col] = X_predict[col].fillna(X_predict[col].median())
                else:
                    X_predict[col] = X_predict[col].fillna(0.0)

    # --- 2. Load Registered Model ---
    print(f"Loading model '{registered_model_name}' version '{model_version}' from MLflow Registry...")
    try:
        model_uri = f"models:/{registered_model_name}/{model_version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from MLflow Registry: {e}")
        print("Please ensure the model is registered and the MLflow tracking server is accessible.")
        return

    # --- 3. Make Predictions ---
    print("Making predictions on new data...")
    # Wrap the prediction with tqdm for progress
    # Note: loaded_model.predict might not expose iterable steps easily.
    # If it's a single call, a progress bar won't be meaningful.
    # For large datasets, it's often a single, optimized operation.
    # We can simulate progress if needed by breaking it down, but for now,
    # let's assume it's a fast operation or the underlying library handles it.
    
    # If X_predict is very large and loaded_model.predict is slow,
    # you might iterate over chunks of X_predict and predict, wrapping with tqdm.
    # For typical use, the predict call is atomic.
    
    # For a simple progress indicator for the 'predict' step itself,
    # we can just show a message before and after.
    # If the actual prediction is long, consider batching and wrapping the batch loop.
    
    # For demonstration, let's just indicate the start/end of prediction.
    # If X_predict is very large, consider:
    # batch_size = 1000
    # all_predictions_proba = []
    # for i in tqdm(range(0, len(X_predict), batch_size), desc="Predicting in batches"):
    #     batch = X_predict.iloc[i:i + batch_size]
    #     all_predictions_proba.extend(loaded_model.predict(batch))
    # predictions_proba = np.array(all_predictions_proba)
    
    predictions_proba = loaded_model.predict(X_predict) # This is usually a single call

    risk_predictions = (predictions_proba >= 0.5).astype(int)

    new_df_raw['predicted_risk_proba'] = predictions_proba
    new_df_raw['predicted_risk_label'] = risk_predictions

    print("\nPredictions complete. Sample of results:")
    print(new_df_raw[['TransactionId', 'CustomerId', 'Amount', 'predicted_risk_proba', 'predicted_risk_label']].head())

    predictions_output_path = project_root / "data" / "predictions" / "new_data_predictions.csv"
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df_raw.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to: {predictions_output_path}")

if __name__ == "__main__":
    dummy_new_data_path = Path("data/raw/new_transactions.csv")
    dummy_new_data_path.parent.mkdir(parents=True, exist_ok=True)
    if not dummy_new_data_path.exists():
        print(f"Creating a dummy new data file at {dummy_new_data_path} for demonstration.")
        dummy_data = {
            'TransactionId': ['T101', 'T102', 'T103', 'T104'],
            'BatchId': ['B10', 'B10', 'B11', 'B11'],
            'AccountId': ['A10', 'A11', 'A12', 'A10'],
            'SubscriptionId': ['S10', 'S11', 'S12', 'S10'],
            'CustomerId': ['C10', 'C11', 'C12', 'C10'],
            'CurrencyCode': ['UGX'] * 4,
            'CountryCode': [256] * 4,
            'ProviderId': ['P1', 'P2', 'P3', 'P1'],
            'ProductId': ['ProdD', 'ProdE', 'ProdF', 'ProdD'],
            'ProductCategory': ['Electronics', 'Airtime', 'Financial Services', 'Electronics'],
            'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile'],
            'Amount': [60.0, 110.0, 400.0, 90.0],
            'Value': [60.0, 110.0, 400.0, 90.0],
            'TransactionStartTime': [
                '2024-01-01 08:00:00',
                '2024-01-05 13:00:00',
                '2024-01-10 10:00:00',
                '2024-01-12 11:00:00'
            ],
            'PricingStrategy': [0, 1, 2, 0]
        }
        pd.DataFrame(dummy_data).to_csv(dummy_new_data_path, index=False)
        print("Dummy data created. You can modify it or replace it with real new data.")

    predict_risk(dummy_new_data_path)

