import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data
from src.data_processing import DataProcessor
from src.models.model_interpreter import ModelInterpreter

# Define paths
RAW_DATA_PATH = project_root / "data" / "raw" / "data.csv"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory

# Ensure MLflow tracking URI is set
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

def run_model_interpretation(
    registered_model_name: str = "CreditRiskClassifier",
    model_version: str = "latest",
    num_samples_for_shap: int = 500, # Number of samples for global SHAP explanation (increased for better representation)
    num_instances_for_lime: int = 5 # Number of individual instances for LIME explanation
):
    """
    Loads a registered model and performs model interpretation using SHAP and LIME.

    Args:
        registered_model_name (str): The name of the registered model in MLflow.
        model_version (str): The version of the model to load (e.g., '1', 'latest').
        num_samples_for_shap (int): Number of samples from the test set to use for SHAP summary plot.
        num_instances_for_lime (int): Number of individual instances from the test set to explain with LIME.
    """
    print("Starting model interpretation process...")
    print(f"MLRUNS_PATH: {MLRUNS_PATH}") # Added for debugging

    # --- 1. Load Data and Preprocess for Interpretation ---
    print("Loading raw data for data processing setup...")
    df_raw = load_data(RAW_DATA_PATH, delimiter=',')
    if df_raw is None or df_raw.empty:
        print("Error: Raw data could not be loaded or is empty. Exiting interpretation.")
        return

    X_raw = df_raw.drop(columns=['FraudResult'])
    y_raw = df_raw['FraudResult']

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

    print("Processing data to get features for interpretation...")
    X_processed_full = processor.fit_transform(X_raw.copy(), y_raw.copy())

    feature_cols = [col for col in X_processed_full.columns if col not in id_columns and col != 'is_high_risk' and col != 'FraudResult']
    X_final = X_processed_full[feature_cols]
    y_final = X_processed_full['is_high_risk']

    if X_final.isnull().any().any():
        print("Warning: NaNs detected in final feature set for interpretation. Imputing with median.")
        for col in X_final.columns:
            if X_final[col].isnull().any():
                if not X_final[col].isnull().all():
                    X_final[col] = X_final[col].fillna(X_final[col].median())
                else:
                    X_final[col] = X_final[col].fillna(0.0)

    _, X_test, _, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    print(f"Using {len(X_test)} samples from test set for interpretation.")

    # --- 2. Load Registered Model ---
    print(f"Loading model '{registered_model_name}' version '{model_version}' from MLflow Registry...")
    try:
        client = mlflow.tracking.MlflowClient()
        
        if model_version == "latest":
            latest_versions = client.get_latest_versions(registered_model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                raise ValueError(f"No versions found for model '{registered_model_name}' in any stage.")
            latest_version_obj = max(latest_versions, key=lambda mv: int(mv.version))
            actual_model_version = latest_version_obj.version
            print(f"Resolved 'latest' to actual version: {actual_model_version}")
        else:
            actual_model_version = model_version
            
        model_version_obj = client.get_model_version(registered_model_name, actual_model_version)
        
        # FIX: Change 'model' to 'best_model' as per how the final model is logged in run_train.py
        sklearn_model_uri = f"runs:/{model_version_obj.run_id}/model" # This should point to "model" artifact
        sklearn_loaded_model = mlflow.sklearn.load_model(sklearn_model_uri)
        print("Scikit-learn model loaded successfully for interpretation.")

    except Exception as e:
        print(f"Error loading model from MLflow Registry for interpretation: {e}")
        print("Please ensure the model is registered as an sklearn model and the MLflow tracking server is accessible.")
        return

    # --- 3. Initialize Model Interpreter ---
    X_test_sampled_for_background = X_test.sample(n=min(num_samples_for_shap, len(X_test)), random_state=42)

    interpreter = ModelInterpreter(
        model=sklearn_loaded_model,
        feature_names=X_test.columns.tolist(),
        model_type='classification',
        class_names=['Low Risk', 'High Risk'],
        training_data_for_lime=X_test_sampled_for_background.values
    )

    # --- 4. Global Interpretation (SHAP) ---
    print("\n--- Performing Global Interpretation with SHAP ---")
    print(f"Generating SHAP explanations for {min(500, len(X_test))} samples...")
    X_shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    
    interpreter.explain_model_shap(X_shap_sample)
    interpreter.plot_shap_summary(X_shap_sample)

    # --- 5. Local Interpretation (LIME) ---
    print(f"\n--- Performing Local Interpretation with LIME for {num_instances_for_lime} instances ---")
    for i in tqdm(range(min(num_instances_for_lime, len(X_test))), desc="LIME Explanations"):
        instance_to_explain = X_test.iloc[i]
        interpreter.explain_instance_lime(instance_to_explain)
        plt.show()

    print("\nModel interpretation process completed.")


if __name__ == "__main__":
    run_model_interpretation()
