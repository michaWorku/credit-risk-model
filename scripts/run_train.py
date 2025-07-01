import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
import json
from tqdm import tqdm 
import os 
import cloudpickle

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data
from src.data_processing import DataProcessor
from src.models.model_trainer import ModelTrainer
from src.models.logistic_regression_strategy import LogisticRegressionStrategy
from src.models.decision_tree_strategy import DecisionTreeStrategy
from src.models.random_forest_strategy import RandomForestStrategy
from src.models.xgboost_strategy import XGBoostStrategy
from src.models.model_evaluator import evaluate_classification_model

# Define paths
RAW_DATA_PATH = project_root / "data" / "raw" / "data.csv"
MLRUNS_PATH = project_root / "mlruns" # MLflow tracking directory
EXPORT_DIR = project_root / "exported_model" # New directory for exported model and processor

# --- Ensure MLflow tracking directory exists (for local file tracking) ---
MLRUNS_PATH.mkdir(parents=True, exist_ok=True) 

# --- Ensure export directory exists ---
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI to use local file system for experiment tracking
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
mlflow.set_experiment("Credit Risk Probability Model Training")


def train_and_evaluate_model(
    model_strategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    model_name: str
):
    """
    Trains, tunes, evaluates a model, and logs results to MLflow.

    Args:
        model_strategy: An instance of BaseModelStrategy (e.g., LogisticRegressionStrategy).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        model_name (str): Name of the model for MLflow logging.
    """
    with mlflow.start_run(run_name=f"{model_name}_GridSearch", nested=True) as run: # Use nested=True and capture run
        mlflow.log_param("model_type", model_name)

        trainer = ModelTrainer(model_strategy)

        print(f"\n--- Starting Hyperparameter Tuning for {model_name} ---")
        estimator_for_grid = model_strategy.get_model()

        grid_search = GridSearchCV(
            estimator=estimator_for_grid,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best cross-validation ROC-AUC for {model_name}: {best_score:.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_score)

        if isinstance(model_strategy, LogisticRegressionStrategy):
            trainer.set_strategy(LogisticRegressionStrategy(random_state=42, **best_params))
        elif isinstance(model_strategy, DecisionTreeStrategy):
            trainer.set_strategy(DecisionTreeStrategy(model_type='classifier', random_state=42, **best_params))
        elif isinstance(model_strategy, RandomForestStrategy):
            trainer.set_strategy(RandomForestStrategy(model_type='classifier', random_state=42, **best_params))
        elif isinstance(model_strategy, XGBoostStrategy):
            trainer.set_strategy(XGBoostStrategy(model_type='classifier', random_state=42, **best_params))
        
        trainer.train_model(X_train, y_train)
        
        y_pred_proba = trainer.predict_model(X_test)

        print(f"\n--- Evaluating {model_name} on Test Set ---")
        metrics = evaluate_classification_model(y_test.values, y_pred_proba)

        mlflow.log_metrics(metrics)

        # Log model artifacts to the current run's artifact URI
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name="model", # Use 'name' for logging within the run's artifacts
            signature=mlflow.models.infer_signature(X_test, best_estimator.predict_proba(X_test))
        )
        print(f"Model '{model_name}' logged to MLflow run.")
        
        return best_score, best_estimator, run.info.run_id # Return run_id

def main():
    print("Starting model training and tracking process...")

    # --- 1. Load and Process Data ---
    print("Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH, delimiter=',')
    if df_raw is None or df_raw.empty:
        print("Error: Raw data could not be loaded or is empty. Exiting.")
        return

    if 'FraudResult' not in df_raw.columns:
        print("Error: 'FraudResult' column not found in raw data. Cannot proceed with training. Exiting.")
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

    print("Processing data...")
    X_processed = processor.fit_transform(X_raw.copy(), y_raw.copy())
    print("Data processing complete.")

    # --- Export the DataProcessor for API deployment ---
    processor_export_path = EXPORT_DIR / "data_processor.pkl"
    with open(processor_export_path, "wb") as f:
        cloudpickle.dump(processor, f)
    print(f"DataProcessor exported to {processor_export_path}")

    # --- Log the DataProcessor as an artifact for MLflow tracking (for run_predict/interpret) ---
    with mlflow.start_run(run_name="Log_DataProcessor_Artifact", nested=True):
        processor_artifact_path = "data_processor.pkl" # Temp file for logging
        with open(processor_artifact_path, "wb") as f:
            cloudpickle.dump(processor, f)
        mlflow.log_artifact(processor_artifact_path, artifact_path="processor") # Log as 'processor' artifact
        os.remove(processor_artifact_path) # Clean up local file
        print("DataProcessor logged as MLflow artifact.")


    if 'is_high_risk' not in X_processed.columns:
        print("Error: 'is_high_risk' column not found after data processing. Cannot proceed with training. Exiting.")
        return
    
    feature_cols = [col for col in X_processed.columns if col not in id_columns and col != 'is_high_risk' and col != 'FraudResult']
    
    X_final = X_processed[feature_cols]
    y_final = X_processed['is_high_risk']

    non_numeric_cols = X_final.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found in final features: {non_numeric_cols}. Dropping them.")
        X_final = X_final.drop(columns=non_numeric_cols)

    if X_final.isnull().any().any():
        print("Warning: NaNs detected in final feature set. Imputing with median.")
        for col in X_final.columns:
            if X_final[col].isnull().any():
                if not X_final[col].isnull().all():
                    X_final[col] = X_final[col].fillna(X_final[col].median())
                else:
                    X_final[col] = X_final[col].fillna(0.0)
    
    y_final = pd.to_numeric(y_final, errors='coerce').astype(int)
    if y_final.isnull().any():
        print("Error: NaNs detected in target variable after processing. Cannot proceed. Exiting.")
        return
    if y_final.nunique() < 2:
        print("Error: Target variable has less than two unique classes. Cannot perform classification. Exiting.")
        return


    # --- 2. Split Data ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


    # --- 3. Model Selection and Training with Hyperparameter Tuning and MLflow Tracking ---
    best_model_overall_score = -1
    best_model_overall = None
    best_model_overall_name = ""
    best_model_run_id = None # Store the run_id of the best model

    models_to_train = [
        ("Logistic Regression Classifier", LogisticRegressionStrategy(random_state=42), {
            'solver': ['liblinear', 'lbfgs'], 'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200]
        }),
        ("Decision Tree Classifier", DecisionTreeStrategy(model_type='classifier', random_state=42), {
            'max_depth': [5, 10, 15], 'min_samples_leaf': [1, 5, 10]
        }),
        ("Random Forest Classifier", RandomForestStrategy(model_type='classifier', random_state=42), {
            'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_leaf': [1, 5]
        }),
        ("XGBoost Classifier", XGBoostStrategy(model_type='classifier', random_state=42), {
            'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]
        })
    ]

    print("\n--- Training and Evaluating Models ---")
    for model_name, strategy_instance, param_grid in tqdm(models_to_train, desc="Overall Model Training Progress"):
        current_best_score, current_best_estimator, current_run_id = train_and_evaluate_model(
            strategy_instance, X_train, y_train, X_test, y_test, param_grid, model_name
        )
        if current_best_score > best_model_overall_score:
            best_model_overall_score = current_best_score
            best_model_overall = current_best_estimator
            best_model_overall_name = model_name
            best_model_run_id = current_run_id # Store the run_id

    print(f"\n--- Best Model Overall: {best_model_overall_name} with ROC-AUC: {best_model_overall_score:.4f} ---")

    # --- 4. Register Best Model in MLflow Model Registry (for run_predict/interpret) ---
    if best_model_overall is not None:
        with mlflow.start_run(run_name="Register_Best_Model_Final", nested=True) as final_run:
            # Register the model using its run_id and artifact_path
            # This is crucial for run_predict.py and run_interpret.py
            mlflow.register_model(
                model_uri=f"runs:/{best_model_run_id}/model", # Point to the artifact logged in its run
                name="CreditRiskClassifier"
            )
            mlflow.log_param("final_best_model_name", best_model_overall_name)
            mlflow.log_metric("final_best_roc_auc", best_model_overall_score)
            print(f"Best model '{best_model_overall_name}' registered in MLflow Model Registry as 'CreditRiskClassifier'.")
    else:
        print("No best model identified for registration.")

    # --- Export the Best Model for API deployment ---
    if best_model_overall is not None:
        model_export_path = EXPORT_DIR / "best_model.pkl"
        with open(model_export_path, "wb") as f:
            cloudpickle.dump(best_model_overall, f)
        print(f"Best model '{best_model_overall_name}' exported to {model_export_path}")
    else:
        print("No best model identified for export.")

    print("\nModel training and tracking process completed.")


if __name__ == "__main__":
    main()
