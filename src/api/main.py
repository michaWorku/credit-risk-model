from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import sys
import os
import logging
import cloudpickle # For loading custom objects

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow absolute imports
# Assuming src/api/main.py is two levels down from project root
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_loader import load_data
from src.data_processing import DataProcessor
from src.api.pydantic_models import PredictRequest, PredictResponse, PredictionResult

app = FastAPI(
    title="Credit Risk Probability Prediction API",
    description="API for predicting credit risk based on transaction data.",
    version="1.0.0",
)

# Define paths to the exported model and processor within the Docker container
# These paths must match where they are copied in the Dockerfile
EXPORTED_MODEL_DIR_IN_CONTAINER = Path("/app/exported_model")
MODEL_PATH_IN_CONTAINER = EXPORTED_MODEL_DIR_IN_CONTAINER / "best_model.pkl"
PROCESSOR_PATH_IN_CONTAINER = EXPORTED_MODEL_DIR_IN_CONTAINER / "data_processor.pkl"
RAW_DATA_PATH_FOR_PROCESSOR_FIT = "/app/data/raw/data.csv" # Still needed for fallback processor fit

# Global variables
model = None
data_processor = None
model_info = {"name": "CreditRiskClassifier", "version": "1.0.0_exported"} # Static info for exported model

@app.on_event("startup")
async def load_model_and_processor():
    global model, data_processor

    logger.info("--- Starting Model and Processor Loading Process ---")
    
    # --- Load Model Directly ---
    logger.info(f"Attempting to load model from: {MODEL_PATH_IN_CONTAINER}")
    try:
        if not MODEL_PATH_IN_CONTAINER.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH_IN_CONTAINER}")
        with open(MODEL_PATH_IN_CONTAINER, "rb") as f:
            model = cloudpickle.load(f)
        logger.info("Model loaded successfully from exported .pkl file.")
    except Exception as e:
        logger.error(f"FATAL ERROR: Could not load model directly from {MODEL_PATH_IN_CONTAINER}: {e}", exc_info=True)
        sys.exit(1)

    # --- Load DataProcessor Directly ---
    logger.info(f"Attempting to load DataProcessor from: {PROCESSOR_PATH_IN_CONTAINER}")
    try:
        if not PROCESSOR_PATH_IN_CONTAINER.exists():
            raise FileNotFoundError(f"DataProcessor file not found at {PROCESSOR_PATH_IN_CONTAINER}")
        with open(PROCESSOR_PATH_IN_CONTAINER, "rb") as f:
            data_processor = cloudpickle.load(f)
        logger.info("DataProcessor loaded successfully from exported .pkl file.")
    except Exception as e:
        logger.warning(f"Could not load DataProcessor from exported .pkl ({e}). Falling back to fitting from raw data.")
        # Fallback to fitting from raw data if exported processor loading fails
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']
        numerical_features = ['Amount', 'Value', 'PricingStrategy']
        categorical_features = ['ProviderId', 'ProductCategory', 'ChannelId']
        time_column = 'TransactionStartTime'

        data_processor = DataProcessor(
            numerical_cols=numerical_features,
            categorical_cols=categorical_features,
            time_column=time_column,
            id_columns=id_columns,
            target_column='FraudResult'
        )

        # Ensure raw data for processor fit is available
        df_raw_for_processor_fit = load_data(RAW_DATA_PATH_FOR_PROCESSOR_FIT, delimiter=',')
        if df_raw_for_processor_fit is None or df_raw_for_processor_fit.empty:
            logger.error(f"Error: Training data for processor fit could not be loaded from {RAW_DATA_PATH_FOR_PROCESSOR_FIT}. Cannot ensure consistent transformation.")
            sys.exit(1)

        X_raw_for_processor_fit = df_raw_for_processor_fit.drop(columns=['FraudResult'])
        y_raw_for_processor_fit = df_raw_for_processor_fit['FraudResult']
        
        data_processor.fit(X_raw_for_processor_fit.copy(), y_raw_for_processor_fit.copy())
        logger.info("DataProcessor initialized and fitted successfully from raw data.")

    logger.info("Model and DataProcessor loading complete.")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status and model/processor readiness.
    """
    if model is not None and data_processor is not None:
        return JSONResponse(content={"status": "healthy", "model_loaded": True, "processor_ready": True, "model_info": model_info})
    else:
        return JSONResponse(content={"status": "unhealthy", "model_loaded": False, "processor_ready": False}, status_code=503)


@app.post("/predict", response_model=PredictResponse)
async def predict_risk(request: PredictRequest):
    """
    Predicts credit risk probability for a list of transactions.
    """
    if model is None or data_processor is None:
        raise HTTPException(status_code=503, detail="Model or DataProcessor not loaded. Service is not ready.")

    try:
        raw_data_list = [t.model_dump() for t in request.transactions]
        df_new_transactions = pd.DataFrame(raw_data_list)

        df_new_transactions['TransactionStartTime'] = pd.to_datetime(df_new_transactions['TransactionStartTime'], errors='coerce', utc=True)

        logger.info(f"Received {len(df_new_transactions)} transactions for prediction.")

        # Transform data using the loaded processor
        X_processed = data_processor.transform(df_new_transactions.copy())

        # Filter to only the features the model expects
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'ProductId', 'CurrencyCode', 'CountryCode']
        feature_cols = [col for col in X_processed.columns if col not in id_columns and col != 'is_high_risk' and col != 'FraudResult']
        X_predict = X_processed[feature_cols]

        if X_predict.isnull().any().any():
            logger.warning("NaNs detected in features for prediction after preprocessing. Imputing with median.")
            for col in X_predict.columns:
                if X_predict[col].isnull().any():
                    if not X_predict[col].isnull().all():
                        X_predict[col] = X_predict[col].fillna(X_predict[col].median())
                    else:
                        X_predict[col] = X_predict[col].fillna(0.0)

        # Use model.predict_proba for classification models
        predictions_proba = model.predict_proba(X_predict)[:, 1]
        
        risk_predictions = (predictions_proba >= 0.5).astype(int)

        results = []
        for i, transaction_id in enumerate(df_new_transactions['TransactionId']):
            results.append(PredictionResult(
                TransactionId=transaction_id,
                predicted_risk_proba=float(predictions_proba[i]),
                predicted_risk_label=int(risk_predictions[i])
            ))
        
        logger.info(f"Generated {len(results)} predictions.")
        return PredictResponse(
            predictions=results,
            model_name=model_info["name"],
            model_version=model_info["version"],
            message="Predictions generated successfully."
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Helper function for data loading (moved here for completeness, if not already in data_loader.py)
def load_data(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Warning: The file {file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None
