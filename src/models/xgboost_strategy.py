import pandas as pd
import numpy as np
import xgboost as xgb

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy

class XGBoostStrategy(BaseModelStrategy):
    """
    Concrete strategy for XGBoost model (can be Regressor or Classifier).
    """
    def __init__(self, objective: str = 'reg:squarederror', n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initializes the XGBoost model.

        Parameters:
        objective (str): The learning objective. 'reg:squarederror' for regression,
                         'binary:logistic' for binary classification.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed for reproducibility.
        kwargs: Additional parameters for xgb.XGBRegressor or xgb.XGBClassifier.
        """
        super().__init__()
        self.objective = objective
        self._name = "XGBoost Regressor" if 'reg' in objective else "XGBoost Classifier"

        if 'reg' in objective:
            self.model = xgb.XGBRegressor(objective=objective, n_estimators=n_estimators, random_state=random_state, **kwargs)
        elif 'binary' in objective or 'multi' in objective:
            self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported objective: {objective}. Use 'reg:squarederror', 'binary:logistic', etc.")

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the XGBoost model.

        Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable for training.
        """
        if X.empty or y.empty:
            print("Warning: Training data (X or y) is empty for XGBoost. Skipping training.")
            return

        print(f"Training {self.name} model...")
        self.model.fit(X, y)
        print(f"{self.name} training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained XGBoost model.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions. For classification, returns class probabilities if objective is binary,
                    otherwise raw predictions. For binary classification, use predict_proba for probabilities.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} model not trained. Call train() first.")
        if X.empty:
            print("Warning: Prediction data (X) is empty for XGBoost. Returning empty array.")
            return np.array([])
            
        if isinstance(self.model, xgb.XGBClassifier) and 'binary' in self.objective:
            # For binary classification, return probabilities of the positive class
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the trained XGBoost model object.
        """
        return self.model

