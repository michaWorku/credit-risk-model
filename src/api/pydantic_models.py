from pydantic import BaseModel
from typing import List, Optional
import datetime

# Define the input schema for a single transaction
class TransactionData(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: datetime.datetime # Use datetime for validation
    PricingStrategy: int

    class Config:
        # Example values
        json_schema_extra = {
            "examples": [
                {
                    "TransactionId": "T12345",
                    "BatchId": "B987",
                    "AccountId": "A654",
                    "SubscriptionId": "S321",
                    "CustomerId": "C9876",
                    "CurrencyCode": "UGX",
                    "CountryCode": 256,
                    "ProviderId": "P1",
                    "ProductId": "ProdX",
                    "ProductCategory": "Electronics",
                    "ChannelId": "Web",
                    "Amount": 150.0,
                    "Value": 150.0,
                    "TransactionStartTime": "2024-06-30T10:30:00Z",
                    "PricingStrategy": 0
                }
            ]
        }

# Define the input schema for a list of transactions
class PredictRequest(BaseModel):
    transactions: List[TransactionData]

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "transactions": [
                        {
                            "TransactionId": "T12345",
                            "BatchId": "B987",
                            "AccountId": "A654",
                            "SubscriptionId": "S321",
                            "CustomerId": "C9876",
                            "CurrencyCode": "UGX",
                            "CountryCode": 256,
                            "ProviderId": "P1",
                            "ProductId": "ProdX",
                            "ProductCategory": "Electronics",
                            "ChannelId": "Web",
                            "Amount": 150.0,
                            "Value": 150.0,
                            "TransactionStartTime": "2024-06-30T10:30:00Z",
                            "PricingStrategy": 0
                        },
                        {
                            "TransactionId": "T12346",
                            "BatchId": "B988",
                            "AccountId": "A655",
                            "SubscriptionId": "S322",
                            "CustomerId": "C9877",
                            "CurrencyCode": "UGX",
                            "CountryCode": 256,
                            "ProviderId": "P2",
                            "ProductId": "ProdY",
                            "ProductCategory": "Airtime",
                            "ChannelId": "Mobile",
                            "Amount": 50.0,
                            "Value": 50.0,
                            "TransactionStartTime": "2024-06-30T11:00:00Z",
                            "PricingStrategy": 1
                        }
                    ]
                }
            ]
        }

# Define the output schema for a single prediction
class PredictionResult(BaseModel):
    TransactionId: str
    predicted_risk_proba: float
    predicted_risk_label: int

# Define the overall response schema
class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    model_name: str
    model_version: str
    message: str
