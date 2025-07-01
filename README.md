# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

Our objective is to develop a Credit Scoring Model for a buy-now-pay-later service at Bati Bank, leveraging alternative data from an eCommerce platform. A thorough understanding of credit risk, regulatory requirements, and data limitations is fundamental to this project's success.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord sets international banking regulations, particularly emphasizing **Pillar 1: Minimum Capital Requirements**, which guides the calculation of capital for credit, operational, and market risk. For credit risk, banks using internal ratings-based (IRB) approaches must employ robust internal models.

This regulatory emphasis on risk measurement profoundly impacts model development:

- **Meticulous Data Lineage:** All data used in the model, from its raw source to its final transformed features, must be traceable and auditable. This ensures data quality, consistency, and supports retrospective analysis, crucial for regulatory compliance and performance monitoring.
- **Data Governance:** Strict processes for data collection, storage, transformation, and access are required to maintain data integrity and security, forming the foundation for reliable risk assessment.
- **Reproducible Pipelines:** All data processing and feature engineering steps must be fully reproducible. This ensures the model can be consistently rebuilt and validated over time, which is a key requirement for regulatory scrutiny.
- **Interpretability (Explainability):** Regulatory bodies frequently demand explanations for credit decisions. This means models cannot be "black boxes"; the drivers behind a risk score must be transparent. This often necessitates using models with inherent interpretability (e.g., Logistic Regression, Decision Trees) or applying explainability techniques (e.g., SHAP, LIME) to more complex models.
- **Well-Documented Models:** Comprehensive documentation is required for the entire model development process, including assumptions, data sources, feature definitions, model selection criteria, and validation results. This documentation serves as essential evidence for internal audits and external regulatory reviews.
- **Robustness and Stability:** Models must demonstrate stability over time and resilience to minor data fluctuations. Basel II promotes continuous monitoring and re-validation to ensure models remain fit for their intended purpose.

In essence, Basel II transforms model building into a highly regulated process where transparency, auditability, and explainability are as critical as predictive accuracy.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Traditional credit scoring relies on explicit "default" labels (e.g., 90 days past due). However, this challenge utilizes alternative data from an eCommerce platform, which does not inherently provide such a direct label.

**Necessity of a Proxy Variable:**

- **Enabling Supervised Learning:** A proxy variable is indispensable because, without a defined target variable, supervised machine learning, which is required to predict credit risk, is not possible.
- **Leveraging Behavioral Insights:** By analyzing Recency, Frequency, and Monetary (RFM) patterns from transaction history, we can infer customer engagement and financial behavior. A "disengaged" customer (e.g., low recency, low frequency, low monetary value) can be hypothesized as a proxy for higher credit risk, as their behavior might correlate with a reduced likelihood of fulfilling future financial obligations.

**Potential Business Risks of Making Predictions Based on this Proxy:**

- **Proxy Mismatch (Type I & II Errors):** The primary risk is that the proxy variable may not perfectly align with actual credit default.
    - **False Positives (Type I Error):** Labelling a genuinely creditworthy customer as "high-risk" due to their RFM behavior (e.g., they simply stopped using the platform but are financially stable). This results in **lost business opportunities** for Bati Bank by denying credit to good customers.
    - **False Negatives (Type II Error):** Conversely, a customer might appear "engaged" by the RFM proxy but still pose a high credit risk (e.g., frequent small transactions but in financial distress). This leads to **increased loan defaults and financial losses** for Bati Bank.
- **Bias Introduction:** The proxy definition could inadvertently introduce biases present in the eCommerce data, potentially leading to unfair or discriminatory credit decisions if certain behavioral patterns are correlated with the proxy but not actual risk.
- **Model Drift:** If the underlying relationship between RFM behavior and actual credit risk evolves over time (e.g., new customer segments emerge, or eCommerce trends shift), the proxy's effectiveness may diminish, leading to degraded model performance.
- **Regulatory Acceptance Challenges:** Regulatory bodies may scrutinize a proxy-based target variable, requiring extensive validation and justification to demonstrate its correlation with actual credit risk and its fairness.

Therefore, while necessary, employing a proxy variable demands meticulous validation, continuous monitoring, and a clear understanding of its limitations and potential impact on business outcomes.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The choice between model complexity and interpretability involves significant trade-offs, particularly in a heavily regulated domain like finance.

**Simple, Interpretable Models (e.g., Logistic Regression with WoE):**

**Pros:**

- **High Interpretability:** Models like Logistic Regression, especially when combined with Weight of Evidence (WoE) transformations, offer clear insights into how each feature influences the probability of default. WoE ensures a monotonic relationship and enhances understandability. This transparency is vital for regulatory compliance (e.g., Basel II) and for explaining credit decisions to customers.
- **Regulatory Acceptance:** Simpler models like Logistic Regression are well-established and generally more readily accepted by financial regulators due to their inherent transparency.
- **Easier Debugging and Auditing:** When a simple model produces an unexpected prediction, it is typically easier to trace back the inputs and coefficients to diagnose the cause.
- **Lower Computational Cost:** These models are generally faster to train and predict, requiring fewer computational resources.

**Cons:**

- **Potentially Lower Performance:** Simpler models may not effectively capture complex, non-linear relationships within the data as well as more sophisticated algorithms. This can lead to lower predictive accuracy (e.g., lower ROC-AUC or F1-score).
- **Feature Engineering Dependency:** Often require more extensive and thoughtful feature engineering (e.g., WoE, binning) to capture non-linearities and interactions, which can be time-consuming.

**Complex, High-Performance Models (e.g., Gradient Boosting Machines - GBMs like XGBoost, LightGBM):**

**Pros:**

- **Higher Predictive Performance:** GBMs are renowned for their ability to capture intricate non-linear relationships and feature interactions, often resulting in superior predictive accuracy and higher ROC-AUC scores.
- **Reduced Explicit Feature Engineering:** Can sometimes perform well with less explicit feature engineering, as they are capable of learning complex patterns intrinsically.
- **Robustness to Outliers/Missing Values:** Some GBM implementations offer built-in mechanisms for handling these or exhibit less sensitivity to them.

**Cons:**

- **Lower Interpretability (Black Box):** GBMs are ensemble models, making it challenging to understand the exact reasoning behind individual predictions. While post-hoc explanation techniques (e.g., SHAP, LIME) exist, they are approximations and can be complex to implement and interpret for regulatory purposes.
- **Regulatory Skepticism:** Regulators often view "black box" models with caution in high-stakes financial applications due to concerns about fairness, bias, and the inability to explain adverse decisions.
- **Higher Computational Cost:** These models can be more resource-intensive and require longer training times, especially with large datasets and extensive hyperparameter tuning.
- **Overfitting Risk:** More susceptible to overfitting if not properly regularized and validated.

**Conclusion on Trade-offs in a Regulated Financial Context:**
In a regulated financial context, the balance often favors **interpretability and regulatory acceptance**, even if it means a slight compromise in raw predictive power. A model that is less accurate but fully auditable and explainable is frequently preferred over a highly accurate "black box" model that cannot withstand regulatory scrutiny or explain loan denials to customers. A common strategic approach involves:

1. Beginning with interpretable models (e.g., Logistic Regression with WoE) as a baseline and for initial deployment due to their transparency.
2. Exploring more complex models for potential performance gains, but always integrating robust explainability frameworks and rigorous validation to ensure compliance with regulatory and business transparency requirements.
3. The ultimate model choice will depend on the specific regulatory environment, the bank's risk appetite, and the performance benefits weighed against the complexity and explainability costs.

## **Project Overview**

This project focuses on building a credit risk probability model leveraging alternative data. It encompasses data loading, robust preprocessing, model training with hyperparameter tuning, evaluation, and interpretability, all integrated with MLflow for experiment tracking and a FastAPI for serving predictions.

**Key Features:**

- **Data Ingestion & Loading:** Efficiently loads raw transaction data from CSV files.
- **Data Preprocessing:** Implements a comprehensive data processing pipeline including handling negative amounts, aggregating features, extracting time-based features, and handling categorical variables (with fallback to One-Hot Encoding if `scorecardpy` is unavailable). It also engineers RFM (Recency, Frequency, Monetary) features and a proxy target variable (`is_high_risk`) for credit risk.
- **Model Training & Evaluation:** Utilizes a strategy pattern to train and evaluate multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with hyperparameter tuning using GridSearchCV. Models are evaluated based on standard classification metrics including ROC-AUC, Accuracy, Precision, Recall, and F1-score.
- **MLflow Integration:** Tracks experiments, parameters, metrics, and models using MLflow for reproducibility and comparison. The best model is registered in the MLflow Model Registry.
- **Model Interpretability:** Provides functionalities for global (SHAP) and local (LIME) model explanations to understand feature importance and individual predictions.
- **FastAPI Prediction Service:** A lightweight and efficient API built with FastAPI for serving real-time credit risk predictions. The API loads the pre-trained model and processor directly for fast inference.
- **Dockerization:** The entire prediction service is containerized using Docker and `docker-compose` for easy deployment and scalability.

## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development and Testing](#development-and-testing)
- [Contributing](#contributing)
- [License](#license)

## **Installation**

### **Prerequisites**

- Python 3.9+
- Git
- `pip` (Python package installer)
- Docker and Docker Compose (for API deployment)

### **Steps**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/credit-risk-model.git
    cd credit-risk-model # Or your project's root directory
    
    ```
    
    *(If you created the project in your current directory, you can skip `git clone` and `cd`.)*
    
2. **Create and activate a virtual environment:**
    
    ```
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate    # On Windows
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

## **Usage**

This project provides scripts to train, evaluate, predict, and interpret the credit risk model.

### **1. Train and Register Models**

This script trains various models, performs hyperparameter tuning, evaluates them, and logs/registers the best model with MLflow. It also exports the best model and the data processor for direct API deployment.

```
python scripts/run_train.py

```

*(After execution, check the `mlruns/` directory for MLflow tracking data and the newly created `exported_model/` directory for `best_model.pkl` and `data_processor.pkl`.)*

### **2. Make Predictions**

Use the registered model from MLflow to make predictions on new data.

```
python scripts/run_predict.py

```

*(This script will generate a dummy `new_transactions.csv` if it doesn't exist and save predictions to `data/predictions/new_data_predictions.csv`.)*

### **3. Interpret Model Predictions**

Utilize SHAP and LIME to understand global feature importance and local prediction explanations.

```
python scripts/run_interpret.py

```

*(This script will generate plots for SHAP summary and individual LIME explanations. Plots will be displayed and may need to be closed manually.)*

### **4. Run the Prediction API (Dockerized)**

Deploy the FastAPI prediction service using Docker Compose. This service loads the manually exported model and processor.

```
# Build the Docker image (only needed the first time or after code changes)
sudo docker-compose build --no-cache

# Run the service
sudo docker-compose up

```

*(The API will be accessible at `http://localhost:8000`. You can test the `/health` endpoint or send prediction requests to `/predict`.)*

## **Project Structure**

```
.
├── .github/                         # GitHub specific configurations (e.g., CI/CD workflows)
│   └── workflows/
│       └── ci.yml                   # CI/CD workflow for tests and linting
├── .gitignore                       # Specifies intentionally untracked files to ignore
├── .dockerignore                    # Specifies files to exclude from Docker build context
├── Dockerfile                       # Dockerfile for building the FastAPI application image
├── docker-compose.yml               # Docker Compose configuration for running the API service
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview, installation, usage
├── exported_model/                  # Directory for manually exported model and data processor (.pkl files)
│   ├── best_model.pkl
│   └── data_processor.pkl
├── src/                             # Core source code for the project
│   ├── api/                         # FastAPI application and Pydantic models
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application entry point
│   │   └── pydantic_models.py       # Pydantic schemas for API requests/responses
│   ├── data_loader.py               # Module for loading raw data
│   ├── data_preparation.py          # Module for inital data preprocessing and preparation
│   ├── data_processing.py           # Module for data preprocessing pipeline
│   ├── models/                      # Machine learning model strategies and interpretation
│   │   ├── base_model_strategy.py   # Abstract base class for model strategies
│   │   ├── decision_tree_strategy.py
│   │   ├── linear_regression_strategy.py
│   │   ├── logistic_regression_strategy.py
│   │   ├── model_evaluator.py       # Functions for model evaluation metrics
│   │   ├── model_interpreter.py     # SHAP and LIME interpretation
│   │   ├── model_trainer.py         # Context for training models using strategies
│   │   ├── random_forest_strategy.py
│   │   └── xgboost_strategy.py
│   ├── target_engineering.py        # Module for RFM and proxy target engineering
│   └── __init__.py                  # Marks src as a Python package
├── scripts/                         # Standalone utility scripts
│   ├── run_interpret.py             # Script to run model interpretation
│   ├── run_predict.py               # Script to run predictions using MLflow model
│   └── run_train.py                 # Script to train, log, register, and export models
├── data/                            # Data storage
│   ├── raw/                         # Original raw data
│   │   ├── data.csv
│   │   ├── data.xlsx
│   │   ├── new_transactions.csv
│   │   ├── Xente_Variable_Definitions.csv
│   │   └── Xente_Variable_Definitions.xlsx
│   ├── processed/                   # Processed data
│   │   └── processed_data.csv
│   ├── predictions/                 # Saved prediction outputs
│   │   └── new_data_predictions.csv
│   ├── results/                     # Placeholder for additional results/outputs
│   └── README.md                    # README for data directory
└── .venv/                           # Python virtual environment (ignored by Git)

```

## **Development and Testing**

- **Running Tests:**
    
    ```
    pytest tests/
    
    ```
    
    *(Ensure you have `pytest` installed, typically via `pip install -r requirements.txt`)*
    
- **Linting:***(You might need to install a linter like `flake8` or `black` via `pip install`)*
    
    ```
    # Example for flake8
    flake8 src/ scripts/ tests/
    # Example for black (auto-formatter)
    black src/ scripts/ tests/
    
    ```
    

## **Contributing**

Contributions are welcome! Please feel free to open issues or submit pull requests.

## **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).