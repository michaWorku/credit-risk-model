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