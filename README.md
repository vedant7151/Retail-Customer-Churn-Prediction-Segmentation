# Retail Customer Churn Prediction & Segmentation

RFM analysis, K-Means customer segmentation, churn prediction (Logistic Regression, Random Forest, XGBoost) with SHAP explainability and scenario ROI, built on the Online Retail II dataset.

## Pipeline (`scripts/`)
1. `01_load_clean.py` – load and clean transactions
2. `02_rfm_churn.py` – RFM features and 90-day churn label
3. `03_eda.py` – exploratory analysis
4. `04_segmentation.py` – K-Means segmentation
5. `05_churn_models.py` – models, SHAP, out-of-fold scoring
6. `06_scenario_roi.py` – retention scenario ROI and Tableau exports

Place `online_retail_II.xlsx` in the project root, then run the scripts in order.
