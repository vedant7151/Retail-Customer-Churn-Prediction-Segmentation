"""
Step 6 - Churn Prediction (Classification)
Step 7 - Model Evaluation

Trains Logistic Regression (baseline), Random Forest, and XGBoost to predict
Churn_Label. Uses SMOTE to address class imbalance, evaluates with
AUC-ROC / Precision / Recall / F1 / Confusion Matrix, and produces SHAP
feature-importance plots for the primary XGBoost model.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "output" / "data"
OUT_FIG = ROOT / "output" / "figures"
OUT_REPORTS = ROOT / "output" / "reports"

RANDOM_STATE = 42


def make_xgb():
    """Primary-model hyperparameters (unchanged); shared by the main fit and the out-of-fold fits."""
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )


def main():
    for d in (OUT_DATA, OUT_FIG, OUT_REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    rfm = pd.read_csv(OUT_DATA / "customer_rfm_segmented.csv")
    rfm["CustomerID"] = rfm["CustomerID"].astype(str)

    # --- Feature set ---
    le_country = LabelEncoder()
    rfm["Country_Enc"] = le_country.fit_transform(rfm["Country"].fillna("Unknown"))
    le_segment = LabelEncoder()
    rfm["Segment_Enc"] = le_segment.fit_transform(rfm["Segment_Label"])

    feature_cols = [
        "Frequency", "Monetary", "F_Score", "M_Score",
        "Country_Enc", "CancellationRate",
    ]
    # NOTE: Recency, R_Score, and RFM_Sum are excluded because Churn_Label is
    # defined directly from Recency (Recency > 90 days).
    # Segment_Enc and DaysSinceFirstPurchase are ALSO excluded (audit fix):
    #   - Segment_Enc: the K-Means clusters are built partly from Recency, and two
    #     clusters (Cannot Lose Them, Lost / Churned) are 100% churned by construction.
    #   - DaysSinceFirstPurchase = reference_date - first purchase, which is always >=
    #     Recency, so a value <= 90 guarantees Churn_Label == 0.
    # Both columns remain in the output CSVs for reference but are never given to a model.
    LEAKY_EXCLUDED = ["Recency", "R_Score", "RFM_Sum", "Segment_Enc", "DaysSinceFirstPurchase"]
    assert not set(LEAKY_EXCLUDED) & set(feature_cols), "leaky feature in model inputs"
    X = rfm[feature_cols].copy()
    y = rfm["Churn_Label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # --- SMOTE on training set only ---
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"Train class balance before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Train class balance after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    scaler = StandardScaler()
    X_train_sm_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # === Model A: Logistic Regression (baseline) ===
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_sm_scaled, y_train_sm)
    proba = logreg.predict_proba(X_test_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["Logistic Regression"] = {
        "model": logreg, "proba": proba, "pred": pred,
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    # === Model B: Random Forest ===
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["Random Forest"] = {
        "model": rf, "proba": proba, "pred": pred,
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    # === Model C: XGBoost (primary production model) ===
    xgb = make_xgb()
    xgb.fit(X_train_sm, y_train_sm)
    proba = xgb.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["XGBoost"] = {
        "model": xgb, "proba": proba, "pred": pred,
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    # --- Comparison table ---
    comparison = pd.DataFrame({
        name: {k: v for k, v in r.items() if k in ["auc", "precision", "recall", "f1"]}
        for name, r in results.items()
    }).T
    comparison.to_csv(OUT_REPORTS / "model_comparison.csv")
    print("\n=== Model Comparison ===")
    print(comparison.round(3))

    # --- ROC curves ---
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Churn Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "12_roc_curves.png", bbox_inches="tight")
    plt.close()

    # --- Confusion matrix for primary model (XGBoost) ---
    cm = confusion_matrix(y_test, results["XGBoost"]["pred"])
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — XGBoost (Primary Model)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "13_confusion_matrix_xgb.png", bbox_inches="tight")
    plt.close()

    with open(OUT_REPORTS / "classification_report_xgb.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, results["XGBoost"]["pred"],
                                        target_names=["Retained", "Churned"]))

    # --- Feature importance (XGBoost native) ---
    importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#2E5EAA")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "14_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("\nFeature importance (XGBoost):")
    print(importances)

    # --- SHAP values ---
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "15_shap_summary.png", bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    mean_abs_shap.to_csv(OUT_REPORTS / "shap_feature_importance.csv")
    print("\nMean |SHAP| feature importance:")
    print(mean_abs_shap)

    # --- Score ALL customers with the primary model (for churn risk ranking) ---
    # Churn_Probability = final model scored on everyone. IN-SAMPLE for the 75% of
    # customers that were in the training split (InTrainingSet == True).
    rfm["Churn_Probability"] = xgb.predict_proba(X)[:, 1]
    rfm["InTrainingSet"] = rfm.index.isin(X_train.index)

    # --- Out-of-fold (OOF) probabilities for every customer ---
    # 5-fold stratified CV; SMOTE is fit on each fold's training part only, and the same
    # XGBoost hyperparameters are used. Each customer is scored by a model that never saw
    # them. Used by 06 for the Full Retention scenario.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(rfm))
    for tr_idx, va_idx in skf.split(X, y):
        Xf, yf = SMOTE(random_state=RANDOM_STATE).fit_resample(X.iloc[tr_idx], y.iloc[tr_idx])
        m = make_xgb().fit(Xf, yf)
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    rfm["Churn_Probability_OOF"] = oof
    oof_auc = roc_auc_score(y, oof)
    oof_pred = (oof >= 0.5).astype(int)
    pd.DataFrame([{
        "oof_auc": oof_auc,
        "oof_precision": precision_score(y, oof_pred),
        "oof_recall": recall_score(y, oof_pred),
        "oof_f1": f1_score(y, oof_pred),
        "n_folds": 5,
    }]).to_csv(OUT_REPORTS / "oof_validation_xgb.csv", index=False)
    print(f"\n5-fold OOF XGBoost: AUC {oof_auc:.4f}, "
          f"precision {precision_score(y, oof_pred):.4f}, recall {recall_score(y, oof_pred):.4f}")

    rfm.to_csv(OUT_DATA / "customer_rfm_scored.csv", index=False)

    top_model = comparison["auc"].idxmax()
    print(f"\nBest model by AUC-ROC: {top_model} ({comparison.loc[top_model, 'auc']:.3f})")
    print(f"Saved scored customer file: {OUT_DATA / 'customer_rfm_scored.csv'}")


if __name__ == "__main__":
    main()
