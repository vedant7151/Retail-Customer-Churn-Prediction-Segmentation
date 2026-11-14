"""
Step 8 - Scenario Analysis
Computes retention ROI under three scenarios: Base Case, Targeted Retention,
and Full Retention Campaign. Also exports the final clean CSVs Tableau will
consume for the dashboard.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "output" / "data"
OUT_REPORTS = ROOT / "output" / "reports"
TABLEAU_DIR = ROOT / "output" / "tableau"

HIGH_RISK_THRESHOLD = 0.5          # churn probability above which a customer is "high risk"
COST_PER_TARGETED_INTERVENTION = 5.0   # GBP — discount voucher, At Risk / Cannot Lose Them
COST_PER_FULL_INTERVENTION = 12.0      # GBP — personalized outreach, all high-risk customers
TARGETED_RETENTION_UPLIFT = 0.20       # assumed % of targeted at-risk customers retained by the intervention
FULL_RETENTION_UPLIFT = 0.30           # assumed % of high-risk customers retained by full campaign


def main():
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    rfm = pd.read_csv(OUT_DATA / "customer_rfm_scored.csv")
    rfm["CustomerID"] = rfm["CustomerID"].astype(str)

    n_customers = len(rfm)
    total_monetary = rfm["Monetary"].sum()

    # Recommended action per segment (for the dashboard + risk ranking table)
    action_map = {
        "Champions": "Reward & upsell — loyalty perks, early access",
        "Loyal Customers": "Nurture — cross-sell, loyalty points",
        "New Customers": "Onboard — welcome series, second-purchase incentive",
        "At Risk": "Re-engage — personalized discount, reminder email",
        "Cannot Lose Them": "Urgent win-back — high-value outreach, priority offer",
        "Lost / Churned": "Low-cost re-engagement campaign only",
    }
    rfm["Recommended_Action"] = rfm["Segment_Label"].map(action_map)

    # === Scenario 1: Base Case — no intervention ===
    base_churn_rate = rfm["Churn_Label"].mean()
    base_revenue_at_risk = rfm.loc[rfm["Churn_Label"] == 1, "Monetary"].sum()

    # === Scenario 2: Targeted Retention — At Risk + Cannot Lose Them segments ===
    targeted_segments = ["At Risk", "Cannot Lose Them"]
    targeted_customers = rfm[rfm["Segment_Label"].isin(targeted_segments)]
    n_targeted = len(targeted_customers)
    targeted_cost = n_targeted * COST_PER_TARGETED_INTERVENTION
    targeted_revenue_at_risk = targeted_customers.loc[targeted_customers["Churn_Label"] == 1, "Monetary"].sum()
    targeted_revenue_recovered = targeted_revenue_at_risk * TARGETED_RETENTION_UPLIFT
    targeted_new_churn_rate = base_churn_rate - (
        targeted_customers["Churn_Label"].sum() * TARGETED_RETENTION_UPLIFT / n_customers
    )
    targeted_roi = (targeted_revenue_recovered - targeted_cost) / targeted_cost if targeted_cost > 0 else 0

    # === Scenario 3: Full Retention Campaign — all customers with churn probability > threshold ===
    # Selection uses OUT-OF-FOLD probabilities (05_churn_models.py): every customer is scored by a
    # model that never trained on them. The in-sample probability (Churn_Probability) is
    # only used for the clearly labelled comparison row below.
    def full_campaign(prob_col):
        hr = rfm[rfm[prob_col] > HIGH_RISK_THRESHOLD]
        n = len(hr)
        cost = n * COST_PER_FULL_INTERVENTION
        at_risk = hr.loc[hr["Churn_Label"] == 1, "Monetary"].sum()
        recovered = at_risk * FULL_RETENTION_UPLIFT
        new_churn = base_churn_rate - (hr["Churn_Label"].sum() * FULL_RETENTION_UPLIFT / n_customers)
        roi = (recovered - cost) / cost if cost > 0 else 0
        return hr, n, cost, at_risk, recovered, new_churn, roi

    high_risk, n_high_risk, full_cost, full_revenue_at_risk, full_revenue_recovered, full_new_churn_rate, full_roi = \
        full_campaign("Churn_Probability_OOF")
    _, n_hr_ins, cost_ins, risk_ins, rec_ins, churn_ins, roi_ins = full_campaign("Churn_Probability")
    # Out-of-sample-only view of the in-sample score: only customers held out of the train split
    test_only = rfm[~rfm["InTrainingSet"]]
    hr_test = test_only[test_only["Churn_Probability"] > HIGH_RISK_THRESHOLD]
    cost_test = len(hr_test) * COST_PER_FULL_INTERVENTION
    rec_test = hr_test.loc[hr_test["Churn_Label"] == 1, "Monetary"].sum() * FULL_RETENTION_UPLIFT
    roi_test = (rec_test - cost_test) / cost_test if cost_test > 0 else 0
    print(f"Full campaign ROI: OOF (all customers) {full_roi:.2f}x | in-sample score (all customers) {roi_ins:.2f}x | "
          f"held-out test split only {roi_test:.2f}x ({len(hr_test)} customers)")

    scenarios = pd.DataFrame([
        {
            "Scenario": "Base Case", "Customers_Targeted": 0, "Cost_GBP": 0,
            "Revenue_At_Risk_GBP": base_revenue_at_risk, "Revenue_Recovered_GBP": 0,
            "Net_ROI": 0, "Projected_Churn_Rate": base_churn_rate,
        },
        {
            "Scenario": "Targeted Retention (At Risk + Cannot Lose Them)",
            "Customers_Targeted": n_targeted, "Cost_GBP": targeted_cost,
            "Revenue_At_Risk_GBP": targeted_revenue_at_risk,
            "Revenue_Recovered_GBP": targeted_revenue_recovered,
            "Net_ROI": targeted_roi, "Projected_Churn_Rate": targeted_new_churn_rate,
        },
        {
            "Scenario": f"Full Retention Campaign (out-of-fold P(churn) > {HIGH_RISK_THRESHOLD:.0%})",
            "Customers_Targeted": n_high_risk, "Cost_GBP": full_cost,
            "Revenue_At_Risk_GBP": full_revenue_at_risk,
            "Revenue_Recovered_GBP": full_revenue_recovered,
            "Net_ROI": full_roi, "Projected_Churn_Rate": full_new_churn_rate,
        },
        {
            "Scenario": f"[Comparison only] Full Retention, in-sample P(churn) > {HIGH_RISK_THRESHOLD:.0%}",
            "Customers_Targeted": n_hr_ins, "Cost_GBP": cost_ins,
            "Revenue_At_Risk_GBP": risk_ins, "Revenue_Recovered_GBP": rec_ins,
            "Net_ROI": roi_ins, "Projected_Churn_Rate": churn_ins,
        },
    ])
    scenarios["Scoring_Basis"] = ["none", "segment rule (ground-truth label)", "out-of-fold probability",
                                  "in-sample probability (train rows were seen by the model)"]
    scenarios.to_csv(OUT_REPORTS / "scenario_roi.csv", index=False)
    print("=== Scenario ROI Analysis ===")
    print(scenarios.round(3).to_string(index=False))

    # --- Business recommendations (plain English) ---
    recs = [
        {
            "Recommendation": "Launch urgent win-back campaign for 'Cannot Lose Them' segment",
            "Based_On": "High historical value + high Recency (lapsing high-value customers)",
            "For_Whom": "Marketing Manager",
            "Expected_Impact": f"Recover ~{TARGETED_RETENTION_UPLIFT:.0%} of at-risk revenue in this segment "
                                f"(~GBP {targeted_revenue_recovered:,.0f})",
        },
        {
            "Recommendation": "Set automated churn alert at 90 days of inactivity",
            "Based_On": "Churn label threshold; Recency is the structural driver of churn by definition",
            "For_Whom": "CRM / Analytics Team",
            "Expected_Impact": "Earlier intervention before churn probability crosses actionable threshold",
        },
        {
            "Recommendation": "Prioritize retention budget on top 20% of customers by Monetary value",
            "Based_On": "Revenue Pareto concentration (see EDA findings)",
            "For_Whom": "Finance + Marketing",
            "Expected_Impact": "Maximum ROI per GBP of retention spend",
        },
        {
            "Recommendation": "Introduce onboarding/loyalty incentives for New Customer segment",
            "Based_On": "Early lifecycle nurturing opportunity",
            "For_Whom": "Marketing Manager",
            "Expected_Impact": "Convert first-time buyers into repeat/loyal customers",
        },
    ]
    pd.DataFrame(recs).to_csv(OUT_REPORTS / "business_recommendations.csv", index=False)

    # =========================================================
    # Export clean CSVs for the Tableau dashboard
    # =========================================================

    # Page 1/3 — customer-level master table (segments, churn scores, actions)
    dashboard_customers = rfm[[
        "CustomerID", "Country", "Recency", "Frequency", "Monetary",
        "RFM_Score", "RFM_Sum", "Segment_Label", "Churn_Label",
        "Churn_Probability", "Churn_Probability_OOF", "InTrainingSet",
        "Recommended_Action", "LastPurchaseDate", "CancellationRate",
    ]].copy()
    dashboard_customers.to_csv(TABLEAU_DIR / "customer_master.csv", index=False)

    # Page 2 — segment summary table
    segment_summary = rfm.groupby("Segment_Label").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
        Churn_Rate=("Churn_Label", "mean"),
        Avg_Churn_Probability=("Churn_Probability", "mean"),
    ).reset_index()
    segment_summary["Churn_Rate"] = segment_summary["Churn_Rate"] * 100
    segment_summary["Avg_Churn_Probability"] = segment_summary["Avg_Churn_Probability"] * 100
    # Intent text = what the archetype name is meant to convey (worded to match the data after audit).
    # Profile_Caption (from 04_segmentation.py) restates the actual centroid values.
    segment_summary["Segment_Intent"] = segment_summary["Segment_Label"].map({
        "Champions": "Bought recently, buy often, spend the most — best customers.",
        "Loyal Customers": "Regular buyers with solid spend — reward and upsell.",
        "New Customers": "Recent buyers with low order count and spend so far — nurture toward repeat purchase.",
        "At Risk": "Low-spend, mostly single/low-order customers who are cooling off — cheap re-engagement; "
                   "not a high-value group.",
        "Cannot Lose Them": "Mid-value (not top-value) customers who have lapsed past 90 days — the highest-spend "
                            "lapsed group, so the priority win-back target; check spend before sizing the offer.",
        "Lost / Churned": "Longest inactive, lowest engagement — low-cost win-back only.",
    })
    seg_profile = pd.read_csv(OUT_REPORTS / "segment_profile.csv")[["Segment_Label", "Profile_Caption"]]
    segment_summary = segment_summary.merge(seg_profile, on="Segment_Label", how="left")
    segment_summary["Segment_Description"] = segment_summary["Segment_Intent"] + " | " + segment_summary["Profile_Caption"]
    segment_summary["Recommended_Action"] = segment_summary["Segment_Label"].map(action_map)
    segment_summary.to_csv(TABLEAU_DIR / "segment_summary.csv", index=False)

    # Page 3 — churn risk ranking (all customers, sorted by probability)
    risk_ranking = dashboard_customers.sort_values("Churn_Probability_OOF", ascending=False).copy()
    risk_ranking.to_csv(TABLEAU_DIR / "churn_risk_ranking.csv", index=False)

    # Page 4 — scenario ROI
    scenarios.to_csv(TABLEAU_DIR / "scenario_roi.csv", index=False)

    # Page 5 — data quality report + monthly cancellation trend (already in output/reports;
    # copy into tableau folder so it's a single source folder to connect to)
    dq_report = pd.read_csv(OUT_REPORTS / "data_quality_report.csv")
    dq_report.to_csv(TABLEAU_DIR / "data_quality_report.csv", index=False)

    country_summary = rfm.groupby("Country").agg(
        Customer_Count=("CustomerID", "count"),
        Total_Revenue=("Monetary", "sum"),
        Churn_Rate=("Churn_Label", "mean"),
    ).reset_index()
    country_summary["Churn_Rate"] = country_summary["Churn_Rate"] * 100
    country_summary = country_summary.sort_values("Total_Revenue", ascending=False)
    country_summary.to_csv(TABLEAU_DIR / "country_summary.csv", index=False)

    # KPI summary for executive overview page
    # Report both: the best-AUC model (may differ from the scoring model) AND the model that
    # actually produces every probability in this pipeline (XGBoost). Names are written out so a
    # dashboard reader cannot mistake one for the other.
    comparison = pd.read_csv(OUT_REPORTS / "model_comparison.csv", index_col=0)
    best_model = comparison["auc"].idxmax()
    oof_val = pd.read_csv(OUT_REPORTS / "oof_validation_xgb.csv")
    kpis = pd.DataFrame([{
        "Overall_Churn_Rate_Pct": base_churn_rate * 100,
        "Total_Customers": n_customers,
        "Total_Revenue_GBP": total_monetary,
        "Revenue_At_Risk_GBP": base_revenue_at_risk,
        "High_Risk_Customer_Count": n_high_risk,   # out-of-fold P(churn) > threshold
        "Best_Model_Name": best_model,
        "Best_Model_AUC": comparison.loc[best_model, "auc"],
        "Scoring_Model_AUC_XGBoost": comparison.loc["XGBoost", "auc"],
        "Scoring_Model_OOF_AUC_XGBoost": oof_val.loc[0, "oof_auc"],
    }])
    kpis.to_csv(TABLEAU_DIR / "executive_kpis.csv", index=False)

    print(f"\nTableau-ready CSVs exported to: {TABLEAU_DIR}")
    for f in sorted(TABLEAU_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
