"""
Step 2 - Feature Engineering: RFM
Step 3 - Churn Label Definition

Builds a customer-level table with Recency, Frequency, Monetary, RFM scores,
and a binary churn label based on an inactivity threshold.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "output" / "data"

# Churn threshold: gift/general-merchandise retailer -> 90 days (~1 quarter) of
# inactivity is used as the churn definition. This is a business design
# decision documented in the guide, not a statistically derived number.
CHURN_THRESHOLD_DAYS = 90


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DATA / "clean_transactions.csv", parse_dates=["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Exclude flagged wholesale customers from the retail RFM/segmentation base
    retail_df = df[~df["IsWholesaleCustomer"]].copy()

    reference_date = retail_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_country = (
        retail_df.groupby("CustomerID")["Country"]
        .agg(lambda s: s.mode().iat[0])
        .rename("Country")
    )

    first_purchase = retail_df.groupby("CustomerID")["InvoiceDate"].min().rename("FirstPurchaseDate")
    last_purchase = retail_df.groupby("CustomerID")["InvoiceDate"].max().rename("LastPurchaseDate")

    rfm = retail_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (reference_date - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    rfm = rfm.join(customer_country).join(first_purchase).join(last_purchase)
    rfm["DaysSinceFirstPurchase"] = (reference_date - rfm["FirstPurchaseDate"]).dt.days

    # Cancellation rate: cancelled invoices (all customers, including wholesale-flagged
    # ones since cancellations were removed pre-flagging) vs total invoices attempted
    cancelled = pd.read_csv(OUT_DATA / "cancelled_invoices.csv")
    cancelled["CustomerID"] = cancelled["CustomerID"].astype(str)
    cancel_counts = cancelled.groupby("CustomerID")["Invoice"].nunique().rename("CancelledInvoices")
    rfm = rfm.join(cancel_counts, how="left")
    rfm["CancelledInvoices"] = rfm["CancelledInvoices"].fillna(0)
    rfm["CancellationRate"] = rfm["CancelledInvoices"] / (rfm["Frequency"] + rfm["CancelledInvoices"]).replace(0, np.nan)
    rfm["CancellationRate"] = rfm["CancellationRate"].fillna(0)

    # --- RFM scoring: quintiles (1-5), 5 = best ---
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["RFM_Sum"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # --- Churn label ---
    rfm["Churn_Label"] = (rfm["Recency"] > CHURN_THRESHOLD_DAYS).astype(int)

    rfm = rfm.reset_index()
    rfm.to_csv(OUT_DATA / "customer_rfm.csv", index=False)

    churn_rate = rfm["Churn_Label"].mean() * 100
    print(f"Reference date: {reference_date.date()}")
    print(f"Churn threshold: {CHURN_THRESHOLD_DAYS} days of inactivity")
    print(f"Customers (retail, non-wholesale): {len(rfm):,}")
    print(f"Overall churn rate: {churn_rate:.1f}%")
    print(rfm[["Recency", "Frequency", "Monetary", "CancellationRate", "Churn_Label"]].describe())


if __name__ == "__main__":
    main()
