"""
Step 4 - Exploratory Data Analysis
Produces charts + a plain-English findings text file.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "output" / "data"
OUT_FIG = ROOT / "output" / "figures"
OUT_REPORTS = ROOT / "output" / "reports"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_FIG / name, bbox_inches="tight")
    plt.close()


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DATA / "clean_transactions.csv", parse_dates=["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    rfm = pd.read_csv(OUT_DATA / "customer_rfm.csv")
    rfm["CustomerID"] = rfm["CustomerID"].astype(str)

    findings = []

    # --- Revenue by country ---
    retail_df = df[~df["IsWholesaleCustomer"]]
    rev_by_country = retail_df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    top10 = rev_by_country.head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top10.values, y=top10.index, color="#2E5EAA")
    plt.xlabel("Revenue (GBP)")
    plt.title("Top 10 Countries by Revenue")
    savefig("01_revenue_by_country.png")

    uk_share = rev_by_country.get("United Kingdom", 0) / rev_by_country.sum() * 100
    findings.append(f"The UK accounts for {uk_share:.1f}% of total retail revenue — the model is largely "
                     f"a UK customer-behavior model, so results should be validated before generalizing to other markets.")

    # --- RFM distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(rfm["Recency"], bins=40, ax=axes[0], color="#2E5EAA")
    axes[0].set_title("Recency Distribution (days)")
    sns.histplot(rfm["Frequency"], bins=40, ax=axes[1], color="#2E5EAA")
    axes[1].set_title("Frequency Distribution (# invoices)")
    axes[1].set_xlim(0, rfm["Frequency"].quantile(0.99))
    sns.histplot(np.log1p(rfm["Monetary"]), bins=40, ax=axes[2], color="#2E5EAA")
    axes[2].set_title("Monetary Distribution (log-transformed)")
    savefig("02_rfm_distributions.png")

    findings.append("Frequency and Monetary are both strongly right-skewed (a small number of customers place "
                     "many/large orders) — this is why Monetary is log-transformed before clustering.")

    # --- Churn rate overall & by country ---
    churn_rate = rfm["Churn_Label"].mean() * 100
    churn_by_country = (
        rfm.groupby("Country")["Churn_Label"].mean().mul(100)
        .loc[rfm["Country"].value_counts().head(10).index]
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(x=churn_by_country.values, y=churn_by_country.index, color="#C0392B")
    plt.xlabel("Churn Rate (%)")
    plt.title("Churn Rate by Country (Top 10 Countries by Customer Count)")
    savefig("03_churn_rate_by_country.png")

    findings.append(f"Overall churn rate (no purchase in 90+ days) is {churn_rate:.1f}% of retail customers.")

    # --- Avg RFM by churned vs retained ---
    grp = rfm.groupby("Churn_Label")[["Recency", "Frequency", "Monetary", "CancellationRate"]].mean()
    grp.index = ["Retained", "Churned"]
    grp.to_csv(OUT_REPORTS / "rfm_by_churn_status.csv")

    plt.figure(figsize=(8, 5))
    grp[["Frequency", "Monetary"]].plot(kind="bar", ax=plt.gca())
    plt.title("Average Frequency & Monetary: Retained vs Churned")
    plt.ylabel("Value")
    plt.xticks(rotation=0)
    savefig("04_churn_vs_retained_rfm.png")

    cancel_lift = (grp.loc["Churned", "CancellationRate"] - grp.loc["Retained", "CancellationRate"])
    direction = "higher" if cancel_lift >= 0 else "lower"
    findings.append(f"Churned customers have a cancellation rate {abs(cancel_lift)*100:.1f} percentage points "
                     f"{direction} than retained customers — "
                     + ("cancellations are an early warning signal of churn."
                        if cancel_lift >= 0 else
                        "cancellation rate is not a strong standalone churn signal in this dataset; "
                        "churn is driven more by declining purchase frequency than by dissatisfaction signals."))
    findings.append(f"Retained customers order {grp.loc['Retained','Frequency']:.1f} times on average vs "
                     f"{grp.loc['Churned','Frequency']:.1f} times for churned customers, "
                     f"and spend GBP {grp.loc['Retained','Monetary']:,.0f} vs GBP {grp.loc['Churned','Monetary']:,.0f}.")

    # --- Top products by revenue ---
    top_products = (
        retail_df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_products.values, y=top_products.index, color="#2E5EAA")
    plt.xlabel("Revenue (GBP)")
    plt.title("Top 10 Products by Revenue")
    savefig("05_top_products.png")

    # --- Monthly revenue trend ---
    monthly = retail_df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
    plt.figure(figsize=(10, 5))
    monthly.plot(marker="o", color="#2E5EAA")
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Revenue (GBP)")
    savefig("06_monthly_revenue_trend.png")

    peak_month = monthly.idxmax().strftime("%B %Y")
    findings.append(f"Revenue peaks in {peak_month}, consistent with a gift/seasonal retailer pattern "
                     f"(pre-holiday buying surge).")

    # --- Cancellation rate by month ---
    cancelled = pd.read_csv(OUT_DATA / "cancelled_invoices.csv", parse_dates=["InvoiceDate"])
    cancelled_monthly = cancelled.set_index("InvoiceDate").resample("ME")["Invoice"].nunique()
    all_invoices_monthly = df.set_index("InvoiceDate").resample("ME")["Invoice"].nunique()
    cancel_rate_monthly = (cancelled_monthly / (cancelled_monthly + all_invoices_monthly) * 100).fillna(0)
    plt.figure(figsize=(10, 5))
    cancel_rate_monthly.plot(marker="o", color="#C0392B")
    plt.title("Cancellation Rate by Month")
    plt.ylabel("Cancellation Rate (%)")
    savefig("07_cancellation_rate_by_month.png")

    # --- Pareto: revenue concentration ---
    cust_rev = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)
    cust_rev["CumRevPct"] = cust_rev["Monetary"].cumsum() / cust_rev["Monetary"].sum() * 100
    cust_rev["CumCustPct"] = (cust_rev.index + 1) / len(cust_rev) * 100
    top20_idx = (cust_rev["CumCustPct"] <= 20).sum()
    rev_from_top20 = cust_rev.loc[:top20_idx, "CumRevPct"].iloc[-1] if top20_idx > 0 else 0

    plt.figure(figsize=(8, 5))
    plt.plot(cust_rev["CumCustPct"], cust_rev["CumRevPct"], color="#2E5EAA")
    plt.axvline(20, color="gray", linestyle="--")
    plt.xlabel("% of Customers (sorted by spend)")
    plt.ylabel("Cumulative % of Revenue")
    plt.title("Revenue Concentration (Pareto Curve)")
    savefig("08_pareto_revenue_concentration.png")

    findings.append(f"The top 20% of customers by spend generate {rev_from_top20:.0f}% of total revenue — "
                     f"a strong Pareto concentration meaning retention spend should prioritize this segment.")

    with open(OUT_REPORTS / "eda_findings.txt", "w", encoding="utf-8") as f:
        f.write("EDA FINDINGS (plain English)\n")
        f.write("=" * 40 + "\n\n")
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n\n")

    print("\n".join(findings))
    print("\nCharts saved to output/figures/, findings saved to output/reports/eda_findings.txt")


if __name__ == "__main__":
    main()
