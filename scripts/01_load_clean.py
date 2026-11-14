"""
Step 1 - Data Pipeline & Integrity Layer
Loads both sheets of online_retail_II.xlsx, merges them, runs data integrity
checks with business justification, and writes a clean transaction-level
dataset plus a data quality report.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "online_retail_II.xlsx"
OUT_DATA = ROOT / "output" / "data"
OUT_REPORTS = ROOT / "output" / "reports"

WHOLESALE_QTY_THRESHOLD = 1000  # units per line item — above this, flag as likely wholesale


def load_raw():
    sheet_2009_2010 = pd.read_excel(RAW_XLSX, sheet_name="Year 2009-2010")
    sheet_2010_2011 = pd.read_excel(RAW_XLSX, sheet_name="Year 2010-2011")
    df = pd.concat([sheet_2009_2010, sheet_2010_2011], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Customer ID": "CustomerID"})
    return df


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    report_rows = []
    start_count = len(df)
    report_rows.append({"check": "Raw rows loaded (both sheets)", "rows_removed": 0, "rows_remaining": start_count})

    # --- Check 1: Missing CustomerID (guest checkouts) ---
    before = len(df)
    missing_cust = df["CustomerID"].isna().sum()
    df = df[df["CustomerID"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    report_rows.append({
        "check": "Missing CustomerID removed (guest checkouts, cannot attribute to a customer)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- Check 2: Cancelled invoices (Invoice starts with 'C') ---
    before = len(df)
    df["Invoice"] = df["Invoice"].astype(str)
    is_cancelled = df["Invoice"].str.startswith("C")
    cancelled_df = df[is_cancelled].copy()
    df = df[~is_cancelled].copy()
    report_rows.append({
        "check": "Cancelled invoices removed (Invoice starts with 'C' — not revenue events)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- Check 3: Negative / zero Quantity (returns not tied to a 'C' invoice) ---
    before = len(df)
    negative_qty = df[df["Quantity"] <= 0].copy()
    df = df[df["Quantity"] > 0].copy()
    report_rows.append({
        "check": "Non-positive Quantity removed (returns/data errors distort Monetary)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- Check 4: Zero or negative Price ---
    before = len(df)
    df = df[df["Price"] > 0].copy()
    report_rows.append({
        "check": "Zero/negative Price removed (cannot compute valid TotalPrice)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- Check 5: Duplicate records ---
    before = len(df)
    df = df.drop_duplicates(subset=["Invoice", "StockCode", "CustomerID", "Quantity", "InvoiceDate", "Price"])
    report_rows.append({
        "check": "Duplicate line items dropped (inflates Frequency and Monetary)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- Check 6: Non-product StockCodes (postage, fees, bank charges, adjustments) ---
    before = len(df)
    non_product_codes = {
        "POST", "DOT", "M", "BANK CHARGES", "PADS", "CRUK", "C2", "AMAZONFEE",
        # Added after audit: manual adjustments, test records, discount line, misc non-merchandise
        "ADJUST", "ADJUST2", "TEST001", "TEST002", "D", "SP1002",
    }
    df = df[~df["StockCode"].astype(str).str.upper().isin(non_product_codes)].copy()
    report_rows.append({
        "check": "Non-product StockCodes removed (postage/fees/adjustments are not merchandise sales)",
        "rows_removed": before - len(df), "rows_remaining": len(df)
    })

    # --- TotalPrice ---
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # --- Flag wholesale line items (kept, but flagged — do not drop) ---
    df["IsWholesaleLine"] = df["Quantity"] >= WHOLESALE_QTY_THRESHOLD
    wholesale_customers = (
        df.groupby("CustomerID")["IsWholesaleLine"].mean()
    )
    # A customer is treated as wholesale if >=20% of their line items are bulk orders
    wholesale_customer_ids = wholesale_customers[wholesale_customers >= 0.2].index
    df["IsWholesaleCustomer"] = df["CustomerID"].isin(wholesale_customer_ids)
    n_wholesale_customers = df.loc[df["IsWholesaleCustomer"], "CustomerID"].nunique()
    n_total_customers = df["CustomerID"].nunique()

    report_rows.append({
        "check": f"Wholesale customers flagged ({n_wholesale_customers} of {n_total_customers}) — "
                 f">=20% of their orders are bulk (Qty>={WHOLESALE_QTY_THRESHOLD}); kept but excluded from retail clustering",
        "rows_removed": 0, "rows_remaining": len(df)
    })

    final_count = len(df)
    pct_customer_ids_retained = 100 * (start_count - missing_cust) / start_count

    report_rows.append({
        "check": "FINAL clean retail transaction dataset",
        "rows_removed": start_count - final_count, "rows_remaining": final_count
    })

    report = pd.DataFrame(report_rows)
    report["pct_of_raw"] = (report["rows_remaining"] / start_count * 100).round(2)
    report.to_csv(OUT_REPORTS / "data_quality_report.csv", index=False)

    # Save cancelled and negative-qty rows separately for reference (returns dataset)
    cancelled_df.to_csv(OUT_DATA / "cancelled_invoices.csv", index=False)
    negative_qty.to_csv(OUT_DATA / "returns_negative_qty.csv", index=False)

    # Save clean transaction-level dataset
    df.to_csv(OUT_DATA / "clean_transactions.csv", index=False)

    print("=== Data Quality Report ===")
    print(report.to_string(index=False))
    print(f"\nFinal clean transactions: {final_count:,} rows, {df['CustomerID'].nunique():,} unique customers")
    print(f"CustomerIDs retained: {pct_customer_ids_retained:.1f}% of raw rows had a valid CustomerID")
    print(f"Wholesale customers flagged: {n_wholesale_customers:,} ({100*n_wholesale_customers/n_total_customers:.1f}%)")


if __name__ == "__main__":
    main()
