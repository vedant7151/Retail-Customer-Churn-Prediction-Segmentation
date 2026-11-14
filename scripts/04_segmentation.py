"""
Step 5 - Customer Segmentation (K-Means on RFM)
Elbow method + Silhouette score to pick K, then cluster and assign
plain-English segment labels based on centroid RFM values.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "output" / "data"
OUT_FIG = ROOT / "output" / "figures"
OUT_REPORTS = ROOT / "output" / "reports"

RANDOM_STATE = 42


def main():
    for d in (OUT_DATA, OUT_FIG, OUT_REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    rfm = pd.read_csv(OUT_DATA / "customer_rfm.csv")
    rfm["CustomerID"] = rfm["CustomerID"].astype(str)

    # --- Preprocessing: log-transform Monetary (right-skewed), standardize R/F/M ---
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Monetary"] = np.log1p(features["Monetary"])
    features["Frequency"] = np.log1p(features["Frequency"])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # --- Elbow method + Silhouette score, K = 2..10 ---
    k_range = range(2, 11)
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(k_range), inertias, marker="o", color="#2E5EAA")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method")
    axes[1].plot(list(k_range), sil_scores, marker="o", color="#C0392B")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score by K")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "09_elbow_silhouette.png", bbox_inches="tight")
    plt.close()

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Silhouette scores by K: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
    print(f"K with highest silhouette score: {best_k}")

    # K=6 is HARD-CODED for business interpretability (six campaign actions).
    # Silhouette is mathematically highest at K=2 (see printed scores); best_k above is
    # reported but NOT used. No ">0.05 silhouette override" rule is implemented.
    FINAL_K = 6
    print(f"Using K={FINAL_K} for final segmentation (chosen for interpretability; silhouette-best K is {best_k}).")

    km_final = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = km_final.fit_predict(X)
    final_sil = silhouette_score(X, rfm["Cluster"])
    print(f"Final silhouette score at K={FINAL_K}: {final_sil:.3f}")

    # --- Centroid profile (in original units) for labeling ---
    centroid_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    centroid_profile["CustomerCount"] = rfm.groupby("Cluster").size()
    centroid_profile["ChurnRate"] = rfm.groupby("Cluster")["Churn_Label"].mean() * 100
    centroid_profile = centroid_profile.sort_values("Monetary", ascending=False)
    print("\nCluster centroid profile (original units):")
    print(centroid_profile)

    # --- Rule-based plain-English labeling from centroid ranks ---
    # Greedy assignment: each cluster gets exactly one distinct label, assigned
    # in priority order to whichever remaining cluster best fits that archetype.
    r_rank = centroid_profile["Recency"].rank(ascending=True)   # 1 = lowest recency = best
    f_rank = centroid_profile["Frequency"].rank(ascending=False)  # 1 = highest frequency = best
    m_rank = centroid_profile["Monetary"].rank(ascending=False)   # 1 = highest monetary = best

    n = len(centroid_profile)
    median_rank = (n + 1) / 2
    remaining = list(centroid_profile.index)
    labels = {}

    def pick(score_fn, minimize=True, candidates=None):
        pool = candidates if candidates else remaining
        pool = [i for i in pool if i in remaining]
        if not pool:
            pool = remaining
        best = min(pool, key=score_fn) if minimize else max(pool, key=score_fn)
        remaining.remove(best)
        return best

    # 1. Champions: best (lowest) combined rank across all three dimensions
    c = pick(lambda i: r_rank[i] + f_rank[i] + m_rank[i], minimize=True)
    labels[c] = "Champions"

    # 2. Lost / Churned: worst (highest) combined rank across all three dimensions
    c = pick(lambda i: r_rank[i] + f_rank[i] + m_rank[i], minimize=False)
    labels[c] = "Lost / Churned"

    # 3. Cannot Lose Them: bad recency (above-median rank) but the strongest
    #    remaining frequency & monetary among that bad-recency group
    bad_recency = [i for i in remaining if r_rank[i] > median_rank]
    c = pick(lambda i: f_rank[i] + m_rank[i], minimize=True, candidates=bad_recency)
    labels[c] = "Cannot Lose Them"

    # 4. New Customers: good recency (below-median rank) but the weakest
    #    remaining frequency & monetary among that good-recency group
    good_recency = [i for i in remaining if r_rank[i] <= median_rank]
    c = pick(lambda i: f_rank[i] + m_rank[i], minimize=False, candidates=good_recency)
    labels[c] = "New Customers"

    # 5. At Risk: worst remaining recency (declining engagement)
    c = pick(lambda i: r_rank[i], minimize=False)
    labels[c] = "At Risk"

    # 6. Loyal Customers: whatever remains — reliable, moderate-to-good on all fronts
    labels[remaining[0]] = "Loyal Customers"

    centroid_profile["Segment_Label"] = centroid_profile.index.map(labels)
    rfm["Segment_Label"] = rfm["Cluster"].map(labels)

    # --- Data-driven profile caption per segment ---
    # Labels are rule-based archetype names; the caption restates what the centroid
    # actually looks like so a reader can check the name against the data. Names are kept
    # (they are keys in 06_scenario_roi.py and the Tableau exports); clustering is unchanged.
    pop_mean_m = rfm["Monetary"].mean()
    pop_median_m = rfm["Monetary"].median()

    def caption(i):
        row = centroid_profile.loc[i]
        spend_rank = int(m_rank[i])
        vs_mean = "above" if row["Monetary"] >= pop_mean_m else "below"
        vs_median = "above" if row["Monetary"] >= pop_median_m else "below"
        return (
            f"Avg {row['Recency']:.0f} days since last order, {row['Frequency']:.1f} orders, "
            f"GBP {row['Monetary']:,.0f} spend (spend rank {spend_rank} of {n}; {vs_mean} population mean "
            f"GBP {pop_mean_m:,.0f}, {vs_median} median GBP {pop_median_m:,.0f}); "
            f"{row['ChurnRate']:.0f}% already past the 90-day churn threshold."
        )

    centroid_profile["Profile_Caption"] = [caption(i) for i in centroid_profile.index]

    centroid_profile.to_csv(OUT_REPORTS / "segment_profile.csv")
    print("\nSegment labels assigned:")
    print(centroid_profile[["Recency", "Frequency", "Monetary", "CustomerCount", "ChurnRate", "Segment_Label"]])
    for i in centroid_profile.index:
        print(f"  {centroid_profile.loc[i, 'Segment_Label']}: {centroid_profile.loc[i, 'Profile_Caption']}")

    # --- Scatter plot: Recency vs Monetary colored by segment ---
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=rfm, x="Recency", y="Monetary", hue="Segment_Label",
        palette="tab10", alpha=0.6, s=25
    )
    plt.yscale("log")
    plt.title("Customer Segments: Recency vs Monetary")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "10_segmentation_scatter.png", bbox_inches="tight")
    plt.close()

    # --- Bar chart: avg RFM score per segment ---
    seg_scores = rfm.groupby("Segment_Label")[["R_Score", "F_Score", "M_Score"]].mean()
    seg_scores.plot(kind="bar", figsize=(9, 5))
    plt.title("Average RFM Scores by Segment")
    plt.ylabel("Score (1-5)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "11_segment_rfm_scores.png", bbox_inches="tight")
    plt.close()

    rfm.to_csv(OUT_DATA / "customer_rfm_segmented.csv", index=False)
    print(f"\nSaved: {OUT_DATA / 'customer_rfm_segmented.csv'}")


if __name__ == "__main__":
    main()
