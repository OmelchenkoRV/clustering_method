import io
import base64
import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
import matplotlib
# Use a non-interactive backend so it works in headless environments like Azure:
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI(title="Clustering Microservice")


def compute_service_stats(price, support_cost, daily_orders, n_days=4):
    net_margin = price - support_cost
    # Compute daily profit for each day: x_ij
    x_ij = daily_orders * net_margin
    # Relative frequency of orders per day
    W_j = daily_orders / n_days
    # Expected profit for the service
    expected_profit = np.sum(x_ij * W_j)
    # Risk (standard deviation)
    risk = np.sqrt(np.sum(W_j * (x_ij - expected_profit) ** 2))
    return expected_profit, risk


def calculate_package_statistics(prices, support_costs, orders, n_days):
    stats = {}
    for i, service in enumerate(orders):
        exp_profit, risk = compute_service_stats(prices[i], support_costs[i], orders[service], n_days)
        stats[service] = {'expected_profit': exp_profit, 'risk': risk}

    # Create a DataFrame from the statistics
    df_stats = pd.DataFrame(stats).T

    # Create feature matrix for clustering
    X = df_stats[['expected_profit', 'risk']].values

    # 2. Apply the DBSCAN Clustering Algorithm
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=0.8, min_samples=1)
    clusters = dbscan.fit_predict(X_scaled)
    df_stats['cluster'] = clusters

    # Convert individual service stats to dictionary for JSON output
    service_stats = df_stats.to_dict(orient="index")

    # 3. Calculate Kendall Rank Correlation Coefficient per cluster
    kendall_results = {}
    for cluster in np.unique(clusters):
        cluster_data = df_stats[df_stats['cluster'] == cluster]
        if len(cluster_data) > 1:
            tau, p_value = kendalltau(cluster_data['expected_profit'], cluster_data['risk'])
            kendall_results[str(cluster)] = {"kendall_tau": tau, "p_value": p_value}
        else:
            kendall_results[str(cluster)] = {"message": "Only one service, no correlation computed."}

    # 4. Validity of clustering based on correlation (simplified decision)
    validity = "Needs review"
    for vals in kendall_results.values():
        if "kendall_tau" in vals and abs(vals["kendall_tau"]) > 0.5 and vals["p_value"] < 0.05:
            validity = "Valid for forming a service package"
            break

    # 5. Overall cost assessment of clusters
    cluster_assessment_df = df_stats.groupby('cluster').agg(
        total_expected_profit=('expected_profit', 'sum'),
        total_risk=('risk', 'sum'),
        n_services=('expected_profit', 'count')
    )
    cluster_assessment_df['profitability_index'] = cluster_assessment_df['total_expected_profit'] / \
                                                   cluster_assessment_df['total_risk']
    cluster_assessment = cluster_assessment_df.reset_index().to_dict(orient="records")

    # Create a scatter plot of the clusters and encode it as base64
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="Set1", s=100)
    plt.xlabel("Scaled Expected Profit")
    plt.ylabel("Scaled Risk")
    plt.title("DBSCAN Clustering of Services")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Construct the result
    result = {
        "service_stats": service_stats,
        "kendall_results": kendall_results,
        "validity": validity,
        "cluster_assessment": cluster_assessment,
        "plot": plot_base64
    }
    return result
