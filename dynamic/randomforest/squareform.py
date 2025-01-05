import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sqlalchemy.dialects.mssql.information_schema import columns
from statsmodels.sandbox.panel.correlation_structures import corr_ar


def get_correlation_matrix(df, prefix="java:", suffix="_created"):
    # Extract correlation matrix for columns matching the given prefix and suffix
    selected_cols = [col for col in df.columns if col.startswith(prefix) and col.endswith(suffix)]
    if not selected_cols:
        raise ValueError("No columns match the specified prefix and suffix.")
    return df[selected_cols].corr(method='spearman')


def compute_clusters(corr_matrix, threshold=0.7, method="ward"):
    # Perform hierarchical clustering and return the cluster assignments
    corr = 1 - corr_matrix.abs()
    corr.fillna(0, inplace=True)
    squre = squareform(corr.values, checks=False)
    linkage_matrix = linkage(squre, method=method)
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    return corr, clusters, linkage_matrix


def plot_dendrogram(sub_corr_matrix, cluster_id, output_path):
    # Plot and save the dendrogram for a given cluster
    sub_distance_matrix = 1 - sub_corr_matrix.abs()
    sub_distance_matrix.fillna(0, inplace=True)
    sub_condensed_distance = squareform(sub_distance_matrix.values, checks=False)
    sub_linkage_matrix = linkage(sub_condensed_distance, method='ward')

    plt.figure(figsize=(8, 6))
    dendrogram(
        sub_linkage_matrix,
        labels=sub_corr_matrix.columns,
        leaf_rotation=90,
        leaf_font_size=10,
    )
    plt.title(f"Dendrogram for Cluster {cluster_id}")
    plt.xlabel("Variables")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def select_representative_features(corr_matrix, clusters):

    # Select the most representative feature from each cluster.
    # The representative feature is the one with the highest average correlation
    # with other variables in the same cluster.

    representative_features = []
    for cluster_id in set(clusters):
        cluster_vars = corr_matrix.columns[clusters == cluster_id]
        if len(cluster_vars) > 1:
            sub_corr_matrix = corr_matrix.loc[cluster_vars, cluster_vars]
            avg_corr = sub_corr_matrix.mean(axis=1)
            # Select feature with max average correlation
            representative_feature = avg_corr.idxmax()
            representative_features.append(representative_feature)
        elif len(cluster_vars) == 1:
            representative_features.append(cluster_vars[0])
    return representative_features


if __name__ == "__main__":
    # File paths
    input_filepath = "../output/seatunnel_compare.pkl"

    # Load data
    df_original = pd.read_pickle(input_filepath)

    # Compute correlation matrix
    corr_matrix = get_correlation_matrix(df_original)

    # Perform clustering
    # clusters, linkage_matrix = compute_clusters(corr_matrix)
    crr, clusters, linkage_matrix = compute_clusters(corr_matrix)


    # Number of clusters
    num_clusters = len(set(clusters))
    print(f"Number of clusters: {num_clusters}")

    # Select representative features from clusters
    representative_features = select_representative_features(corr_matrix, clusters)
    print(f"Selected representative features: {representative_features}")

    # Reduce redundancy in DataFrame
    df_reduced = df_original[representative_features]

    # Plot dendrograms for each cluster
    for cluster_id in range(1, num_clusters + 1):
        cluster_vars = corr_matrix.columns[clusters == cluster_id]

        if len(cluster_vars) > 1:
            print(f"Cluster {cluster_id}: {list(cluster_vars)}")
            sub_corr_matrix = corr_matrix.loc[cluster_vars, cluster_vars]
            output_path = f"../output/seatunnel_cluster_{cluster_id}_squareform.png"
            plot_dendrogram(sub_corr_matrix, cluster_id, output_path)
