import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


if __name__ == "__main__":
    # File paths
    input_filepath = "../../output/pulsar_compare.pkl"
    data = pd.read_pickle(input_filepath)

    prefix = "java:"
    suffix = "_created"

    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].fillna(0)

    # Drop non-numeric and unnecessary columns
    data_cleaned = selected_cols.select_dtypes(include=['number'])

    # Spearman's rank correlation matrix and p-value
    corr_matrix, corr_matrix_2 = spearmanr(data_cleaned)
    corr_matrix = pd.DataFrame(corr_matrix, index=data_cleaned.columns, columns=data_cleaned.columns)
    corr_matrix_2 = pd.DataFrame(corr_matrix_2, index=data_cleaned.columns, columns=data_cleaned.columns)
    corr_matrix_arr = np.nan_to_num(corr_matrix)
    corr_matrix_2_arr = np.nan_to_num(corr_matrix_2)

    # Calculate p-value


    # Perform hierarchical clustering
    distance_matrix = 1 - np.abs(corr_matrix_arr)
    df_distance_matrix = pd.DataFrame(1 - np.abs(corr_matrix_arr), index=corr_matrix.index, columns=corr_matrix.columns)

    linkage_matrix = linkage(df_distance_matrix, method='average')
    print(f"linkage_matrix shape: {linkage_matrix.shape}")
    print(f"df_distance_matrix shape: {df_distance_matrix.shape}")

    df_linkage = pd.DataFrame(linkage_matrix, columns=['clust1', 'clust2', 'distance', 'sample_count'])

    # Create a dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=data_cleaned.columns, leaf_rotation=90)
    plt.title('Variable Clustering Dendrogram')
    plt.xlabel('Variables')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig("../../output/variable_clustering_dendrogram.png")
    plt.show()



    # Agglomerative clustering
    cluster_agg = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
    cluster_agg.fit(1 - np.abs(corr_matrix_arr))

    cluster_fc = fcluster(linkage_matrix, t=0.7, criterion='distance')

    df_clusters_agg = pd.DataFrame(cluster_agg.labels_, index=data_cleaned.columns, columns=['Cluster'])
    df_clusters_fc = pd.DataFrame(cluster_fc, index=data_cleaned.columns, columns=['Cluster'])

