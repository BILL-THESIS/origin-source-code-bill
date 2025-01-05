import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


if __name__ == "__main__":
    # File paths
    input_filepath = "../../output/seatunnel_compare.pkl"
    data = pd.read_pickle(input_filepath)

    prefix = "java:"
    suffix = "_created"

    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols]
    selected_cols.fillna(0, inplace=True)

    # Drop non-numeric and unnecessary columns
    data_cleaned = selected_cols.select_dtypes(include=['number'])

    # Check the shape of the cleaned data to confirm columns removed
    data_cleaned.shape

    # คำนวณ Spearman's rank correlation matrix และ p-value
    correlation_matrix = data_cleaned.corr(method='spearman')
    correlation_matrix.fillna(0, inplace=True)

    corr_matrix, p_value = spearmanr(data_cleaned, axis=0)
    corr_matrix = np.nan_to_num(corr_matrix)
    p_value = np.nan_to_num(p_value)

    # Perform hierarchical clustering
    distance_matrix = 1 - correlation_matrix.abs()
    linkage_matrix = linkage(distance_matrix, method='average')

    # สร้าง dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=correlation_matrix.columns, leaf_rotation=90)
    plt.title('Variable Clustering Dendrogram')
    plt.xlabel('Variables')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    cluster_agg = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
    cluster_agg.fit(1 - np.abs(corr_matrix))

    cluster_fc = fcluster(linkage_matrix, t=0.7, criterion='distance')

    df_clusters_agg = pd.DataFrame(cluster_agg.labels_, index=data_cleaned.columns, columns=['Cluster'])
    df_clusters_fc = pd.DataFrame(cluster_fc, index=correlation_matrix.columns, columns=['Cluster'])
