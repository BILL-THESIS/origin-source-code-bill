import numpy as np
import pandas as pd
import itertools

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# step 1
# input: Dataframe X = Original 20 columns
# output: Combination of DataFrame columns
class KMeansCluster:

    def __init__(self, df):
        self.df = df

    # functions to divide the combinations of columns in the dataframe X into 8 parts
    def chunkify(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def all_sub_combinations(self):
        return KMeansCluster.chunkify(all_combinations_divide, 8)

    # step 2
    # input: Combination of DataFrame 20 columns to 1 million data sets
    #  output:  MinMax scaler

    def scaler(df):
        scal = MinMaxScaler()
        scaled = scal.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)
        return scaled_df

    # step 3
    # input: data frame Minmax scaler
    # output:  Kmeans cluster (score cluster, labels cluster, number cluster)

    def kmeans_cluster(scaled_df):

        kmeans_list = []

        for n in range(2, 5):

            kmeans = KMeans(n_clusters=n, n_init=10)
            kmeans.fit_transform(scaled_df)

            score = silhouette_score(scaled_df, kmeans.labels_)
            kmeans_list.append([score, kmeans.labels_, kmeans.n_clusters])
        return kmeans_list


if __name__ == '__main__':
    # prepare the data frame
    df_result = []

    df_X = pd.read_parquet('seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
    col_names = df_X.columns
    # list all combinations of columns in the dataframe X
    all_combinations_list_col = [list(itertools.combinations(col_names, r)) for r in range(2, len(col_names))]

    # amout of columns in the dataframe X
    all_combianations_col = [itertools.combinations(col_names, r) for r in range(1, len(col_names))]

    # all amout of combinations of columns in the dataframe X 1048554
    all_combinations = list(itertools.chain(*all_combinations_list_col))
    all_combinations_divide = [a for a in all_combinations if len(a) > 0]

    # step 1
    df_combi_chunk_list = KMeansCluster.chunkify(all_combinations_divide, 8)
    sub_combi = KMeansCluster.all_sub_combinations(df_combi_chunk_list)

    # step 2
    scalers = [KMeansCluster.scaler(df_X[list(i)]) for i in sub_combi[0]]

    # step 3
    kmeans = [KMeansCluster.kmeans_cluster(x) for x in scalers]

    # step 4 merge the results
    # create the new data frame with the results
    df_result = [KMeansCluster.all_sub_combinations(df_X[list(i)]) for i in sub_combi[0]]
    # df_result = pd.concat([df_result, df_kamens], axis=1)
    # df_result = pd.concat([df_result, pd.DataFrame(df_original, columns=['total_time'])], axis=1)
    # df_kamens_conacat = pd.concat([df_kamens, pd.DataFrame(sub_combi[0])], axis=1)
    # df_kamens_conacat_time = pd.concat([df_kamens_conacat, pd.DataFrame(df_original, columns=['total_time'])], axis=1)
