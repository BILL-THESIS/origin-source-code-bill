from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\scaled'
# directory_path_cluster = '/models/KMeans/cluster3'


pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

results_0 = []
results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []
results_6 = []
results_7 = []
results_8 = []
results_9 = []

cluster_list = []
max_list = []
min_list = []
arg_list = []

for pkl_file in pkl_files:
    file_path = os.path.join(directory_path, pkl_file)
    # print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_file)[0]
    # print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    # print("DF ::" , df_col_combined)
    print('\n')

    df_last_col = df_col_combined.iloc[:,-3].unique()
    # print("Last Columns ::" , df_last_col)
    print("\n")

    col_srt = df_col_combined.columns[-3]

    for cluster_value in df_last_col :
        print("I :::", cluster_value)
        cluster_data = df_col_combined[df_col_combined[col_srt] == cluster_value]

        cluster_list.append(cluster_data)
        cluster_data_df = pd.DataFrame(cluster_data)

        min = cluster_data_df['total_time'].min()
        min_list.append(min)
        arg = cluster_data_df['total_time'].mean()
        arg_list.append(arg)
        max = cluster_data_df['total_time'].max()
        max_list.append(max)
        shape = cluster_data_df.shape

        print(f'{col_srt}')
        print(f"Cluster {cluster_value}:")
        print("Minimum timedelta:", min)
        print("Average timedelta:", arg)
        print("Maximum timedelta:", max)
        print("Cluster shape:", shape)
        print("\n")

        if cluster_value == 0 :
            results_dict_0 = {
                'name_col' : col_srt ,
                'scores' : cluster_data['scored'].values[1],
                'cluster' : cluster_data['clusters'].values[1],
                f'Minimum{cluster_value}' : min ,
                f'Maximum{cluster_value}': max,
                f'Average{cluster_value}': cluster_data['total_time'].mean()
            }
        if cluster_value == 1 :
            results_dict_1 = {
                'name_col' : col_srt ,
                'scores' : cluster_data['scored'].values[1],
                'cluster' : cluster_data['clusters'].values[1],
                f'Minimum{cluster_value}' : cluster_data['total_time'].min() ,
                f'Maximum{cluster_value}': cluster_data['total_time'].max(),
                f'Average{cluster_value}': cluster_data['total_time'].mean()
            }
        if cluster_value == 2:
            results_dict_2 = {
                'name_col' : col_srt ,
                'scores' : cluster_data['scored'].values[1],
                'cluster' : cluster_data['clusters'].values[1],
                f'Minimum{cluster_value}': cluster_data['total_time'].min(),
                f'Maximum{cluster_value}': cluster_data['total_time'].max(),
                f'Average{cluster_value}': cluster_data['total_time'].mean()
            }
        if cluster_value == 3 :
            results_dict_3 = {
                'name_col' : col_srt ,
                'scores' : cluster_data['scored'].values[1],
                'cluster' : cluster_data['clusters'].values[1],
                f'Minimum{cluster_value}' : cluster_data['total_time'].min() ,
                f'Maximum{cluster_value}': cluster_data['total_time'].max(),
                f'Average{cluster_value}': cluster_data['total_time'].mean()
            }
        if cluster_value == 4 :
            results_dict_4 = {
                'name_col' : col_srt ,
                'scores' : cluster_data['scored'].values[1],
                'cluster' : cluster_data['clusters'].values[1],
                f'Minimum{cluster_value}' : cluster_data['total_time'].min() ,
                f'Maximum{cluster_value}': cluster_data['total_time'].max(),
                f'Average{cluster_value}': cluster_data['total_time'].mean()
            }
        # if cluster_value == 5 :
        #     results_dict_5 = {
        #         'name_col' : col_srt ,
        #         'scores' : cluster_data['scored'].values[1],
        #         'cluster' : cluster_data['clusters'].values[1],
        #         f'Minimum{cluster_value}' : cluster_data['total_time'].min() ,
        #         f'Maximum{cluster_value}': cluster_data['total_time'].max(),
        #         f'Average{cluster_value}': cluster_data['total_time'].mean()
        #     }
        # if cluster_value == 6:
        #     results_dict_6 = {
        #         'name_col' : col_srt ,
        #         'scores' : cluster_data['scored'].values[1],
        #         'cluster' : cluster_data['clusters'].values[1],
        #         f'Minimum{cluster_value}': cluster_data['total_time'].min() ,
        #         f'Maximum{cluster_value}': cluster_data['total_time'].max(),
        #         f'Average{cluster_value}': cluster_data['total_time'].mean()
        #     }


    results_0.append(results_dict_0)
    results_1.append(results_dict_1)
    results_2.append(results_dict_2)
    results_3.append(results_dict_3)
    results_4.append(results_dict_4)
    # results_5.append(results_dict_5)
    # results_6.append(results_dict_6)
    # results_7.append(results_dict_7)
    # results_8.append(results_dict_8)
    # results_9.append(results_dict_9)

    results_df_0 = pd.DataFrame(results_0)
    results_df_0 = results_df_0.drop_duplicates()
    results_df_1 = pd.DataFrame(results_1)
    results_df_1 = results_df_1.drop_duplicates()
    results_df_2 = pd.DataFrame(results_2)
    results_df_2 = results_df_2.drop_duplicates()
    results_df_3 = pd.DataFrame(results_3)
    results_df_3 = results_df_3.drop_duplicates()
    # results_df_3.to_csv("results_df_3.csv")
    results_df_4 = pd.DataFrame(results_4)
    results_df_4 = results_df_4.drop_duplicates()
    # results_df_4.to_csv("results_df_4.csv")
    # results_df_5 = pd.DataFrame(results_5)
    # results_df_5 =results_df_5.drop_duplicates()
    # # results_df_5.to_csv("results_df_5.csv")
    # results_df_6 = pd.DataFrame(results_6)
    # results_df_6 = results_df_6.drop_duplicates()
    # # results_df_6.to_csv("results_df_6.csv")
    # results_df_7 = pd.DataFrame(results_7)
    # results_df_7 = results_df_7.drop_duplicates()
    # # results_df_7.to_csv("results_df_7.csv")
    # results_df_8 = pd.DataFrame(results_8)
    # results_df_8 = results_df_8.drop_duplicates()
    # # results_df_8.to_csv("results_df_8.csv")
    # results_df_9 = pd.DataFrame(results_9)
    # results_df_9 = results_df_9.drop_duplicates()
    # results_df_9.to_csv("results_df_9.csv")

    results_df_0 = pd.DataFrame(results_0).set_index(['name_col', 'scores', 'cluster'])
    results_df_1 = pd.DataFrame(results_1).set_index(['name_col', 'scores', 'cluster'])
    results_df_2 = pd.DataFrame(results_2).set_index(['name_col', 'scores', 'cluster'])
    results_df_3 = pd.DataFrame(results_3).set_index(['name_col', 'scores', 'cluster'])
    results_df_4 = pd.DataFrame(results_4).set_index(['name_col', 'scores', 'cluster'])
    # results_df_5 = pd.DataFrame(results_5).set_index(['name_col', 'scores', 'cluster'])
    # results_df_6 = pd.DataFrame(results_6).set_index(['name_col', 'scores', 'cluster'])
    # results_df_7 = pd.DataFrame(results_7).set_index(['name_col', 'scores', 'cluster'])
    # results_df_8 = pd.DataFrame(results_8).set_index(['name_col', 'scores', 'cluster'])
    # results_df_9 = pd.DataFrame(results_9).set_index(['name_col', 'scores', 'cluster'])

    result3 = pd.merge(results_df_0, results_df_1, on=['name_col', 'scores', 'cluster'])
    result3 = result3.drop_duplicates()
    result4 = pd.merge(result3, results_df_2, how="left", on=['name_col', 'scores', 'cluster'])
    result4 = result4.drop_duplicates()
    # result5 = pd.merge(result4, results_df_3, how="left", on=['name_col', 'scores', 'cluster'])
    # result5 = result5.drop_duplicates()
    # result6 = pd.merge(result5, results_df_4, how="left", on=['name_col', 'scores', 'cluster'])
    # result6 = result6.drop_duplicates()
    result4.to_csv("all_data_scores_clusters.csv")
    # result7 = pd.merge(result6, results_df_5, how="left", on=['name_col', 'scores', 'cluster'])
    # result8 = pd.merge(result7, results_df_6, how="left", on=['name_col', 'scores', 'cluster'])
    # result9 = pd.merge(result8, results_df_7, how="left", on=['name_col', 'scores', 'cluster'])
    # result10 = pd.merge(result9, results_df_8, how="left", on=['name_col', 'scores', 'cluster'])
    # result11 = pd.merge(result10, results_df_9, how="left", on=['name_col', 'scores', 'cluster'])
    # result11 = result11.drop_duplicates()
    # result11.to_csv(f"all_data.csv")