from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os

directory_path = 'D:\origin-source-code-bill\models\KMeans\group_cluster'
directory_path_time = 'D:\origin-source-code-bill\models\KMeans\group_time'
df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
labels = pd.read_pickle('lable/labels_final.pkl')
group_lables = pd.read_pickle('lable/List_df_labels.pkl')

pkl_files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
print(pkl_files)

for pkl_file in pkl_files:
    file_path = os.path.join(directory_path, pkl_file)
    # print("file :::" , file_path)
    variable_name = os.path.splitext(pkl_file)[0]
    print("Var ::" , variable_name)
    df_col_combined = pd.read_pickle(file_path)
    print("DF ::" , df_col_combined)
    print('\n')

    # df_last_col = df_col_combined.iloc[:,-1]
    df_last_col = df_col_combined.columns[-1]
    print("Last Columns ::" , df_last_col)
    print("\n")

    if df_col_combined.columns[0] in df_col_combined:
        merged_df = pd.merge(df_original , df_col_combined, on=df_col_combined.columns[0])
        print("Merged DF :::" , merged_df)
        # merged_df.to_pickle(f'{directory_path_time}/{df_col_combined.columns[-1]}.pkl')



    # for i, row in group_lables.iterrows():
    #     print("I :::", i)
    #     print("row labels ::" , row[1])
    #     print("Columns :::" , group_lables.columns[1])
    #     # df_merg_labels = pd.merge( df_col_combined , row[1] , left_on=df_col_combined.iloc[:,-1] , right_on=group_lables.columns[0] )
    #     print('\n')
    #     print("Cancat DF :::: " , df_merg_labels)