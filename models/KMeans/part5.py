import pandas as pd
import os
from itertools import chain, combinations, permutations

df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')
d1 = df_original.iloc[:, [5,9,10,11,12,13]]
labels = pd.read_pickle('lable/labels_final.pkl')
group_lables = pd.read_pickle('lable/labels_group.pkl')
directory_path = 'D:\origin-source-code-bill\models\KMeans\combia'

columns = d1.columns
list_com = []
result_dfs = []

for r in range(1, len(columns) + 1):
    for column_combination in combinations(columns , r):
        combined_df = d1[list(column_combination)]
        list_com.append(combined_df)
        print(list_com)


for col in group_lables.columns:
    per_list = []
    col_perfix = col[:-2]
    per_list.append(col_perfix)
    # print(col_perfix)

    for i in list_com:
        df_list = pd.DataFrame(i)
        # print(df_list)
        df_list_col = df_list.columns

    for i in per_list:
        if i in df_list_col:
            print("ok")
            concat_df = pd.concat([df_list, group_lables[i]] , axis=1)
            print("+++++++++++++++")
            print(concat_df)
            result_dfs.append(concat_df)
