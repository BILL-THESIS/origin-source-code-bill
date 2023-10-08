from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import chain, combinations, permutations
import numpy as np
import os


directory_path_group = 'D:\origin-source-code-bill\models\KMeans\group_cluster'

# labels = pd.read_pickle('lable/new_labels.pkl')
labels = pd.read_pickle('lable/List_df_labels.pkl')

for i , row in labels.iterrows():

    print("I :::::::::::" , i)
    print('\n')

    print("ROW 1 " , row[0])
    print('\n')

    print("ROW 2 " , row[1])
    print('\n')

    print("ROW 3 " , row[2])
    print('\n')

    print("ROW 4 " , row[3])
    print('\n')

    df1 = pd.concat([row[0] ,row[1]] , axis=1)
    print("==========================")
    df1 =  df1.rename(columns={0 : f'{row[0].columns.to_list()}_{row[3]}'})
    print(df1)
    print('\n')
    # df1.to_pickle(f'{directory_path_group}/{row[0].columns.to_list()}_{row[3]}.pkl')



