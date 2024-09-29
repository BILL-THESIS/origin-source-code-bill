import time
from collections import Counter
from datetime import date

import pandas as pd
import numpy as np
import itertools

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
import joblib
from dask.distributed import Client, LocalCluster


def percentage_smell(df):
    df = df.rename(columns={'created_Dispensables': 'created_D',
                            'created_Bloaters': 'created_B',
                            'created_Change Preventers': 'created_CP',
                            'created_Couplers': 'created_C',
                            'created_Object-Orientation Abusers': 'created_OOA',
                            'ended_Dispensables': 'ended_D',
                            'ended_Bloaters': 'ended_B',
                            'ended_Change Preventers': 'ended_CP',
                            'ended_Couplers': 'ended_C',
                            'ended_Object-Orientation Abusers': 'ended_OOA'})

    df['created_D'].astype(float)
    df['percentage_d'] = ((df['ended_D'] - df['created_D'].astype(float)) / df['created_D'].astype(float)) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100
    return df


def calculate_percentiles(date_series):
    percentiles = [np.percentile(date_series, range(0, 101))]
    percentile_df = pd.DataFrame(percentiles).T
    percentile_df.values.flatten()
    return percentile_df


def set_index_combinations_percentiles(percentile_list):
    list_each_percentile = []
    for points in percentile_list:
        df = pd.DataFrame(points)
        list_each_percentile.append(df)
    return list_each_percentile


def table_time_fix_percentile(list_each_percentile):
    time01_list = []
    time12_list = []
    index_time_01_list = []
    index_time_12_list = []

    for points in list_each_percentile:
        sort_point = points.sort_values(by=[0], ascending=True)

        time_01 = sort_point.iloc[0][0]
        # print("time_01", time_01)
        time_12 = sort_point.iloc[1][0]
        # print("time_12", time_12)

        index_time01 = sort_point.index[0]
        # print("index_time01", index_time01)
        index_time12 = sort_point.index[1]
        # print("index_time12", index_time12)

        time01_list.append(time_01)
        time12_list.append(time_12)

        index_time_01_list.append(index_time01)
        index_time_12_list.append(index_time12)

    df_time_point = pd.DataFrame(
        {'index_time01': index_time_01_list, 'index_time12': index_time_12_list, 'time01': time01_list,
         'time12': time12_list})
    return df_time_point


def divide_time_class(df_original, df_time_point):
    results = []
    for index, row in df_time_point.iterrows():
        # Create a copy of the DataFrame for the current percentile
        time01 = row['time01']
        time12 = row['time12']
        time_fix_hours = df_original['total_time']

        values_time = []
        for time_i in time_fix_hours:
            if time_i <= time01:
                values_time.append(0)
                # print(f"time modify :: {time_i} < time01 :: {time01}")
            elif (time_i > time01) & (time_i < time12):
                values_time.append(1)
                # print(f"time01 :: {time01} >= time modify :: {time_i} < time12 :: {time12}")
            else:  # time_i >= time12
                values_time.append(2)
                # print(f"time modify :: {time_i} >=  time12 :: {time12}")

        # Create the 'time_class' column directly during iteration
        df_original['time_class'] = values_time
        df_original['index_time01'] = row['index_time01']
        df_original['time_01'] = row['time01']
        df_original['index_time12'] = row['index_time12']
        df_original['time_12'] = row['time12']
        results.append(df_original.copy())
    return results


def count_time_class(df):
    values_time = []
    for col in df:
        time_class = col['time_class']
        col['time0'] = (time_class == 0).sum()
        col['time1'] = (time_class == 1).sum()
        col['time2'] = (time_class == 2).sum()
        values_time.append(df.copy())
    return df


def set_time_class(df):
    index_t1, index_t2, time_01, time_12, t_0, t_1, t_2 = [], [], [], [], [], [], []
    for col in df:
        index_time01 = col['index_time01'].iloc[0]
        index_time12 = col['index_time12'].iloc[0]
        time01 = col['time_01'].iloc[0]
        time12 = col['time_12'].iloc[0]
        t0 = col['time0'].iloc[0]
        t1 = col['time1'].iloc[0]
        t2 = col['time2'].iloc[0]
        index_t1.append(index_time01)
        index_t2.append(index_time12)
        time_01.append(time01)
        time_12.append(time12)
        t_0.append(t0)
        t_1.append(t1)
        t_2.append(t2)

    return index_t1, index_t2, time_01, time_12, t_0, t_1, t_2


def prepare_to_data_frame(df_original, project_name):
    data_percentiles = calculate_percentiles(df_original['total_time'])
    data_time_point_list = list(itertools.combinations(data_percentiles.iloc, 2))
    data_percentiles_list = set_index_combinations_percentiles(data_time_point_list)
    data_table_time = table_time_fix_percentile(data_percentiles_list)
    data_each_case_time = divide_time_class(df_original, data_table_time)
    data_time_class_count = count_time_class(data_each_case_time)
    index_t1, index_t2, time_01, time_12, t_0, t_1, t_2 = set_time_class(data_time_class_count)

    data_time_class = pd.DataFrame({
        'index_time01': index_t1,
        'index_time12': index_t2,
        'time01': time_01,
        'time12': time_12,
        'time0': t_0,
        'time1': t_1,
        'time2': t_2
    })

    data_time_class.to_parquet(f'../output/{project_name}_counts_time_class_newversion9Sep.parquet')

    return data_time_class


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    pulsar_original_rename = pd.read_parquet('../output/pulsar_prepare_to_train.parquet')
    pulsar_original_rename = percentage_smell(pulsar_original_rename)

    ozone_original_rename = pd.read_parquet('../output/ozone_prepare_to_train_newversion_9Sep.parquet')
    ozone_original_rename = percentage_smell(ozone_original_rename)

    seatunnel_original_rename = pd.read_parquet('../output/seatunnel_prepare_to_train.parquet')
    seatunnel_original_rename = percentage_smell(seatunnel_original_rename)
    #
    ozone_time_class = prepare_to_data_frame(ozone_original_rename, 'ozone')
    # pulsar_time_class = prepare_to_data_frame(pulsar_original_rename, 'pulsar')
    # seatunnel_time_class = prepare_to_data_frame(seatunnel_original_rename, 'seatunnel')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
