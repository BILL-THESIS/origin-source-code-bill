import os
import joblib
import pandas as pd
import numpy as np


def chenge_time_to_hours(df, time01, time12):
    # Change the time to hours
    df['time01'] = df[time01].dt.total_seconds() / 3600
    df['time12'] = df[time12].dt.total_seconds() / 3600

    return df


def calculate_std_interval_time(df, time01, time12, min_time, max_time):
    df['interval1'] = df[time01] - min_time
    df['interval2'] = df[time12] - df[time01]
    df['interval3'] = max_time - df[time12]

    df['time_diff'] = df[time12] - df[time01]

    df['mean_interval'] = df[['interval1', 'interval2', 'interval3']].mean(axis=1)
    df['std_interval'] = df[['interval1', 'interval2', 'interval3']].std(axis=1)
    return df


def calculate_time_counts_std(data):
    data['std_counts'] = data[['time0', 'time1', 'time2']].std(axis=1)
    return data


def process_time_interval(data_split, time_col1, time_col2, prepare_data, data_counts, project_name):
    min_time = prepare_data['total_time_hours'].min()
    max_time = prepare_data['total_time_hours'].max()
    data_split = chenge_time_to_hours(data_split, time_col1, time_col2)
    counts_time = calculate_time_counts_std(data_counts)
    data_std = calculate_std_interval_time(data_split, time_col1, time_col2, min_time, max_time)
    data_mrege = pd.merge(data_std, counts_time, on=['index_time01', 'index_time12'])
    data_mrege.drop(columns=['accuracy', 'f1_macro',
                             'precision_macro',
                             'recall_macro', 'time01_y', 'precision_smote',
                             'recall_smote', 'roc_auc_smote',
                             'y_original',
                             'y_resample',
                             'y_train',
                             'y_train_resample', 'time01_y',
                             'time12_y', ], inplace=True)

    data_mrege.to_parquet(f'../output/{project_name}_result_std_class_time3.parquet')

    return data_mrege


def sort_f1_60_percent(df, project_name):
    mask = (df['f1_smote'] >= 0.6) & (df['f1_smote'] < 0.7)
    # Invert the mask to keep rows that do not meet the condition
    df_filtered_point = df[mask]

    return df_filtered_point


def sort_f1_70_percent(df, project_name):
    mask = (df['f1_smote'] >= 0.7) & (df['f1_smote'] < 0.8)
    # Invert the mask to keep rows that do not meet the condition
    df_filtered_point = df[mask]

    return df_filtered_point


def sort_f1_80_percent(df, project_name):
    mask = (df['f1_smote'] >= 0.8) & (df['f1_smote'] < 0.9)

    # Invert the mask to keep rows that do not meet the condition
    df_filtered_point = df[mask]

    return df_filtered_point


if __name__ == '__main__':
    ozone_prepare = pd.read_parquet("../output/ozone_prepare_to_train.parquet")
    ozone_counts = pd.read_parquet("../output/ozone_counts_time_class.parquet")

    with open(os.path.join('../output/ozone_split_f1_smote_time_class3.pkl'), 'rb') as f:
        ozone = joblib.load(f)

    pulsar_prepare = pd.read_parquet("../output/pulsar_prepare_to_train.parquet")
    pulsar_counts = pd.read_parquet("../output/pulsar_counts_time_class.parquet")

    with open(os.path.join('../output/pulsar_split_f1_smote_time_class3.pkl'), 'rb') as f:
        pulsar = joblib.load(f)

    seatunnel_prepare = pd.read_parquet("../output/seatunnel_prepare_to_train.parquet")
    seatunnel_counts = pd.read_parquet("../output/seatunnel_counts_time_class.parquet")

    with open(os.path.join('../output/seatunnel_split_f1_smote_time_class3.pkl'), 'rb') as f:
        seatunnel = joblib.load(f)

    pulsar_std = process_time_interval(pulsar, 'time01', 'time12', ozone_prepare, ozone_counts, 'pulsar')
    ozone_std = process_time_interval(ozone, 'time01', 'time12', pulsar_prepare, pulsar_counts, 'ozone')
    seatunnel_std = process_time_interval(seatunnel, 'time01', 'time12', seatunnel_prepare, seatunnel_counts,
                                          'seatunnel')

    seatunnel_60_percent = sort_f1_60_percent(seatunnel_std, 'seatunnel')
    seatunnel_70_percent = sort_f1_70_percent(seatunnel_std, 'seatunnel')
    seatunnel_80_percent = sort_f1_80_percent(seatunnel_std, 'seatunnel')

    ozone_60_percent = sort_f1_60_percent(ozone_std, 'ozone')
    ozone_70_percent = sort_f1_70_percent(ozone_std, 'ozone')
    ozone_80_percent = sort_f1_80_percent(ozone_std, 'ozone')

    pulsar_60_percent = sort_f1_60_percent(pulsar_std, 'pulsar')
    pulsar_70_percent = sort_f1_70_percent(pulsar_std, 'pulsar')
    pulsar_80_percent = sort_f1_80_percent(pulsar_std, 'pulsar')

    ozone_70_percent_fiter = ozone_70_percent[
        (ozone_70_percent['std_counts'] < 100) & (ozone_70_percent['f1_smote'] < 0.75)]
    pulsar_70_percent_fiter = pulsar_70_percent[
        (pulsar_70_percent['std_counts'] < 140) & (pulsar_70_percent['f1_smote'] < 0.75)]
    seatunnel_70_percent_fiter = seatunnel_70_percent[
        (seatunnel_70_percent['std_counts'] < 120) & (seatunnel_70_percent['f1_smote'] < 0.75)]

    ozone_sort = ozone_70_percent_fiter.sort_values(by='f1_smote', ascending=False).head(10)
    pulsar_sort = pulsar_70_percent_fiter.sort_values(by='f1_smote', ascending=False).head(10)
    seatunnel_sort = seatunnel_70_percent_fiter.sort_values(by='f1_smote', ascending=False).head(10)

    ozone_sort.drop(columns=['accuracy_smote', 'interval1', 'interval2', 'interval3',
                             'time_diff', 'mean_interval', 'std_interval'], inplace=True)
    ozone_sort.reset_index(drop=True, inplace=True)

    pulsar_sort.drop(columns=['accuracy_smote', 'interval1', 'interval2', 'interval3',
                              'time_diff', 'mean_interval', 'std_interval'], inplace=True)
    pulsar_sort.reset_index(drop=True, inplace=True)

    seatunnel_sort.drop(columns=['accuracy_smote', 'interval1', 'interval2', 'interval3',
                                 'time_diff', 'mean_interval', 'std_interval'], inplace=True)
    seatunnel_sort.reset_index(drop=True, inplace=True)

    df = pd.concat([ozone_sort, pulsar_sort, seatunnel_sort], axis=1)
