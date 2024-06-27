import time
from collections import Counter
from datetime import date

import pandas as pd
import numpy as np
import itertools

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV

import joblib
from dask.distributed import Client, LocalCluster


def percentage_smell(df):
    df = df.rename(columns={'begin_Dispensables': 'created_D',
                            'begin_Bloaters': 'created_B',
                            'begin_Change Preventers': 'created_CP',
                            'begin_Couplers': 'created_C',
                            'begin_Object-Orientation Abusers': 'created_OOA',
                            'end_Dispensables': 'ended_D',
                            'end_Bloaters': 'ended_B',
                            'end_Change Preventers': 'ended_CP',
                            'end_Couplers': 'ended_C',
                            'end_Object-Orientation Abusers': 'ended_OOA'})

    df['created_D'].astype(float)
    df['percentage_b'] = ((df['ended_D'] - df['created_D'].astype(float)) / df['created_D'].astype(float)) * 100
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


def divide_time_class(df_time_fix, df_time_point):
    results = []
    df_copy = df_time_fix.copy()
    time_fix_hours = df_copy['total_time'].dt.total_seconds() / 3600

    for index, row in df_time_point.iterrows():
        time01 = row['time01']
        # print("time01", time01)
        time12 = row['time12']
        # print("time12", time12)

        for index, row in time_fix_hours.iteritems():
            if row < time01:
                df_copy.loc['time_class'] = 0
            elif (row >= time01) & (row < time12):
                df_copy.loc['time_class'] = 1
            elif row >= time12:
                df_copy.loc['time_class'] = 2

        # df_copy['time_class'] = time_fix_hours.apply(lambda s: 0 if s < time01 else 1 if ((s >= time01) & (s < time12)) else 2 if s >= time12 else 3)
        # df_copy.loc[(time_fix_hours < time01), 'time_class'] = 0
        # df_copy.loc[(time_fix_hours >= time01) & (time_fix_hours < time12), 'time_class'] = 1
        # df_copy.loc[(time_fix_hours >= time12), 'time_class'] = 2

        results.append(df_copy)

    return results


# def split_data_x_y(df, df_time_point, random_state=3, test_size=0.3):
#     precision_macro_list = []
#     recall_macro_list = []
#     f1_macro_list = []
#
#     precision_micro_list = []
#     recall_micro_list = []
#     f1_micro_list = []
#
#     precision_macro_smote_list = []
#     recall_macro_smote_list = []
#     f1_macro_smote_list = []
#
#     precision_micro_smote_list = []
#     recall_micro_smote_list = []
#     f1_micro_smote_list = []
#
#     acc_normal_list = []
#     acc_smote_list = []
#
#     y_original_list = []
#     y_resampled_list = []
#     y_train_list = []
#     y_train_smote_list = []
#
#     df_copy = df.copy()
#     time_fix_hours = df_copy['total_time'].dt.total_seconds() / 3600
#
#     for index, row in df_time_point.iterrows():
#         time01 = row['time01']
#         time12 = row['time12']
#
#         df_copy['time_class'] = time_fix_hours.apply(
#             lambda s: 0 if s < time01 else 1 if ((s >= time01) & (s < time12)) else 2 if s >= time12 else 3)
#
#         X = df_copy[['created_D', 'created_B', 'created_CP', 'created_C', 'created_OOA',
#                      'ended_D', 'ended_B', 'ended_CP', 'ended_C', 'ended_OOA',
#                      'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
#         y = df_copy['time_class']
#
#         print('Original dataset shape %s' % Counter(y))
#
#         smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=2)
#         X_resampled, y_resampled = smote.fit_resample(X, y)
#         print('Resampled dataset shape %s', Counter(y_resampled))
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
#                                                             random_state=random_state)
#         print('y_train dataset shape %s', Counter(y_train))
#
#         X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled,
#                                                                                                  y_resampled,
#                                                                                                  test_size=test_size,
#                                                                                                  random_state=random_state)
#         print('y_train_resampled dataset shape %s', Counter(y_train_resampled))
#
#         GradientBoosting = GradientBoostingClassifier()
#         normal_fit = GradientBoosting
#         normal_fit.fit(X_train, y_train)
#         smote_fit = GradientBoosting
#         smote_fit.fit(X_train_resampled, y_train_resampled)
#
#         y_pred = cross_val_predict(normal_fit, X_train, y_train, cv=5)
#         acc_normal = accuracy_score(y_train, y_pred)
#
#         y_pred_smote = cross_val_predict(smote_fit, X_train_resampled, y_train_resampled, cv=5)
#         acc_smote = accuracy_score(y_train_resampled, y_pred_smote)
#
#         precision_macro_list.append(precision_score(y_train, y_pred, average='macro'))
#         recall_macro_list.append(recall_score(y_train, y_pred, average='macro'))
#         f1_macro_list.append(f1_score(y_train, y_pred, average='macro'))
#
#         precision_macro_smote_list.append(precision_score(y_train_resampled, y_pred_smote, average='macro'))
#         recall_macro_smote_list.append(recall_score(y_train_resampled, y_pred_smote, average='macro'))
#         f1_macro_smote_list.append(f1_score(y_train_resampled, y_pred_smote, average='macro'))
#
#         precision_micro_list.append(precision_score(y_train, y_pred, average='micro'))
#         recall_micro_list.append(recall_score(y_train, y_pred, average='micro'))
#         f1_micro_list.append(f1_score(y_train, y_pred, average='micro'))
#
#         precision_micro_smote_list.append(precision_score(y_train_resampled, y_pred_smote, average='micro'))
#         recall_micro_smote_list.append(recall_score(y_train_resampled, y_pred_smote, average='micro'))
#         f1_micro_smote_list.append(f1_score(y_train_resampled, y_pred_smote, average='micro'))
#
#         acc_smote_list.append(acc_smote)
#         acc_normal_list.append(acc_normal)
#
#         y_original_list.append(Counter(y))
#         y_resampled_list.append(Counter(y_resampled))
#         y_train_list.append(Counter(y_train))
#         y_train_smote_list.append(Counter(y_train_resampled))
#
#     return (precision_macro_list, recall_macro_list, f1_macro_list,
#             precision_micro_list, recall_micro_list, f1_micro_list,
#             precision_macro_smote_list, recall_macro_smote_list, f1_macro_smote_list,
#             precision_micro_smote_list, recall_micro_smote_list, f1_micro_smote_list,
#             acc_normal_list, y_original_list, y_resampled_list, y_train_list, y_train_smote_list)


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    df_original_rename = pd.read_parquet('../../../models/KMeans/output/seatunnel_all_information.parquet')
    df_original_rename = percentage_smell(df_original_rename)

    hour = df_original_rename['total_time'].dt.total_seconds() / 3600
    percentiles = calculate_percentiles(hour)

    # combianations of percentiles to divide time class for 3 classes
    time_point_list = list(itertools.combinations(percentiles.iloc, 2))
    df_time_point_index = set_index_combinations_percentiles(time_point_list)
    df_time_point_sort = table_time_fix_percentile(df_time_point_index)

    add_time_class = divide_time_class(df_original_rename, df_time_point_sort)

    add_time_class.to_parquet('../../../models/KMeans/output/tables_time_class3.parquet')

    # (precision_macro_list, recall_macro_list, f1_macro_list,
    #  precision_micro_list, recall_micro_list, f1_micro_list,
    #  precision_macro_smote_list, recall_macro_smote_list, f1_macro_smote_list,
    #  precision_micro_smote_list, recall_micro_smote_list, f1_micro_smote_list,
    #  acc_normal_list, y_original_list, y_resampled_list, y_train_list, y_train_smote_list) = split_data_x_y(df_original_rename, df_time_point_sort)
    #
    # df_time_class3 = pd.DataFrame({
    #     'acc_normal': acc_normal_list,
    #     'precision_macro': precision_macro_list,
    #     'recall_macro': recall_macro_list,
    #     'f1_macro': f1_macro_list,
    #     'precision_micro': precision_micro_list,
    #     'recall_micro_tune': recall_micro_list,
    #     'f1_micro_tune': f1_micro_list,
    #     'acc_smote': acc_normal_list,
    #     'precision_macro_smote': precision_macro_smote_list,
    #     'recall_macro_smote': recall_macro_smote_list,
    #     'f1_macro_smote': f1_macro_smote_list,
    #     'precision_micro_smote': precision_micro_smote_list,
    #     'recall_micro_smote': recall_micro_smote_list,
    #     'f1_micro_smote': f1_micro_smote_list,
    #     'y_original': y_original_list,
    #     'y_train': y_train_list,
    #     'y_resample': y_resampled_list,
    #     'y_train_smote': y_train_smote_list
    # })

    # df_time_class3.to_parquet('../../../models/KMeans/output/seatunnel_time_class3.parquet')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
