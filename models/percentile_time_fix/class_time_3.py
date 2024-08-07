import time
from collections import Counter
from datetime import date

import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
import joblib


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


def divide_time_class_2(df_original, df_time_point):
    """
    Assigns time classes (0, 1, 2) to rows in df_original based on time intervals in df_time_point.

    Args:
        df_original (pd.DataFrame): The original DataFrame with a 'total_time' column.
        df_time_point (pd.DataFrame): A DataFrame with 'time01' and 'time12' columns defining intervals.

    Returns:
        list: A list of DataFrames with a new 'time_class' column.
    """

    results = []
    for index, row in df_time_point.iterrows():
        time01 = row['time01']
        time12 = row['time12']
        time_fix_hours = df_original['total_time'].dt.total_seconds() / 3600

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

        # Append the modified DataFrame to results
        results.append(df_original.copy())  # Avoid modifying the original

    return results


def prepare_data_time_class(list_df):
    # check columns time class in the list of dataframes
    store_data_time_3 = []
    store_data_time_2 = []
    for df in list_df:
        uni = df['time_class'].unique()
        if len(uni) != 3:
            print(f"Time class is not 3: {uni}")
        elif len(uni) == 3:
            print(f"Time class is 3: {uni}")
            values_3 = df
            store_data_time_3.append(values_3)
        elif len(uni) == 2:
            print(f"Time class is 2: {uni}")
            values_2 = df
            store_data_time_2.append(values_2)
        else:
            print(f"Time class is 1: {uni}")
    return store_data_time_2, store_data_time_3


def check_amount_time_class(df):
    save_df_good = []
    save_df_bad = []
    for df in df:
        t_0 = df[df['time_class'] == 0].shape[0]
        t_1 = df[df['time_class'] == 1].shape[0]
        t_2 = df[df['time_class'] == 2].shape[0]
        print("Time class 0: ", df[df['time_class'] == 0].shape[0])
        print("Time class 1: ", df[df['time_class'] == 1].shape[0])
        print("Time class 2: ", df[df['time_class'] == 2].shape[0])

        if (t_0 > 1) & (t_1 > 1) & (t_2 > 1):
            save_df_good.append(df)
        elif (t_0 > 1) & (t_1 > 1) & (t_2 == 0):
            save_df_good.append(df)
        elif (t_0 > 1) & (t_1 == 0) & (t_2 > 1):
            save_df_good.append(df)
        elif (t_0 == 0) & (t_1 > 1) & (t_2 > 1):
            save_df_good.append(df)
        elif (t_0 > 1) & (t_1 == 0) & (t_2 == 0):
            save_df_bad.append(df)
        elif (t_0 == 0) & (t_1 > 1) & (t_2 == 0):
            save_df_bad.append(df)
        elif (t_0 == 0) & (t_1 == 0) & (t_2 > 1):
            save_df_bad.append(df)
        elif (t_0 == 0) & (t_1 == 0) & (t_2 == 0):
            save_df_bad.append(df)
        else:
            print("Time class is not enough")
    return save_df_good, save_df_bad


def split_data_x_y(df, random_state=3, test_size=0.3):
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    acc_normal_list = []

    y_original_list = []
    y_train_list = []

    list_indx_time01 = []
    list_indx_time12 = []
    list_time01 = []
    list_time12 = []

    for col in df:
        index_time01 = col['index_time01'].iloc[0]
        index_time12 = col['index_time12'].iloc[0]
        time01 = col['time_01'].iloc[0]
        time12 = col['time_12'].iloc[0]

        X = col[['created_D', 'created_B', 'created_CP', 'created_C', 'created_OOA',
                 'ended_D', 'ended_B', 'ended_CP', 'ended_C', 'ended_OOA',
                 'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
        y = col['time_class']
        print('Original dataset shape %s' % Counter(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state)
        print('y_train dataset shape %s', Counter(y_train))

        model = GradientBoostingClassifier()
        normal_fit = model.fit(X_train, y_train)

        y_pred = cross_val_predict(normal_fit, X_train, y_train, cv=5)
        acc_normal = accuracy_score(y_train, y_pred)

        precision_macro_list.append(precision_score(y_train, y_pred, average='macro'))
        recall_macro_list.append(recall_score(y_train, y_pred, average='macro'))
        f1_macro_list.append(f1_score(y_train, y_pred, average='macro'))

        acc_normal_list.append(acc_normal)
        y_original_list.append(Counter(y))
        y_train_list.append(Counter(y_train))

        list_indx_time01.append(index_time01)
        list_indx_time12.append(index_time12)
        list_time01.append(time01)
        list_time12.append(time12)

    return (precision_macro_list, recall_macro_list, f1_macro_list,
            acc_normal_list, y_original_list, y_train_list,
            list_indx_time01, list_indx_time12, list_time01, list_time12)


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    df_original_rename = pd.read_parquet('../output/ozone_prepare_to_train.parquet')
    df_original_rename = percentage_smell(df_original_rename)

    hour = df_original_rename['total_time'].dt.total_seconds() / 3600
    percentiles = calculate_percentiles(hour)

    # combianations of percentiles to divide time class for 3 classes
    time_point_list = list(itertools.combinations(percentiles.iloc, 2))
    df_time_point_index = set_index_combinations_percentiles(time_point_list)
    df_time_point_sort = table_time_fix_percentile(df_time_point_index)

    df_time_class_lists = divide_time_class_2(df_original_rename, df_time_point_sort)

    class_2, class_3 = prepare_data_time_class(df_time_class_lists)

    g, b = check_amount_time_class(class_3)

    (precision_macro_list, recall_macro_list, f1_macro_list,
     acc_normal_list, y_original_list, y_train_list,
     list_indx_time01, list_indx_time12, list_time01, list_time12) = split_data_x_y(g)

    df_time_class3 = {
        'accuracy': acc_normal_list,
        'precision_macro': precision_macro_list,
        'recall_macro': recall_macro_list,
        'f1_macro': f1_macro_list,
        'y_original': y_original_list,
        'y_train': y_train_list,
        'index_time01': list_indx_time01,
        'time01': list_time01,
        'index_time12': list_indx_time12,
        'time12': list_time12
    }

    df_time_class3 = pd.DataFrame.from_dict(df_time_class3, orient='index')
    df_time_class3 = df_time_class3.T

    with open('../output/class_time_3_normal.parquet', 'wb') as f:
        joblib.dump(df_time_class3, f)
        print("save file Done!")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
