from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import seaborn.objects as so
import time


from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score



def avg_q3_q1(df):
    df['aver_q3'] = (df['cluter0_q3'] + df['cluter1_q3'] + df['cluter2_q3']) / 3
    df['aver_q1'] = (df['cluter0_q1'] + df['cluter1_q1'] + df['cluter2_q1']) / 3
    df['aver_cv_q3'] = (df['cv_q3_c0'] + df['cv_q3_c1'] + df['cv_q3_c2']) / 3
    df['aver_cv_q1'] = (df['cv_q1_c0'] + df['cv_q1_c1'] + df['cv_q1_c2']) / 3
    return df


def process_data(df, df_20_col, df_original):
    df_col_combined_list = []

    for _, row in df.iterrows():
        # print("row:", row)

        df_compare = df_20_col[row['col']]
        df_arr = pd.DataFrame(row['label'], columns=['label'])

        df_col_combined = pd.concat([df_compare, df_arr, df_original['total_time']], axis=1)

        # Convert total_time to hours
        df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600

        # sort the values of 'cluter q1' and 'cluter q3' columns

        df_cluster_q1 = row[['cluter0_q1', 'cluter1_q1', 'cluter2_q1']].sort_values()
        # print("df_cluster_q1 top 1 value:", df_cluster_q1[0])
        # print("df_cluster_q1 top 2 value:", df_cluster_q1[1])
        # print("df_cluster_q1 top 3 value:", df_cluster_q1[2])

        df_cluster_q3 = row[['cluter0_q3', 'cluter1_q3', 'cluter2_q3']].sort_values()
        # print("df_cluster_q3 top 1 value:", df_cluster_q3[0])
        # print("df_cluster_q3 top 2 value:", df_cluster_q3[1])
        # print("df_cluster_q3 top 3 value:", df_cluster_q3[2])

        df_cluster_median = row[['median_0', 'median_1', 'median_2']].sort_values()

        # Assign scalar values to 'min' and 'max' columns for all rows
        # matrix 1pi
        # df_col_combined['point_min'] = [df_cluster_q1[2]] * len(df_col_combined)
        # df_col_combined['point_max'] = [df_cluster_q3[0]] * len(df_col_combined)

        # matrix 2
        # df_col_combined['point_min'] = [row['aver_q1']] * len(df_col_combined)
        # df_col_combined['point_max'] = [row['aver_q3']] * len(df_col_combined)

        # matrix 3
        # df_col_combined['point_min'] = [row['aver_cv_q1']] * len(df_col_combined)
        # df_col_combined['point_max'] = [row['aver_cv_q3']] * len(df_col_combined)

        # matrix 4
        df_col_combined['point_min'] = [df_cluster_median[0]] * len(df_col_combined)
        df_col_combined['point_max'] = [df_cluster_median[2]] * len(df_col_combined)

        df_col_combined_list.append(df_col_combined)

    return df_col_combined_list


def classify_time(df_list):
    df_list_time_class = []
    for df_processed in df_list:
        result_time = []
        for index, row in df_processed.iterrows():

            values = row['hours']
            # print("Values:", values)

            values_min = row['point_min']
            # print("Values Min:", values_min)
            values_max = row['point_max']
            # print("Values Max:", values_max)

            # Classify time based on the values of 'hours' column
            # values is 73 hours < 2 hours
            if values < values_min:
                result_time.append(0)
            # values is between 2 and 143 hours
            elif values_min >= values <= values_max:
                result_time.append(1)
            else:
                # values is greater than 143 hours
                result_time.append(2)

        df_processed['time_class'] = result_time

        counts = df_processed['time_class'].value_counts()

        # print("Counts 0:", counts[0])
        # print("Counts 1:", counts[1])
        # print("Counts 2:", counts[2])

        df_processed['time_0_shape'] = [counts[0]] * len(df_processed)
        df_processed['time_2_shape'] = [counts[2]] * len(df_processed)
        # df_processed['time_1_shape'] = [counts[1]] * len(df_processed)
        # df_processed['time_1_shape'] = [abs((counts[0] + counts[2]) - 1068)] * len(df_processed)
        # Append the processed dataset to the list
        df_list_time_class.append(df_processed)
        # print("DF List Time Class:", df_list_time_class)
    return df_list_time_class


def split_data_x_y(df_classified_time, random_state=3, test_size=0.3):
    [] = []
    X_list_x = []
    y_list = []

    # Iterate over each dataset
    for i in df_classified_time:
        y = i['time_class']
        # print("Y value::", y)
        # X = i.iloc[:, :-6]
        X = i.iloc[:, :-9]
        # print("X value::", X)
        y_list.append(y)
        X_list_x.append(X)
        # print("X list::", X_list)
        # print("Y list::", y_list)

    # Split the data into training and testing sets for each dataset
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for X, y in zip(X_list_x, y_list):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_list.append(X_train)
        # print("X Train List::", X_train_list)
        X_test_list.append(X_test)
        # print("X Test List::", X_test_list)
        y_train_list.append(y_train)
        # print("Y Train List::", y_train_list)
        y_test_list.append(y_test)
        # print("Y Test List::", y_test_list)

    return X_train_list, y_train_list, X_test_list, y_test_list, X_list_x, y_list


def train_model(X_list, y_list, X_train_list, y_train_list, n_estimators=5, learning_rate=0.1, random_state=3, max_depth=3):
    # List to store the column names of the datasets
    col_list = []

    # lists to store the precision, recall, and f1 scores for the model using the 'macro' average
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    accuracy_score_list = []



    for X, y, X_train, y_train in zip(X_list, y_list, X_train_list, y_train_list):
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           random_state=random_state, max_depth=max_depth)
        model_fit = model.fit(X_train, y_train)

        print("Model Fit::", model_fit)

        print("====================================")
        print("col_name::", X.columns.tolist())
        print("====================================")
        print("\n")

        y_pred = cross_val_predict(model_fit, X, y, cv=5)
        print("Y Pred::", y_pred)
        # print("Y Pred equal 0::", (y_pred == 0).sum())
        # print("Y Pred equal 1::", (y_pred == 1).sum())
        # print("Y Pred equal 2::", (y_pred == 2).sum())
        # print("Y Pred Shape::", y_pred.shape)


        print("\n")
        print("Confusion Matrix::", confusion_matrix(y, y_pred))
        print("\n")
        # print("Accuracy Score::", accuracy_score(y_train, y_pred))
        print("\n")


        acc_y_pred = accuracy_score(y, y_pred)
        print("Accuracy Score::", acc_y_pred)
        print("\n")

        col_list.append(X.columns.tolist())

        # Calculate the precision, recall, and f1 scores for the model using the 'macro' average
        precision_macro = precision_score(y, y_pred, average='macro')
        recall_score_val_macro = recall_score(y, y_pred, average='macro')
        f1_score_val_macro = f1_score(y, y_pred, average='macro')

        # Append the scores to the lists for Macro
        precision_macro_list.append(precision_macro)
        recall_macro_list.append(recall_score_val_macro)
        f1_macro_list.append(f1_score_val_macro)

        accuracy_score_list.append(acc_y_pred)


    return (col_list, precision_macro_list, recall_macro_list, f1_macro_list,accuracy_score_list)


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    # input files
    file_features = r'../../models/KMeans/output/q3_c3/q3_c3_not_sort_2024-05-12 16:15:22.parquet'
    df_20_col = pd.read_parquet('../../models/KMeans/output/seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

    # command of the function
    df_all_features = pd.read_parquet(file_features)
    # df_all_features = df_all_features.drop_duplicates(subset=['col'])

    # sort values mix Q3 and min Q1
    df_sort_q3 = df_all_features.sort_values(by='Q3', ascending=False).head(10)
    df_sort_q1 = df_all_features.sort_values(by='Q1', ascending=True).head(10)

    # add average Q3 and Q1
    add_avg_max = avg_q3_q1(df_sort_q3)
    add_avg_min = avg_q3_q1(df_sort_q1)

    # process data
    df_list_max = process_data(df_sort_q3, df_20_col, df_original)
    df_list_min = process_data(df_sort_q1, df_20_col, df_original)

    # call classify_time function for q3 and q1
    df_classified_max = classify_time(df_list_max)
    df_classified_min = classify_time(df_list_min)

    # split data into X and y for q3 and q1
    X_train_list_max, y_train_list_max, X_test_list_max, y_test_list_max, X_list_max, y_list_max = split_data_x_y(df_classified_max)
    X_train_list_min, y_train_list_min, X_test_list_min, y_test_list_min, X_list_min, y_list_min = split_data_x_y(df_classified_min)

    (col_list,precision_macro_list, recall_macro_list, f1_macro_list, accuracy_score_list) = train_model(X_list_max, y_list_max, X_train_list_max, y_train_list_max)

    (col_list_min, precision_macro_list_min, recall_macro_list_min, f1_macro_list_min,accuracy_score_list_min) = train_model(X_list_min, y_list_min, X_train_list_min, y_train_list_min)

    new_df_normal_gbc = pd.DataFrame({
        'col_name_max': col_list,

        'precision_macro': precision_macro_list,
        'recall_macro': precision_macro_list,
        'f1_macro': f1_macro_list,

        'acc_y_pred': accuracy_score_list,


        # min
        'col_name_min': col_list_min,

        'precision_macro_min': precision_macro_list_min,
        'recall_macro_min': precision_macro_list_min,
        'f1_macro_min': f1_macro_list_min,

        'acc_y_pred_min': accuracy_score_list_min,

    })
    # new_df_normal_gbc['aver_f1_max'] = new_df_normal_gbc['f1_macro'].mean()
    # new_df_normal_gbc['aver_f1_min'] = new_df_normal_gbc['f1_macro_min'].mean()
    # new_df_normal_gbc['sum_aver_f1_max-min'] = new_df_normal_gbc['aver_f1_max'] - new_df_normal_gbc['aver_f1_min']
    # new_df_normal_gbc['f1_max-min'] = new_df_normal_gbc['f1_macro'] - new_df_normal_gbc['f1_macro_min']

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
