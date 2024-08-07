import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


def avg_q3_q1(df):
    if isinstance(df.iloc[0]['cluter0_q1'], list):
        df['aver_q1'] = df.apply(
            lambda row: np.mean([row['cluter0_q1'][0], row['cluter1_q1'][0], row['cluter2_q1'][0]]), axis=1)
        df['aver_q3'] = df.apply(
            lambda row: np.mean([row['cluter0_q3'][0], row['cluter1_q3'][0], row['cluter2_q3'][0]]), axis=1)
        df['aver_cv_q1'] = df.apply(lambda row: np.mean([row['cv_q1_c0'][0], row['cv_q1_c1'][0], row['cv_q1_c2'][0]]),
                                    axis=1)
        df['aver_cv_q3'] = df.apply(lambda row: np.mean([row['cv_q3_c0'][0], row['cv_q3_c1'][0], row['cv_q3_c2'][0]]),
                                    axis=1)
    else:
        df['aver_q1'] = df[['cluter0_q1', 'cluter1_q1', 'cluter2_q1']].mean(axis=1)
        df['aver_3'] = df[['cluter0_q3', 'cluter1_q3', 'cluter2_q3']].mean(axis=1)
        df['aver_cv_q1'] = df[['cv_q1_c0', 'cv_q1_c1', 'cv_q1_c2']].mean(axis=1)
        df['aver_cv_q3'] = df[['cv_q3_c0', 'cv_q3_c1', 'cv_q3_c2']].mean(axis=1)
    return df


def process_data(df, df_20_col, df_original):
    df_col_combined_list = []

    for _, row in df.iterrows():
        df_compare = df_20_col[row['col']]
        df_arr = pd.DataFrame(row['label'], columns=['label'])

        df_col_combined = pd.concat([df_compare, df_arr, df_original['total_time']], axis=1)

        # Convert total_time to hours
        df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600

        # sort the values of 'cluter q1' and 'cluter q3' columns
        df_cluster_q1 = row[['cluter0_q1', 'cluter1_q1', 'cluter2_q1']].sort_values()
        df_cluster_q3 = row[['cluter0_q3', 'cluter1_q3', 'cluter2_q3']].sort_values()

        # Assign scalar values to 'min' and 'max' columns for all rows
        # solution 1
        df_col_combined['time_01'] = len(df_col_combined) * [df_cluster_q1.iloc[2]]
        df_col_combined['time_12'] = len(df_col_combined) * [df_cluster_q3.iloc[0]]

        df_col_combined_list.append(df_col_combined)

    return df_col_combined_list


def classify_time(df_list):
    for df in df_list:
        df['time_class'] = df.apply(lambda row: classify_hours(row['hours'], row['time_01'], row['time_12']),
                                    axis=1)
    return df_list


def classify_hours(hours, min_val, max_val):
    if isinstance(hours, list):
        hours = hours[0]
    if isinstance(min_val, list):
        min_val = min_val[0]
    if isinstance(max_val, list):
        max_val = max_val[0]
    if hours < min_val:
        return 0
    elif min_val <= hours <= max_val:
        return 1
    else:
        return 2


def split_data_x_y(df_classified_time, random_state=3, test_size=0.3):
    list_X_t01 = []
    list_X_t12 = []
    # Split the data into training and testing sets for each dataset
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # Iterate over each dataset
    for col in df_classified_time:
        y = col['time_class']
        point_time_01 = col['time_01'].iloc[0]
        point_time_02 = col['time_12'].iloc[0]
        X = col.drop(columns=['total_time', 'hours', 'time_01', 'time_12', 'time_class', 'label'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        y_list.append(y)
        X_list.append(X)
        list_X_t01.append(point_time_01)
        list_X_t12.append(point_time_02)

    return X_train_list, y_train_list, X_test_list, y_test_list, list_X_t01, list_X_t12


def train_model(X_train_list, y_train_list, X_test_list, y_test_list, n_estimators=100,
                learning_rate=1.0, random_state=3,
                max_depth=2, subsample=0.7):
    col_list = []

    # lists to store the precision, recall, and f1 scores for the model using the 'macro' average
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    # time lists
    time_0_list, time_1_list, time_2_list = [], [], []

    accuracy_score_list = []

    for X_train, y_train, X_test, y_test in zip(X_train_list, y_train_list, X_test_list,
                                                y_test_list):
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           random_state=random_state, max_depth=max_depth, subsample=subsample)

        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=3, max_depth=2, subsample=0.7)
        model_fit = model.fit(X_train, y_train)

        time0 = (y_train == 0).sum()
        time1 = (y_train == 1).sum()
        time2 = (y_train == 2).sum()

        time_0_list.append(time0)
        time_1_list.append(time1)
        time_2_list.append(time2)

        print("====================================")
        print("col_name::", X_train.columns.tolist())
        print("====================================")

        y_pred = cross_val_predict(model_fit, X_train, y_train, cv=5)

        acc = accuracy_score(y_train, y_pred)
        # print("Accuracy Score::", accuracy_score(y, y_pred))
        # print("\n")
        accuracy_score_list.append(acc)

        col_list.append(X_train.columns.tolist())

        # Calculate the precision, recall, and f1 scores for the model using the 'macro' average
        precision_macro = precision_score(y_train, y_pred, average='macro')
        recall_score_val_macro = recall_score(y_train, y_pred, average='macro')
        f1_score_val_macro = f1_score(y_train, y_pred, average='macro')

        # Append the scores to the lists for Macro
        precision_macro_list.append(precision_macro)
        recall_macro_list.append(recall_score_val_macro)
        f1_macro_list.append(f1_score_val_macro)

    return (col_list,
            precision_macro_list, recall_macro_list, f1_macro_list,
            time_0_list, time_1_list, time_2_list,
            accuracy_score_list
            )


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    # input files
    # file_features = r'../../models/KMeans/output/3/quartile_data_2024-05-24 04:55:29.parquet.gz'
    file_features = r'../../output/q3_c3_15_col_not_sort_2024.parquet.gz'
    df_20_col = pd.read_parquet('../../output/seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../../Sonar/seatunnel_all_information.parquet')

    # command of the function
    with open(os.path.join(file_features), 'rb') as f:
        df_all_features = joblib.load(f)

    df_all_features = pd.read_parquet(file_features)

    add_avg = avg_q3_q1(df_all_features)

    df_list_q3 = process_data(add_avg, df_20_col, df_original)
    df_classified_q3 = classify_time(df_list_q3)

    X_list, y_list, X_train_list, y_train_list, X_test_list, y_test_list, list_x_t01, list_x_t12 = split_data_x_y(
        df_classified_q3)

    (col_list, precision_macro_list, recall_macro_list, f1_macro_list,
     time_0_list, time_1_list, time_2_list, accuracy_score_list) = train_model(X_list, y_list, X_train_list,
                                                                               y_train_list, X_test_list, y_test_list)

    new_df_normal_gbc = pd.DataFrame({
        'col_name_q3': col_list,

        'accuracy_score': accuracy_score_list,

        'precision_macro_q3': precision_macro_list,
        'recall_macro_q3': precision_macro_list,
        'f1_macro_q3': f1_macro_list,

        'time 0': time_0_list,
        'time 1': time_1_list,
        'time 2': time_2_list,

        'point_time01': list_x_t01,
        'point_time12': list_x_t12

    })

    new_df_normal_gbc.to_parquet("over_all_GBC_matrix1.parquet")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
