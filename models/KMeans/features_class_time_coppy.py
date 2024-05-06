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

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def avg_q3_q1(df):
    for i in range(3):
        print(i)
        df['aver_q3'] = df[[f'cluter{i}_q3']].mean(axis=1)
        df['aver_q1'] = df[[f'cluter{i}_q1']].mean(axis=1)
        df['aver_cv_q3'] = df[[f'cv_q3_c{i}']].mean(axis=1)
        df['aver_cv_q1'] = df[[f'cv_q1_c{i}']].mean(axis=1)
        print(df.columns)
    return df


def process_data(df, df_20_col, df_original):
    df_col_combined_list = []
    for _, row in df.iterrows():
        df_col_combined = pd.concat(
            [df_20_col[row['col']], pd.DataFrame(row['label'], columns=['label']), df_original['total_time']], axis=1)
        df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600
        df_col_combined[['point_min', 'point_max']] = [row['aver_q1'], row['aver_q3']]
        df_col_combined_list.append(df_col_combined)
    return df_col_combined_list


def classify_time(df_list):
    for df_processed in df_list:
        df_processed['time_class'] = df_processed.apply(
            lambda row: 0 if row['hours'] < row['point_min'] else 1 if row['point_min'] <= row['hours'] <= row[
                'point_max'] else 2, axis=1)
    return df_list


def split_data_x_y(df_classified_time, random_state=3, test_size=0.3):
    X_list = [i.iloc[:, :-6] for i in df_classified_time]
    y_list = [i['time_class'] for i in df_classified_time]
    split_data = [train_test_split(X, y, test_size=test_size, random_state=random_state) for X, y in
                  zip(X_list, y_list)]

    [list(i) for i in zip(*split_data)] + [X_list, y_list]
    X_train_list, X_test_list, y_train_list, y_test_list = [list(i) for i in zip(*split_data)]
    return X_train_list, y_train_list, X_test_list, y_test_list, X_list, y_list


def train_model(X_list, y_list, X_train_list, y_train_list, n_estimators=100, learning_rate=0.1, random_state=3,
                max_depth=3):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                       random_state=random_state, max_depth=max_depth)

    # score = (cross_val_score(model.fit(X_train_list, y_train_list), X_list, y_list, cv=5, scoring='accuracy'))
    # score = cross_val_score(model, X_list, y_list, cv=5, scoring='accuracy')
    precision_weighted = precision_score(y_list, y_pred := cross_val_predict(model, X_list, y_list, cv=5),
                                         average='weighted')
    recall_weighted = recall_score(y, y_pred, average='weighted')
    f1_weighted = f1_score(y_list, y_pred, average='weighted')

    precision_macro = precision_score(y_list, y_pred, average='macro')
    recall_macro = recall_score(y_list, y_pred, average='macro')
    f1_macro = f1_score(y_list, y_pred, average='macro')

    precision_micro = precision_score(y_list, y_pred, average='micro')
    recall_micro = recall_score(y_list, y_pred, average='micro')
    f1_micro = f1_score(y_list, y_pred, average='micro')

    return score, precision_weighted, recall_weighted, f1_weighted, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    # input files
    file_features = r'../../models/KMeans/output/q3_c3/q3_c3_5000_normal2024-05-02 07:04:12.parquet'
    df_20_col = pd.read_parquet('../../models/KMeans/output/seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

    # command of the function
    df_all_features = pd.read_parquet(file_features)

    add_avg = avg_q3_q1(df_all_features)
    df_list = process_data(add_avg, df_20_col, df_original)
    df_classified = classify_time(df_list)
    split_df = split_data_x_y(df_classified)

    X_train_list, y_train_list, X_test_list, y_test_list, X_list, y_list = split_data_x_y(df_classified)

    (precision_score, recall_score, f1_score, col_name, pre_macro, recall_macro, f1_macro, pre_micro,
     recall_micro, f1_micro) = train_model(
        X_list, y_list, X_train_list,
        y_train_list)

    new_df_normal = pd.DataFrame({
        'col_name': col_name,

        'precision_weighted': precision_score,
        'recall_weighted': recall_score,
        'f1_weighted': f1_score,

        'precision_macro': pre_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,

        'precision_micro': pre_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    })

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
