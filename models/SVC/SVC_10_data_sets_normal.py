import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


def process_data(df, df_20_col, df_original):
    df_col_combined_list = []
    for _, row in df.iterrows():
        df_compare = df_20_col[row['col']]
        df_arr = pd.DataFrame(row['label'], columns=['label'])

        df_col_combined = pd.concat([df_compare, df_arr, df_original['total_time']], axis=1)

        # Convert total_time to hours
        df_col_combined['hours'] = pd.to_timedelta(df_col_combined['total_time']).dt.total_seconds() / 3600

        # Add min and max columns
        df_col_combined['min'] = row['min']
        df_col_combined['max'] = row['max']

        df_col_combined_list.append(df_col_combined)

        # print("DF Combined ::", df_col_combined)
    return df_col_combined_list


import pandas as pd


def classify_time(df_list):
    df_list_time_class = []
    for df_processed in df_list:
        result_time = []
        for index, row in df_processed.iterrows():
            values = row['hours']
            print("Values:", values)
            values_min = row['min']
            print("Values Min:", values_min)
            values_max = row['max']
            print("Values Max:", values_max)
            if values < values_min:
                result_time.append(0)
            elif values_min <= values <= values_max:
                result_time.append(1)
            else:
                result_time.append(2)
        df_processed['time_class'] = result_time
        df_list_time_class.append(df_processed)
    print("DF List Time Class:", df_list_time_class)
    return df_list_time_class


def split_data_x_y(df_classified_time, random_state=3, test_size=0.3):
    X_list = []
    y_list = []

    # Iterate over each dataset
    for i in df_classified_time:
        y = i['time_class']
        print("Y value::", y)
        X = i.iloc[:, :-6]
        print("X value::", X)
        y_list.append(y)
        X_list.append(X)
        print("X list::", X_list)
        print("Y list::", y_list)

    # Split the data into training and testing sets for each dataset
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for X, y in zip(X_list, y_list):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train_list.append(X_train)
        print("X Train List::", X_train_list)
        X_test_list.append(X_test)
        print("X Test List::", X_test_list)
        y_train_list.append(y_train)
        print("Y Train List::", y_train_list)
        y_test_list.append(y_test)
        print("Y Test List::", y_test_list)

    return X_train_list, y_train_list, X_test_list, y_test_list, X_list, y_list


def train_model(X_list, y_list, X_train_list, y_train_list):
    # model_list = []
    sore_list = []

    precision_list = []
    recall_score_list = []
    f1_score_list = []

    col_list = []

    precision_macro_list = []
    recall_score_list_macro = []
    f1_score_list_macro = []

    precision_micro_list = []
    recall_score_list_micro = []
    f1_score_list_micro = []

    for X, y, X_train, y_train in zip(X_list, y_list, X_train_list, y_train_list):
        model = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=2))

        scores = cross_val_score(model, X, y, cv=5)
        sore_list.append(scores)

        y_pred = cross_val_predict(model, X, y, cv=5)
        print("Y Pred::", y_pred)

        precision = precision_score(y, y_pred, average='weighted')
        recall_score_val = recall_score(y, y_pred, average='weighted')
        f1_score_val = f1_score(y, y_pred, average='weighted')

        precision_macro = precision_score(y, y_pred, average='macro')
        recall_score_val_macro = recall_score(y, y_pred, average='macro')
        f1_score_val_macro = f1_score(y, y_pred, average='macro')

        precision_micro = precision_score(y, y_pred, average='micro')
        recall_score_val_micro = recall_score(y, y_pred, average='micro')
        f1_score_val_micro = f1_score(y, y_pred, average='micro')

        precision_list.append(precision)
        recall_score_list.append(recall_score_val)
        f1_score_list.append(f1_score_val)

        col_list.append(X.columns.tolist())

        precision_macro_list.append(precision_macro)
        recall_score_list_macro.append(recall_score_val_macro)
        f1_score_list_macro.append(f1_score_val_macro)

        precision_micro_list.append(precision_micro)
        recall_score_list_micro.append(recall_score_val_micro)
        f1_score_list_micro.append(f1_score_val_micro)

    return (sore_list, precision_list, recall_score_list, f1_score_list, col_list,
            precision_macro_list, recall_score_list_macro, f1_score_list_macro,
            precision_micro_list, recall_score_list_micro, f1_score_list_micro)


if __name__ == '__main__':
    directory_path_max = r'../../models/KMeans/output/q3_c3/df_max_time_class_normal.parquet'
    df_20_col = pd.read_parquet('../../models/KMeans/output/seatunnal_20col.parquet')
    df_original = pd.read_parquet('../../Sonar/seatunnel_all_information.parquet')

    df_max_class = pd.read_parquet(directory_path_max)
    df_process = process_data(df_max_class, df_20_col, df_original)
    df_classified = classify_time(df_process)
    X_train_list, y_train_list, X_test_list, y_test_list, X_list, y_list = split_data_x_y(df_classified)
    X_train_list, y_train_list, X_test_list, y_test_list, X_list, y_list = split_data_x_y(df_classified)
    (sores_list, precision_score, recall_score, f1_score, col_name, pre_macro, recall_macro, f1_macro, pre_micro,
     recall_micro, f1_micro) = train_model(
        X_list, y_list, X_train_list,
        y_train_list)

    new_df_svc_normal = pd.DataFrame({
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

    # new_df_svc.to_csv('../../models/SVC/SVC_10_data_sets.csv')