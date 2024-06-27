import itertools
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV


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

    df['percentage_b'] = ((df['ended_D'] - df['created_D'].astype(float)) / df['created_D'].astype(float)) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100

    return df


def process_data(df, percentile_df):
    results = []
    point_list = []

    for _, row in percentile_df.iterrows():
        # Assuming 'row' is a Series with one value
        percentile_value = row[0]
        print("point ::", percentile_value)

        # Create a copy of the DataFrame for the current percentile
        df_point = df.copy()

        # Apply the condition using vectorized operations
        # df_point['time_class'] = (df['total_time'] < percentile_value).astype(int)
        # df_point['time_class'] = np.where(df_point['total_time'] < percentile_value, 0, 1)
        df_point['time_class'] = (df_point['total_time'] < percentile_value).apply(lambda x: 0 if x else 1)
        # Append the processed DataFrame to results
        results.append(df_point)
        point_list.append(percentile_value)

    return results, point_list


def split_data_x_y(df, random_state=3, test_size=0.3 ):
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    precision_macro_tune_list = []
    recall_macro_tune_list = []
    f1_macro_tune_list = []
    acc_normal_list = []
    acc_tune_list = []

    y0_list = []
    y1_list = []

    params_list = []

    for col in df:
        X = col[['created_D', 'created_B', 'created_CP', 'created_C', 'created_OOA',
                 'ended_D', 'ended_B', 'ended_CP', 'ended_C', 'ended_OOA',
                 'percentage_b', 'percentage_cp', 'percentage_c', 'percentage_ooa']]
        y = col['time_class']

        y0 = (y == 0).sum()
        y1 = (y == 1).sum()
        print("sum y 0", (y == 0).sum())
        print("sum y 1", (y == 1).sum())
        print(len(y0_list))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        parameters = {

            # 'learning_rate': [0.01, 0.1, 1.0],
            # 'n_estimators': [0.1, 1, 100],
            # 'subsample': [0.7],
            # 'max_depth': [1, 2, 3],
            # 'random_state': [3, 5],

            'learning_rate': [0.01, 1.0],
            'n_estimators': [500, 5000],
            'subsample': [0.1, 1.0],
            'max_depth': [5, 20],
            'min_samples_split': [128, 512],
            'min_samples_leaf': [64, 256],

        }

        GradientBoosting = GradientBoostingClassifier()

        normal_fit = GradientBoosting
        normal_fit.fit(X_train, y_train)

        tune_fit = GridSearchCV(GradientBoosting, parameters, cv=3, scoring='f1_macro')
        tune_fit.fit(X_train, y_train)

        y_pred_normal = cross_val_predict(normal_fit, X_train, y_train, cv=5)
        y_pred_clf = tune_fit.best_estimator_.predict(X_train)
        best_params = tune_fit.best_params_

        acc_normal = accuracy_score(y_train, y_pred_normal)
        acc_tune = accuracy_score(y_train, y_pred_clf)


        precision_macro_list.append(precision_score(y_train, y_pred_normal, average='macro'))
        recall_macro_list.append(recall_score(y_train, y_pred_normal, average='macro'))
        f1_macro_list.append(f1_score(y_train, y_pred_normal, average='macro'))

        precision_macro_tune_list.append(precision_score(y_train, y_pred_clf, average='macro'))
        recall_macro_tune_list.append(recall_score(y_train, y_pred_clf, average='macro'))
        f1_macro_tune_list.append(f1_score(y_train, y_pred_clf, average='macro'))

        acc_normal_list.append(acc_normal)
        acc_tune_list.append(acc_tune)
        y0_list.append(y0)
        y1_list.append(y1)


        params_list.append(best_params)

    return precision_macro_list, recall_macro_list, f1_macro_list, precision_macro_tune_list, recall_macro_tune_list, f1_macro_tune_list, acc_normal_list, acc_tune_list, y0_list, y1_list, params_list


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    df_original = pd.read_parquet('../../../models/KMeans/output/seatunnel_all_information.parquet')

    # prepare the data X
    df_original_rename = percentage_smell(df_original)

    # prepare percentile to divide time class for 2 classes
    percentile_list = [np.percentile(df_original['total_time'], range(1, 100))]

    percentile_df = pd.DataFrame(percentile_list).T
    # print(len(percentile_df))

    p, point_list = process_data(df_original_rename, percentile_df)

    precision_macro_list, recall_macro_list, f1_macro_list, precision_macro_tune_list, recall_macro_tune_list, f1_macro_tune_list, acc_normal_list, acc_tune_list, y0_list, y1_list, params_list = split_data_x_y(
        p)

    df_time_class2 = pd.DataFrame({
        'accuracy_normal': acc_normal_list,
        'precision_macro': precision_macro_list,
        'recall_macro': recall_macro_list,
        'f1_macro': f1_macro_list,
        'accuracy_tune': acc_tune_list,
        'precision_macro_tune': precision_macro_tune_list,
        'recall_macro_tune': recall_macro_tune_list,
        'f1_macro_tune': f1_macro_tune_list,
        'time0': y0_list,
        'time1': y1_list,
        'point': point_list,
        'best_params': params_list
    })

    df_time_class2.to_parquet('../../../models/KMeans/output/seatunnel_time_class2.parquet')

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
