import itertools
import os
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


def check_time_list(df):
    morethen_2024 = []
    lessthen_2024 = []
    df['merged_at'] = pd.to_datetime(df['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
    print("Data date time part Year :", df['merged_at'].dt.year)
    more = df[df['merged_at'].dt.year >= 2024]
    less = df[df['merged_at'].dt.year < 2024]
    return more, less


def check_time_df(df):
    df['merged_at'] = pd.to_datetime(df['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
    return df[df['merged_at'].dt.year >= 2024], df[df['merged_at'].dt.year < 2024]


def percentage_smell(df):
    df = df.rename(columns={'created_Dispensables': 'created_D',
                            'created_Bloaters': 'created_B',
                            'created_Change Preventers': 'created_CP',
                            'created_Couplers': 'created_C',
                            'created_Object-Orientation Abusers': 'created_OOA',
                            'created_Uncategorized': 'created_U',
                            'ended_Dispensables': 'ended_D',
                            'ended_Bloaters': 'ended_B',
                            'ended_Change Preventers': 'ended_CP',
                            'ended_Couplers': 'ended_C',
                            'ended_Object-Orientation Abusers': 'ended_OOA',
                            'ended_Uncategorized': 'ended_U'})

    df['created_D'].astype(float)
    df['percentage_d'] = ((df['ended_D'] - df['created_D']) / df['created_D']) * 100
    df['percentage_b'] = ((df['ended_B'] - df['created_B']) / df['created_B']) * 100
    df['percentage_cp'] = ((df['ended_CP'] - df['created_CP']) / df['created_CP']) * 100
    df['percentage_c'] = ((df['ended_C'] - df['created_C']) / df['created_C']) * 100
    df['percentage_ooa'] = ((df['ended_OOA'] - df['created_OOA']) / df['created_OOA']) * 100
    df['percentage_u'] = ((df['ended_U'] - df['created_U']) / df['created_U']) * 100
    return df


def calculate_percentiles(date_series):
    percentiles = np.percentile(date_series, range(0, 101))
    return pd.DataFrame(percentiles, columns=['percentile'])


def set_index_combinations_percentiles(percentile_df):
    return [percentile_df.iloc[list(comb)] for comb in itertools.combinations(range(len(percentile_df)), 2)]


def table_time_fix_percentile(percentile_dfs):
    time_points = []
    for df in percentile_dfs:
        time_01, time_12 = df['percentile']
        index_time01, index_time12 = df.index

        time_points.append({
            'index_time01': index_time01,
            'index_time12': index_time12,
            'time01': time_01,
            'time12': time_12
        })

    return pd.DataFrame(time_points)


def divide_time_class_2(df_original, df_time_point):
    results = []
    for index, row in df_time_point.iterrows():
        # Create a copy of the DataFrame for the current percentile
        time01 = row['time01']
        time12 = row['time12']
        time_fix_hours = df_original['total_time_hours']

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
        results.append(df_original.copy())
        # Avoid modifying the original DataFrame

    return results


def check_amount_time_class(dfs):
    save_df_good = []
    save_df_bad = []

    for df in dfs:
        class_counts = df['time_class'].value_counts()

        t_0 = class_counts.get(0, 0)
        t_1 = class_counts.get(1, 0)
        t_2 = class_counts.get(2, 0)

        # Check if all classes have at least 5 instances
        if t_0 >= 6 and t_1 >= 6 and t_2 >= 6:
            save_df_good.append(df)
        else:
            save_df_bad.append(df)

    return save_df_good, save_df_bad


def check_amount_time_divded(dfs):
    save_df_good = []
    save_df_bad = []

    for df in dfs:
        class_counts = df['time_class'].value_counts()

        t_0 = class_counts.get(0, 0)
        t_1 = class_counts.get(1, 0)
        t_2 = class_counts.get(2, 0)

        # Check if all classes have at least 4 instances
        if t_0 >= 4 and t_1 >= 4 and t_2 >= 4:
            save_df_good.append(df)
        else:
            save_df_bad.append(df)

    return save_df_good, save_df_bad


def split_data_x_y(df_list, df_more_2024, df_less_2024, random_state=3):
    metrics = {
        'index_time01': [],
        'index_time12': [],
        'time01': [],
        'time12': [],
        'f1_smote': [],
        'precision_class_0': [],
        'recall_class_1': [],
        'f1_score_class_2': []

    }

    # Check the data date time to prepare train and test
    for col in df_list:
        data_more_2024 = pd.merge(col, df_more_2024, how='inner', on='url')
        data_less_2024 = pd.merge(col, df_less_2024, how='inner', on='url')

        print(f"Data more 2024 count values time class : {data_more_2024['time_class'].value_counts()}")
        print(f"Data less 2024 count values time class : {data_less_2024['time_class'].value_counts()}")

        # check the data time
        time_2024, not_used = check_amount_time_class([data_less_2024])

        print(f"Time 2024 : {time_2024}")
        print(f"Time 2024 not used : {not_used}")

        # Extract time-related columns
        metrics['index_time01'].append(col['index_time01'].iloc[0])
        metrics['index_time12'].append(col['index_time12'].iloc[0])
        metrics['time01'].append(col['time_01'].iloc[0])
        metrics['time12'].append(col['time_12'].iloc[0])

        for t in time_2024:
            # print(f"Time 2024 : {t}")
            X_train = t[['created_B_x', 'created_CP_x', 'created_C_x', 'created_D_x', 'created_OOA_x', 'created_U_x',
                         'ended_B_x', 'ended_CP_x', 'ended_C_x', 'ended_D_x', 'ended_OOA_x', 'ended_U_x',
                         'percentage_d_x', 'percentage_b_x', 'percentage_cp_x', 'percentage_c_x', 'percentage_ooa_x',
                         'percentage_u_x']]
            y_train = t['time_class']

            X_test = data_more_2024[['created_B_x',
                                     'created_CP_x', 'created_C_x', 'created_D_x', 'created_OOA_x',
                                     'created_U_x', 'ended_B_x', 'ended_CP_x', 'ended_C_x', 'ended_D_x',
                                     'ended_OOA_x', 'ended_U_x', 'percentage_d_x', 'percentage_b_x',
                                     'percentage_cp_x', 'percentage_c_x', 'percentage_ooa_x',
                                     'percentage_u_x']]
            y_test = data_more_2024['time_class']

            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            X_train_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_train_resampled = pd.Series(y_resampled)
            X_test_resampled = X_test
            y_test_resampled = y_test

            model = GradientBoostingClassifier(random_state=random_state)
            smote_fit = model.fit(X_train_resampled, y_train_resampled)

            y_pred = smote_fit.predict(X_test_resampled)
            y_pred_smote = cross_val_predict(smote_fit, X_train_resampled, y_train_resampled, cv=5)

            report_dict = classification_report(y_test_resampled, y_pred, output_dict=True)
            # Access specific values
            precision_class_0 = report_dict['0']['precision']
            recall_class_1 = report_dict['1']['recall']
            f1_score_class_2 = report_dict['2']['f1-score']

            metrics['precision_class_0'].append(precision_class_0)
            metrics['recall_class_1'].append(recall_class_1)
            metrics['f1_score_class_2'].append(f1_score_class_2)
            metrics['f1_smote'].append(f1_score(y_train_resampled, y_pred_smote, average='macro'))

    return (metrics['index_time01'], metrics['index_time12'],
            metrics['time01'], metrics['time12'], metrics['f1_smote'],
            metrics['precision_class_0'],
            metrics['recall_class_1'], metrics['f1_score_class_2'])


if __name__ == '__main__':
    # step 1: Load the data
    ozone_api = pd.read_parquet('../../models/output/seatunnel_prepare_to_train.parquet')
    ozone_api = percentage_smell(ozone_api)

    # step 2: Check the data time out of 2024
    # data more than 2024 prepare to test  and less than 2024
    more_2024, less_2024 = check_time_df(ozone_api)

    # step 3: Calculate the percentiles of the total_time_hours column on
    ozone_percentiles = calculate_percentiles(ozone_api['total_time_hours'])
    ozone_percentile_combinations = set_index_combinations_percentiles(ozone_percentiles)
    ozone_time_points = table_time_fix_percentile(ozone_percentile_combinations)
    ozone_time_classes = divide_time_class_2(ozone_api, ozone_time_points)
    ozone_good, ozone_bad = check_amount_time_class(ozone_time_classes)

    index1, index2, time01, time02, f1_smote, class0, class1, class2 = split_data_x_y(ozone_good[:4], more_2024,
                                                                                      less_2024)

    ozone_reuslf = pd.DataFrame({'index_time01': index1, 'index_time12': index2, 'time01': time01, 'time12': time02,
                                 'f1_smote': f1_smote, 'precision_class_0': class0, 'recall_class_1': class1,
                                 'f1_score_class_2': class2})
