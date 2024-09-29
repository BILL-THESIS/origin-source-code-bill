import itertools

import joblib
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict, Any

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def check_time_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into two based on whether the 'merged_at' year is >= 2024."""
    df['merged_at'] = pd.to_datetime(df['merged_at'], errors='coerce')

    # Define cutoff date with UTC timezone
    cutoff_date = pd.Timestamp('2024-02-01', tz='UTC')
    after_2024 = df[df['merged_at'] >= cutoff_date]
    before_2024 = df[df['merged_at'] < cutoff_date]

    return before_2024, after_2024


def percentage_smell(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the percentage change for each smell type."""
    rename_dict = {
        'created_Dispensables': 'created_D',
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
        'ended_Uncategorized': 'ended_U'
    }
    df = df.rename(columns=rename_dict)

    for col in ['D', 'B', 'CP', 'C', 'OOA', 'U']:
        df[f'percentage_{col.lower()}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

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



def filter_dfs_by_class_count(dfs: List[pd.DataFrame], min_count: int = 6) -> Tuple[
    List[pd.DataFrame], List[pd.DataFrame]]:
    """Filters DataFrames based on the minimum count of each class."""
    save_df_good, save_df_bad = [], []
    for df in dfs:
        class_counts = df['time_class'].value_counts()
        if all(class_counts.get(i, 0) >= min_count for i in range(3)):
            save_df_good.append(df)
        else:
            save_df_bad.append(df)

    return save_df_good, save_df_bad


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


def split_data_x_y(df_list: List[pd.DataFrame], df_before_year: pd.DataFrame, df_after_year: pd.DataFrame,
                   random_state: int = 3) -> List[Dict[str, Any]]:
    """Splits data into X and y for training and testing, then fits a model and records metrics."""
    ozone_result = []

    for df in df_list:
        before_2024 = pd.merge(df, df_before_year, how='inner', on='url')
        after_2024 = pd.merge(df, df_after_year, how='inner', on='url')

        time_2024, _ = filter_dfs_by_class_count([after_2024])

        entry = {
            'index_time01': df['index_time01'].iloc[0],
            'index_time12': df['index_time12'].iloc[0],
            'time01': df['time_01'].iloc[0],
            'time12': df['time_12'].iloc[0],
            'f1_smote': None,
            'f1_normal': None,
            'f1_score_class0': None,
            'f1_score_class1': None,
            'f1_score_class2': None,
            'f1_smote_class0': None,
            'f1_smote_class1': None,
            'f1_smote_class2': None,
            'time0': None,
            'time1': None,
            'time2': None,
            'report_dict': []
        }

        for t in time_2024:
            X_train = t[['created_B_x',
                         'created_CP_x', 'created_C_x', 'created_D_x', 'created_OOA_x',
                         'created_U_x', 'ended_B_x', 'ended_CP_x', 'ended_C_x', 'ended_D_x',
                         'ended_OOA_x', 'ended_U_x', 'percentage_d_x', 'percentage_b_x',
                         'percentage_cp_x', 'percentage_c_x', 'percentage_ooa_x',
                         'percentage_u_x']]
            y_train = t['time_class']

            X_test = before_2024[['created_B_x',
                                     'created_CP_x', 'created_C_x', 'created_D_x', 'created_OOA_x',
                                     'created_U_x', 'ended_B_x', 'ended_CP_x', 'ended_C_x', 'ended_D_x',
                                     'ended_OOA_x', 'ended_U_x', 'percentage_d_x', 'percentage_b_x',
                                     'percentage_cp_x', 'percentage_c_x', 'percentage_ooa_x',
                                     'percentage_u_x']]
            y_test = before_2024['time_class']

            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            model_fit = model.fit(X_train, y_train)
            smote_fit = model.fit(X_resampled, y_resampled)

            y_pred_normal = cross_val_predict(model_fit, X_train, y_train, cv=5)

            y_pred = smote_fit.predict(X_test)
            y_pred_smote = cross_val_predict(smote_fit, X_resampled, y_resampled, cv=5)

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_dict_smote = classification_report(y_resampled, y_pred_smote, output_dict=True)

            entry['f1_score_class0'] = report_dict.get('0', {}).get('f1-score', 0.0)
            entry['f1_score_class1'] = report_dict.get('1', {}).get('f1-score', 0.0)
            entry['f1_score_class2'] = report_dict.get('2', {}).get('f1-score', 0.0)
            entry['f1_smote_class0'] = report_dict_smote.get('0', {}).get('f1-score', 0.0)
            entry['f1_smote_class1'] = report_dict_smote.get('1', {}).get('f1-score', 0.0)
            entry['f1_smote_class2'] = report_dict_smote.get('2', {}).get('f1-score', 0.0)
            entry['time0'] = (y_train == 0).sum()
            entry['time1'] = (y_train == 1).sum()
            entry['time2'] = (y_train == 2).sum()
            entry['f1_smote'] = f1_score(y_resampled, y_pred_smote, average='macro')
            entry['f1_normal'] = f1_score(y_train, y_pred_normal, average='macro')
            entry['report_dict'] = pd.DataFrame(report_dict).T.to_dict()

        ozone_result.append(entry)

    return ozone_result


if __name__ == '__main__':
    # Load the data
    ozone_api = pd.read_parquet('../../models/output/ozone_prepare_to_train_newversion_9Sep.parquet')
    ozone_api = percentage_smell(ozone_api)

    # Check the data time out of 2024
    before_2024, after_2024 = check_time_df(ozone_api)

    # Calculate the percentiles of the total_time_hours column
    ozone_percentiles = calculate_percentiles(ozone_api['total_time_hours'])
    ozone_percentile_combinations = set_index_combinations_percentiles(ozone_percentiles)
    ozone_time_points = table_time_fix_percentile(ozone_percentile_combinations)
    ozone_time_classes = divide_time_class_2(ozone_api, ozone_time_points)
    ozone_good, ozone_bad = check_amount_time_class(ozone_time_classes)


    ozone_result = split_data_x_y(ozone_good, before_2024, after_2024)

    # Convert the list of dictionaries into a DataFrame
    ozone_result_df = pd.DataFrame(ozone_result)

    ozone_result_df['std_counts'] = ozone_result_df[['time0', 'time1', 'time2']].std(axis=1)
    ozone_result_df['std_f1'] = ozone_result_df[['f1_score_class0', 'f1_score_class1', 'f1_score_class2']].std(axis=1)
    ozone_result_df['std_f1_smote'] = ozone_result_df[['f1_smote_class0', 'f1_smote_class1', 'f1_smote_class2']].std(axis=1)

    # Save the DataFrame to a parquet file
    with open('../../models/output/ozone_test_model_svc_timeclass3_28Sep.parquet', 'wb') as f:
        joblib.dump(ozone_result_df, f)
        print("save file Done!")
        print(ozone_result_df)  # This will now print the DataFrame containing the concatenated results
