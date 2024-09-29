import itertools

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict, Any


def check_time_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into two based on whether the 'merged_at' year is >= 2024."""
    # df['merged_at'] = pd.to_datetime(df['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)

    df['merged_at'] = pd.to_datetime(df['merged_at'], errors='coerce')

    # Define cutoff date with UTC timezone
    cutoff_date = pd.Timestamp('2024-01-01', tz='UTC')
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
        df_point['time_class'] = (df_point['total_time_hours'] < percentile_value).apply(lambda x: 0 if x else 1)
        df_point['index_time01'] = [row.name] * len(df_point)
        df_point['time01'] = [percentile_value] * len(df_point)
        # Append the processed DataFrame to results
        results.append(df_point)
        point_list.append(percentile_value)

    return results, point_list


def filter_dfs_by_class_count(dfs: List[pd.DataFrame], min_count: int = 6) -> Tuple[
    List[pd.DataFrame], List[pd.DataFrame]]:
    """Filters DataFrames based on the minimum count of each class."""
    save_df_good = []
    save_df_bad = []
    for df in dfs:
        t_0 = df[df['time_class'] == 0].shape[0]
        t_1 = df[df['time_class'] == 1].shape[0]

        if (t_0 >= min_count) & (t_1 >= min_count):
            save_df_good.append(df)
        elif (t_0 >= min_count) & (t_1 < min_count):
            save_df_bad.append(df)
        elif (t_0 < min_count) & (t_1 >= min_count):
            save_df_bad.append(df)
        else:
            print("Time class is not enough")
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
            'time01': df['time01'].iloc[0],
            'f1_smote': None,
            'f1_normal': None,
            'f1_score_class0': None,
            'f1_score_class1': None,
            'f1_smote_class0': None,
            'f1_smote_class1': None,
            'time0': None,
            'time1': None,

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

            model = GradientBoostingClassifier(random_state=random_state)
            model_fit = model.fit(X_train, y_train)
            smote_fit = model.fit(X_resampled, y_resampled)

            y_pred_normal = cross_val_predict(model_fit, X_train, y_train, cv=5)

            y_pred = smote_fit.predict(X_test)
            y_pred_smote = cross_val_predict(smote_fit, X_resampled, y_resampled, cv=5)

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_dict_smote = classification_report(y_resampled, y_pred_smote, output_dict=True)

            entry['f1_score_class0'] = report_dict.get('0', {}).get('f1-score', 0.0)
            entry['f1_score_class1'] = report_dict.get('1', {}).get('f1-score', 0.0)
            entry['f1_smote_class0'] = report_dict_smote.get('0', {}).get('f1-score', 0.0)
            entry['f1_smote_class1'] = report_dict_smote.get('1', {}).get('f1-score', 0.0)

            entry['time0'] = (y_train == 0).sum()
            entry['time1'] = (y_train == 1).sum()
            entry['f1_smote'] = f1_score(y_resampled, y_pred_smote, average='macro')
            entry['f1_normal'] = f1_score(y_train, y_pred_normal, average='macro')
            # entry['report_dict'] = pd.DataFrame(report_dict).T.to_dict()

        ozone_result.append(entry)

    return ozone_result


if __name__ == '__main__':
    # Load the data
    pulsar_api = pd.read_parquet('../../models/output/pulsar_prepare_to_train_newversion_9Sep.parquet')
    pulsar_api = percentage_smell(pulsar_api)

    # Check the data time out of 2024
    before_2024, after_2024 = check_time_df(pulsar_api)

    # prepare percentile to divide time class for 2 classes
    percentile_list = [np.percentile(pulsar_api['total_time_hours'], range(1, 101))]
    percentile_df = pd.DataFrame(percentile_list).T

    data_time_class_list, data_time_point_list = process_data(pulsar_api, percentile_df)

    pulsar_result = split_data_x_y(data_time_class_list, before_2024, after_2024)

    # Convert the list of dictionaries into a DataFrame
    pulsar_result_df = pd.DataFrame(pulsar_result)

    with open('../../models/output/pulsar_teat_model_somte_newversion_class2_18Sep.parquet', 'wb') as f:
        joblib.dump(pulsar_result_df, f)
        print("save file Done!")
        print(pulsar_result_df)  # This will now print the DataFrame containing the concatenated results
