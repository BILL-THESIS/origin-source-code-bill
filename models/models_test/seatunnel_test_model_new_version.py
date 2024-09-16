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
    df['merged_at'] = pd.to_datetime(df['merged_at'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)
    return df[df['merged_at'].dt.year >= 2024], df[df['merged_at'].dt.year < 2024]


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


def calculate_percentiles(date_series: pd.Series) -> pd.DataFrame:
    """Calculates percentiles for a given Series."""
    percentiles = np.percentile(date_series, range(0, 101))
    return pd.DataFrame(percentiles, columns=['percentile'])


def set_index_combinations_percentiles(percentile_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Generates combinations of percentile indexes."""
    return [percentile_df.iloc[list(comb)] for comb in itertools.combinations(range(len(percentile_df)), 2)]


def table_time_fix_percentile(percentile_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Creates a table of time points based on percentiles."""
    time_points = [{
        'index_time01': df.index[0],
        'index_time12': df.index[1],
        'time01': df['percentile'].iloc[0],
        'time12': df['percentile'].iloc[1]
    } for df in percentile_dfs]

    return pd.DataFrame(time_points)


def divide_time_class(df_original: pd.DataFrame, df_time_point: pd.DataFrame) -> List[pd.DataFrame]:
    """Divides time classes based on time points and returns a list of DataFrames."""
    results = []
    for _, row in df_time_point.iterrows():
        time01, time12 = row['time01'], row['time12']
        time_class = pd.cut(df_original['total_time_hours'], bins=[-np.inf, time01, time12, np.inf], labels=[0, 1, 2])
        df_copy = df_original.copy()
        df_copy['time_class'] = time_class
        df_copy['index_time01'] = row['index_time01']
        df_copy['time_01'] = row['time01']
        df_copy['index_time12'] = row['index_time12']
        df_copy['time_12'] = row['time12']
        results.append(df_copy)
    return results


def filter_dfs_by_class_count(dfs: List[pd.DataFrame], min_count: int = 6) -> Tuple[
    List[pd.DataFrame], List[pd.DataFrame]]:
    """Filters DataFrames based on the minimum count of each class."""
    save_df_good, save_df_bad = [], []
    for df in dfs:
        class_counts = df['time_class'].value_counts()
        if all(class_counts.get(i, 0) >= min_count for i in range(3)):
            save_df_good.append(df)
        else :
            save_df_bad.append(df)

    return save_df_good, save_df_bad


def split_data_x_y(df_list: List[pd.DataFrame], df_more_2024: pd.DataFrame, df_less_2024: pd.DataFrame,
                   random_state: int = 3) -> List[Dict[str, Any]]:
    """Splits data into X and y for training and testing, then fits a model and records metrics."""
    ozone_result = []

    for df in df_list:
        data_more_2024 = pd.merge(df, df_more_2024, how='inner', on='url')
        data_less_2024 = pd.merge(df, df_less_2024, how='inner', on='url')

        time_2024, _ = filter_dfs_by_class_count([data_less_2024])

        entry = {
            'index_time01': df['index_time01'].iloc[0],
            'index_time12': df['index_time12'].iloc[0],
            'time01': df['time_01'].iloc[0],
            'time12': df['time_12'].iloc[0],
            'f1_smote': None,
            'precision_class0': None,
            'precision_class1': None,
            'precision_class2': None,
            'recall_class1': None,
            'recall_class2': None,
            'recall_class3': None,
            'f1_score_class1': None,
            'f1_score_class2': None,
            'f1_score_class3': None,
            'support_class1': None,
            'support_class2': None,
            'support_class3': None,
            'f1_smote': None,
            'time0': None,
            'time1': None,
            'time2': None,
            'report_dict': []
        }

        for t in time_2024:
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

            model = GradientBoostingClassifier(random_state=random_state)
            smote_fit = model.fit(X_resampled, y_resampled)

            y_pred = smote_fit.predict(X_test)
            y_pred_smote = cross_val_predict(smote_fit, X_resampled, y_resampled, cv=5)

            report_dict = classification_report(y_test, y_pred, output_dict=True)

            entry['precision_class0'] = (report_dict.get('0', {}).get('precision', 0.0))
            entry['precision_class1'] = report_dict.get('1', {}).get('precision', 0.0)
            entry['precision_class2'] = report_dict.get('2', {}).get('precision', 0.0)
            entry['recall_class1'] = report_dict.get('0', {}).get('recall', 0.0)
            entry['recall_class2'] = report_dict.get('1', {}).get('recall', 0.0)
            entry['recall_class3'] = report_dict.get('2', {}).get('recall', 0.0)
            entry['f1_score_class1'] = report_dict.get('0', {}).get('f1-score', 0.0)
            entry['f1_score_class2'] = report_dict.get('1', {}).get('f1-score', 0.0)
            entry['f1_score_class3'] = report_dict.get('2', {}).get('f1-score', 0.0)
            entry['support_class1'] = report_dict.get('0', {}).get('support', 0.0)
            entry['support_class2'] = report_dict.get('1', {}).get('support', 0.0)
            entry['support_class3'] = report_dict.get('2', {}).get('support', 0.0)
            entry['time0'] = (y_train == 0).sum()
            entry['time1'] = (y_train == 1).sum()
            entry['time2'] = (y_train == 2).sum()
            entry['f1_smote'] = f1_score(y_resampled, y_pred_smote, average='macro')
            entry['report_dict'] = pd.DataFrame(report_dict).T.to_dict()

        ozone_result.append(entry)

    return ozone_result


if __name__ == '__main__':
    # Load the data
    ozone_api = pd.read_parquet('../../models/output/seatunnel_prepare_to_train.parquet')
    ozone_api = percentage_smell(ozone_api)

    # Check the data time out of 2024
    more_2024, less_2024 = check_time_df(ozone_api)

    # Calculate the percentiles of the total_time_hours column
    ozone_percentiles = calculate_percentiles(ozone_api['total_time_hours'])
    ozone_percentile_combinations = set_index_combinations_percentiles(ozone_percentiles)
    ozone_time_points = table_time_fix_percentile(ozone_percentile_combinations)
    ozone_time_classes = divide_time_class(ozone_api, ozone_time_points)
    ozone_good, ozone_bad = filter_dfs_by_class_count(ozone_time_classes)

    ozone_result = split_data_x_y(ozone_good, more_2024, less_2024)

    # Convert the list of dictionaries into a DataFrame
    ozone_result_df = pd.DataFrame(ozone_result)

    with open('../../models/output/seatunnel_teat_model_time_class3_somte_newversion.parquet', 'wb') as f:
        joblib.dump(ozone_result_df, f)
        print("save file Done!")
        print(ozone_result_df)  # This will now print the DataFrame containing the concatenated results
