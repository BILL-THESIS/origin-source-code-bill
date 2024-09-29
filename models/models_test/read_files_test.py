import os
from typing import Tuple

import joblib
import pandas as pd


def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test


def calculate_time_counts_std(data):
    data['std_counts'] = data[['time0', 'time1', 'time2']].std(axis=1)
    return data


def check_time_df(df: pd.DataFrame, date_time) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into two based on whether the 'merged_at' year is >= 2024."""

    df['merged_at'] = pd.to_datetime(df['merged_at'], errors='coerce')

    # Define cutoff date with UTC timezone
    cutoff_date = pd.Timestamp(date_time, tz='UTC')
    after_2024 = df[df['merged_at'] >= cutoff_date]
    before_2024 = df[df['merged_at'] < cutoff_date]

    return before_2024, after_2024


if __name__ == '__main__':
    ozone_test = read_file_joblib('../../models/output/ozone_teat_model_time_class3_somte_newversion26.parquet')
    seatunnel_test = read_file_joblib('../../models/output/seatunnel_teat_model_time_class3_somte_newversion26.parquet')

    ozone_api = pd.read_parquet('../../models/output/ozone_prepare_to_train_new26.parquet')
    pulsar_api = pd.read_parquet('../../models/output/seatunnel_prepare_to_train_new26.parquet')
    seatunnal_api = pd.read_parquet('../../models/output/seatunnel_prepare_to_train_new26.parquet')

    # Check the data time out of 2024
    ozone_api_before_2024, ozone_api_after_2024 = check_time_df(ozone_api, '2024-02-01')
    pulsar_api_before_2024, pulsar_api_after_2024 = check_time_df(pulsar_api, '2024-02-01')
    seatunnal_api_before_2024, seatunnal_api_after_2024 = check_time_df(seatunnal_api, '2024-04-01')

    ozone_test_std = calculate_time_counts_std(ozone_test)
    ozone_test_std.drop(['report_dict'], axis=1, inplace=True)
    ozone_test_std.rename(columns={'precision_class0': 'precision_class1',
                                   'precision_class1': 'precision_class2',
                                   'precision_class2': 'precision_class3'},
                          inplace=True)

    ozone_more_50_percent = ozone_test_std[(ozone_test_std['f1_smote'] >= 0.50)]
    ozone_more_60_percent = ozone_test_std[(ozone_test_std['f1_smote'] >= 0.60) & (ozone_test_std['f1_smote'] <= 0.77)]
    ozone_test_std_sort = ozone_test_std[(ozone_test_std['std_counts'] >= 100) & (ozone_test_std['f1_smote'] >= 0.70)]
