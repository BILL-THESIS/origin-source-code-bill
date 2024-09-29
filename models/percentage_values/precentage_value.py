import os
import joblib


def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test


def sort_values_time(df):
    for index, row in df.iterrows():
        time = row[['time0', 'time1', 'time2']]
        soret_time = time.sort_values(ascending=False)
        print(soret_time.iloc[0])

        # The most value in the time0, time1, time2
        most_value = soret_time.iloc[0]

        # The difference between the second value and the there value
        sumtime = abs((soret_time.iloc[1] + soret_time.iloc[2]) + (soret_time.iloc[1] - soret_time.iloc[2]))

        # The percentage difference between the most value
        percen10 = abs(soret_time.iloc[0] * 0.1)
        percen15 = abs(soret_time.iloc[0] * 0.15)
        percen20 = abs(soret_time.iloc[0] * 0.2)
        percen25 = abs(soret_time.iloc[0] * 0.25)
        percen30 = abs(soret_time.iloc[0] * 0.3)

        # add new columns sum
        df.loc[index, 'sum_time'] = sumtime
        df.loc[index, '10percen_diff_time'] = percen10
        df.loc[index, '15percen_diff_time'] = percen15
        df.loc[index, '20percen_diff_time'] = percen20
        df.loc[index, '25percen_diff_time'] = percen25
        df.loc[index, '30percen_diff_time'] = percen30

        df.loc[index, 'diff_time10'] = abs((most_value - sumtime) / percen10)
        df.loc[index, 'diff_time15'] = abs((most_value - sumtime) / percen15)
        df.loc[index, 'diff_time20'] = abs((most_value - sumtime) / percen20)
        df.loc[index, 'diff_time25'] = abs((most_value - sumtime) / percen25)
        df.loc[index, 'diff_time30'] = abs((most_value - sumtime) / percen30)

    return df


if __name__ == '__main__':
    ozone_class3 = read_file_joblib('../../models/output/ozone_test_model_timeclass3_16Sep.parquet')
    ozone_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    seatunnal_class3 = read_file_joblib('../../models/output/seatunnel_test_model_timeclass3_16Sep.parquet')
    seatunnal_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    pulsar_class3 = read_file_joblib('../../models/output/pulsarl_test_model_timeclass3_16Sep.parquet')
    pulsar_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    ozone_check_values = sort_values_time(ozone_class3)
    seatunnal_check_values = sort_values_time(seatunnal_class3)
    pulsar_check_values = sort_values_time(pulsar_class3)

