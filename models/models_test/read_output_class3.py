import os
import joblib


def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test


if __name__ == '__main__':
    ozone_class3 = read_file_joblib('../../models/output/ozone_test_model_timeclass3_16Sep.parquet')
    ozone_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    ozone_class3_svc = read_file_joblib('../../models/output/ozone_test_model_svc_timeclass3_28Sep.parquet')
    ozone_class3_svc.drop(columns=['report_dict'], axis=1, inplace=True)

    seatunnal_class3 = read_file_joblib('../../models/output/seatunnel_test_model_timeclass3_16Sep.parquet')
    seatunnal_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    pulsar_class3 = read_file_joblib('../../models/output/pulsarl_test_model_timeclass3_16Sep.parquet')
    pulsar_class3.drop(columns=['report_dict'], axis=1, inplace=True)

    pulsar_class3_svc = read_file_joblib('../../models/output/pulsar_test_model_svc_timeclass3_28Sep.parquet')
    pulsar_class3_svc.drop(columns=['report_dict'], axis=1, inplace=True)

    ozone_count_std_f1_smote_60_class3 = ozone_class3[
        (ozone_class3['f1_smote_class0'] >= 0.6) & (ozone_class3['f1_smote_class1'] >= 0.6) & (
                ozone_class3['f1_smote_class2'] >= 0.6)]
    ozone_out_f1_smote_class3 = ozone_count_std_f1_smote_60_class3[(ozone_count_std_f1_smote_60_class3['f1_smote'] <= 0.75)]

    seatunnal_count_std_f1_smote_60_class3 = seatunnal_class3[
        (seatunnal_class3['f1_smote_class0'] >= 0.6) & (seatunnal_class3['f1_smote_class1'] >= 0.6) & (
                seatunnal_class3['f1_smote_class2'] >= 0.6)]
    seatunnal_out_f1_smote_class3 = seatunnal_count_std_f1_smote_60_class3[(seatunnal_count_std_f1_smote_60_class3['f1_smote'] <= 0.75)]

    pulsar_count_std_f1_smote_60_class3 = pulsar_class3[
        (pulsar_class3['f1_smote_class0'] >= 0.6) & (pulsar_class3['f1_smote_class1'] >= 0.6) & (
                pulsar_class3['f1_smote_class2'] >= 0.6)]
    pulsar_out_f1_smote_class3 = pulsar_count_std_f1_smote_60_class3[(pulsar_count_std_f1_smote_60_class3['f1_smote'] <= 0.75)]

    ozone_out_f1_normal =  ozone_class3[(ozone_class3['f1_smote'] >= 0.6)]