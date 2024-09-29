import os
import joblib

def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test

if __name__ == '__main__':
    ozone_class2 = read_file_joblib('../../models/output/ozone_teat_model_somte_newversion_class2_18Sep.parquet')
    ozone_class2.drop(columns=['report_dict'], axis=1, inplace=True)
    ozone_class2['std_counts'] = ozone_class2[['time0', 'time1']].std(axis=1)
    ozone_class2['std_f1'] = ozone_class2[['f1_score_class0', 'f1_score_class1']].std(axis=1)
    ozone_class2['std_f1_smote'] = ozone_class2[['f1_smote_class0', 'f1_smote_class1']].std(axis=1)

    seatunnal_class2 = read_file_joblib('../../models/output/seatunnel_teat_model_somte_newversion_class2_18Sep.parquet')
    seatunnal_class2.drop(columns=['report_dict'], axis=1, inplace=True)
    seatunnal_class2['std_counts'] = seatunnal_class2[['time0', 'time1']].std(axis=1)
    seatunnal_class2['std_f1'] = seatunnal_class2[['f1_score_class0', 'f1_score_class1']].std(axis=1)
    seatunnal_class2['std_f1_smote'] = seatunnal_class2[['f1_smote_class0', 'f1_smote_class1']].std(axis=1)

    pulsar_class2 = read_file_joblib('../../models/output/pulsar_teat_model_somte_newversion_class2_18Sep.parquet')
    pulsar_class2.drop(columns=['report_dict'], axis=1, inplace=True)
    pulsar_class2['std_counts'] = pulsar_class2[['time0', 'time1']].std(axis=1)
    pulsar_class2['std_f1'] = pulsar_class2[['f1_score_class0', 'f1_score_class1']].std(axis=1)
    pulsar_class2['std_f1_smote'] = pulsar_class2[['f1_smote_class0', 'f1_smote_class1']].std(axis=1)

    ozone_count_std = ozone_class2[(ozone_class2['f1_score_class0'] >= 0.4) & (ozone_class2['f1_score_class1'] >= 0.4)]
    ozone_count_std_f1_60 = ozone_class2[(ozone_class2['f1_smote_class0'] >= 0.6) & (ozone_class2['f1_smote_class1'] >= 0.6)]
    ozone_out_f1_smote = ozone_count_std_f1_60[(ozone_count_std_f1_60['f1_smote'] <=0.75)]

    seatunnal_count_std = seatunnal_class2[(seatunnal_class2['f1_score_class0'] >= 0.4) & (seatunnal_class2['f1_score_class1'] >= 0.4)]
    seatunnal_count_std_f1_60 = seatunnal_class2[(seatunnal_class2['f1_smote_class0'] >= 0.6) & (seatunnal_class2['f1_smote_class1'] >= 0.6)]
    seatunnal_out_f1_smote = seatunnal_count_std_f1_60[(seatunnal_count_std_f1_60['f1_smote'] <=0.75)]

    pulsar_count_std = pulsar_class2[(pulsar_class2['f1_score_class0'] >= 0.4) & (pulsar_class2['f1_score_class1'] >= 0.4)]
    pulsar_count_std_f1_60 = pulsar_class2[(pulsar_class2['f1_smote_class0'] >= 0.6) & (pulsar_class2['f1_smote_class1'] >= 0.6)]
    pulsar_out_f1_smote = pulsar_count_std_f1_60[(pulsar_count_std_f1_60['f1_smote'] <=0.75)]
