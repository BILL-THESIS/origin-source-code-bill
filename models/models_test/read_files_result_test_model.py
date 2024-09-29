import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test

def liners_regression(df, project_name):

    return plt.show()


if __name__ == '__main__':
    ozone_class3 = read_file_joblib('../../models/output/ozone_test_model_timeclass3_12Sep.parquet')
    ozone_class2 = read_file_joblib('../../models/output/ozone_teat_model_time_class3_somte_newversion_class2.parquet')

    class_2 = ozone_class2.drop(columns=['precision_class0',
                                         'precision_class1', 'recall_class0', 'recall_class1', 'support_class0',
                                         'support_class1', 'report_dict'], axis=0)

    class_3 = ozone_class3.drop(columns=['precision_class0', 'precision_class1', 'precision_class2',
                                         'recall_class1', 'recall_class2', 'recall_class3', 'support_class1',
                                         'support_class2', 'support_class3', 'report_dict'], axis=0)

    class_3.rename(columns={'f1_score_class1': 'f1_score_class0',
                            'f1_score_class2': 'f1_score_class1',
                            'f1_score_class3': 'f1_score_class2'}, inplace=True)

    class_3 = class_3[['index_time01', 'index_time12', 'time01', 'time12', 'f1_smote', 'f1_normal',
                       'f1_score_class0', 'f1_score_class1', 'f1_score_class2',
                       'time0', 'time1', 'time2']]

    mask = (class_3['f1_smote'] >= 0.7) & (class_3['f1_smote'] < 0.8)
