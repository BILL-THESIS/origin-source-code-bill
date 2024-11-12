import os

import joblib
import pandas as pd

def read_parquet_files(path_files):
    file_tests = []
    for path_file in path_files:
        try:
            df = pd.read_parquet(os.path.join(path_file))
            file_tests.append((os.path.basename(path_file), df))
        except Exception as e:
            with open(os.path.join(path_file), 'rb') as f:
                file_tests.append(joblib.load(f))
                file_tests.append((os.path.basename(path_file)))
            print(f"Error loading {path_file}: {e}")
    return file_tests

if __name__ == '__main__':
    path = [
        '../output/seatunnal_20col.parquet',
        '../output/seatunnel_all_information.parquet',
        '../output/seatunnel_counts_time_class.parquet',
        '../output/seatunnel_filtered_robust_outlier.parquet',
        '../output/seatunnel_filtered_robust_outlier_new.parquet',
        '../output/seatunnel_GBC_class_time3_smote.parquet',
        '../output/seatunnel_prepare_to_train.parquet',
        '../output/seatunnel_prepare_to_train_new26.parquet',
        '../output/seatunnel_prepare_to_train_newversion_9Sep.parquet',
        '../output/seatunnel_result_std_class_time3.parquet',
        '../output/seatunnel_teat_model_somte_newversion_class2_18Sep.parquet',
        '../output/seatunnel_teat_model_time_class3_somte_newversion.parquet',
        '../output/seatunnel_teat_model_time_class3_somte_newversion26.parquet',
        '../output/seatunnel_test_model_timeclass3_16Sep.parquet',
    ]

    seatunnal = read_parquet_files(path)
