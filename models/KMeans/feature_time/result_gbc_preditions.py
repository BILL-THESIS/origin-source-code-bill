import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import seaborn.objects as so
import time


if __name__ == '__main__':
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    time_str = str(start_time_gmt)

    # input files
    file_features = r'../../output/over_all_GBC_matrix1_2024-05-27 04:58:08.parquet'

    df = pd.read_parquet(file_features)

    f1_maximum = df.sort_values(by='f1_macro_q3', ascending=False).head(1)
    f1_minimum = df.sort_values(by='f1_macro_q3', ascending=True).head(1)

    compare_f1_macro_gbc = pd.concat([f1_maximum, f1_minimum])

    all_15col_combinations = df.iloc[28657]

    different_f1_macro_gbc = f1_maximum['f1_macro_q3'].values - f1_minimum['f1_macro_q3'].values

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))
