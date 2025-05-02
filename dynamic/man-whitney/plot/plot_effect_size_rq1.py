import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fill_col(df: pd.DataFrame):
    cols_to_fill = [col for col in df.columns if 'java:' in col and '_created' in col]
    return cols_to_fill

if __name__ == "__main__":
    # โหลดไฟล์
    # project_name = "seatunnal"
    # project_name = "pulsar"
    project_name = "ozone"
    file_path = f'../../output/{project_name}_cut_time.pkl'
    df = pd.read_pickle(file_path)
    time_delta = pd.to_timedelta(df['total_time'])
    df['hours'] = time_delta.dt.total_seconds() / 3600

    se_col = fill_col(df)
    df_se = df[se_col]
    df_se = pd.concat([df_se, df['hours'] ], axis=1)
    df_se.fillna("0")
    df_se_drop = df_se.dropna(axis=1, how='all')

    data = df_se_drop['hours']

    # คำนวณ Q1 และ Q3
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # แบ่งข้อมูล
    group1 = data[data < q1]
    group2 = data[data > q3]

    # สร้าง box plot
    plt.boxplot([group1, group2],
                labels=['Time Modification < Q1', 'Time Modification > Q3'],
                showfliers=False)
    plt.title(f'Boxplot of {project_name} \n Time Modification < Q1 and  Time Modification > Q3')
    plt.ylabel('Total Time Modifications (hours)')
    plt.tight_layout()
    plt.savefig(f'{project_name}_cut_time_boxplot_outliner.png')
    plt.show()






