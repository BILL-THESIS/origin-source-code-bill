import pandas as pd
import seaborn as sns
import plotly.express as px
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import time
from ast import literal_eval


def scatter_plot(data, name):
    plt.figure(figsize=(35, 12))
    data['features'] = data['features'].astype(str)
    plt.scatter(data['features'], data['test_f1'], alpha=0.5, c=data['test_f1'], cmap='viridis')
    plt.title('Scatter Plot of Accuracy vs X')
    plt.xlabel('features')
    plt.xticks(rotation=90)
    plt.ylabel('f1 score')
    plt.colorbar(label='f1 score')
    plt.tight_layout()
    plt.savefig(f'../../output_resample/seatunnel_f1_score_scatter{name}.png')
    plt.show()


def plot_bar(data, title):
    # หาค่า X ที่ให้ f1 สูงสุด
    best_result = data.loc[data['test_f1'].idxmax()]
    print(f"Best X: {best_result['features']}, Best f1: {best_result['test_f1']:.4f}")
    # Plot with the best point
    plt.figure(figsize=(35, 12))

    data['features'] = data['features'].astype(str)

    plt.plot(data['features'], data['test_f1'], marker='o')
    plt.scatter(best_result['features'], best_result['test_f1'], color='red', label='Best X')

    # เพิ่มเส้นแนวนอนที่ y = 0.7
    plt.axhline(y=0.7, color='red', linestyle='--', label='Threshold (y = 0.7)')
    plt.title('f1 score by features')
    plt.xlabel('features')
    plt.xticks(rotation=90)
    plt.ylabel('f1 score')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'../../output_resample/seatunnel_f1_score_bar{title}.png')
    plt.show()


def compare_group(data_group, data_each_smell):
    # สร้าง data frame ใหม่ จาก เปรียบเทียบชุดข้อมูลแบบกลุ่ม และ แยกกลุ่ม
    # สร้าง DataFrame ใหม่สำหรับผลลัพธ์
    results = []
    for _, row in data_group.iterrows():
        # ดึงข้อมูลจากแต่ละแถว
        fit_time, score_time, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, features = row

        group_data = [
            [fit_time, score_time, test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, features]]

        for name in features:  # features เป็น iterable เช่น list
            match = data_each_smell[data_each_smell["features"] == name]
            if not match.empty:
                group_data.append(match.iloc[0].tolist())

        final_df = pd.DataFrame(group_data, columns=data_each_smell.columns)
        results.append(final_df)

    return results


def plot_scatter(data_compare, important_factures):
    # สร้างกราฟหลาย ๆ รูปใน 1 แผ่น (Grid 2x5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    # สร้าง Scatter Plot พร้อมการเปลี่ยนสี
    for idx, df in enumerate(data_compare):

        # เปลี่ยนรูปแบบการวน loop เพื่อวางกราฟในแต่ละ subplot
        # for idx, (df, ax) in enumerate(zip(data_compare, axes.flatten())):

        plt.figure(figsize=(12, 10))

        # ค้นหาค่ามากที่สุดในแต่ละกลุ่ม
        max_idx = df["test_f1"].idxmax()

        for i, row in df.iterrows():
            # ตรวจสอบว่าเป็นข้อมูลสำคัญหรือไม่
            if row["features"] in important_factures:
                color = "orange"
            # ตรวจสอบว่าคือข้อมูลที่มีค่ามากที่สุดหรือไม่
            elif i == max_idx:
                color = "red"
            else:
                color = "blue"

            # วาดจุดแต่ละจุดในกราฟ
            plt.scatter(row["test_f1"], row["test_roc_auc"], c=color, label=row["features"] if i == 0 else "")

            # เพิ่ม labels ให้กับจุด
            plt.text(row["test_f1"], row["test_roc_auc"], str(row["features"]), fontsize=9, ha="right")

        plt.title(f"Scatter Plot for Group {idx + 1}")
        plt.xlabel("f1 score")
        plt.ylabel("roc auc score")
        plt.legend(['Max Value"', "Important feature", "others"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'../../output_resample/seatunnel_scatter_group{idx + 1}.png')
        # plt.show()

        time.sleep(2)


if __name__ == "__main__":
    file_significant = "../../output/output_resample/pulsar_all_status_significant.pkl"
    file_feature_importances = '../../output/output_resample/pulsar_feature_importances.pkl'
    file_group_smell = "../../output/output_resample/pulsar_rdf_quantile_all.pkl"
    file_each_smell = "../../output/output_resample/pulsar_rdf_quantile_each_smell.pkl"
    file_main_group = "../../output/output_resample/pulsar_correlation_main_group_7.pkl"

    data_qr1 = pd.read_pickle(file_significant)
    data_feature_importances = pd.read_pickle(file_feature_importances)
    data_feature_importances["feature_set"] = data_feature_importances["feature_set"].apply(literal_eval)
    data_group_smell = pd.read_pickle(file_group_smell)
    data_each_smell = pd.read_pickle(file_each_smell)
    data_mian_group = pd.read_pickle(file_main_group)
    data_original = pd.read_pickle("../../output/output_resample/pulsar_compare.pkl")

    # data_each_smell['qr1'] = data_each_smell['features'].isin(data_qr1['metric'])

    grop_80_percen = data_group_smell.loc[data_group_smell['test_f1'] >= 0.8]
    grop_70_percen = data_group_smell.loc[(data_group_smell['test_f1'] >= 0.7) & (data_group_smell['test_f1'] < 0.8)]
    grop_60_percen = data_group_smell.loc[(data_group_smell['test_f1'] >= 0.6) & (data_group_smell['test_f1'] < 0.7)]
    grop_50_percen = data_group_smell.loc[data_group_smell['test_f1'] < 0.6]

    data_compare = compare_group(grop_70_percen, data_each_smell)

    important_feature = data_qr1["metric"].tolist()

    groups = [data_each_smell[data_each_smell["features"].isin(data_mian_group[i])] for i in range(7)]
    groups = [group.merge(data_qr1, left_on='features', right_on='metric', how='left') for group in groups]
    group1, group2, group3, group4, group5, group6, group7 = groups

    group_max_values = []
    for group in [group1, group2, group3, group4, group5, group6, group7]:
        max_f1 = group.loc[group['test_f1'].idxmax()]
        max_d = group.loc[group['d_value'].idxmax()]
        group_max_values.append((max_f1['features'], max_d['features']))

    group1_max_f1_value, group1_max_d_value = group_max_values[0]
    group2_max_f1_value, group2_max_d_value = group_max_values[1]
    group3_max_f1_value, group3_max_d_value = group_max_values[2]
    group4_max_f1_value, group4_max_d_value = group_max_values[3]
    group5_max_f1_value, group5_max_d_value = group_max_values[4]
    group6_max_f1_value, group6_max_d_value = group_max_values[5]
    group7_max_f1_value, group7_max_d_value = group_max_values[6]

    list_max_f1 = [group1_max_f1_value, group2_max_f1_value, group3_max_f1_value, group4_max_f1_value,
                   group5_max_f1_value, group6_max_f1_value, group7_max_f1_value]
    list_max_d_values = [group1_max_d_value, group2_max_d_value, group3_max_d_value, group4_max_d_value,
                         group5_max_d_value, group6_max_d_value, group7_max_d_value]

    where_conbi_maxf1 = data_group_smell[
        data_group_smell["features"].apply(lambda x: x == [
            group1_max_f1_value,
            group2_max_f1_value,
            group3_max_f1_value,
            group4_max_f1_value,
            group5_max_f1_value,
            group6_max_f1_value,
            group7_max_f1_value
        ])
    ]

    where_conbi_maxd = data_group_smell[
        data_group_smell["features"].apply(lambda x: x == [
            group1_max_d_value,
            group2_max_d_value,
            group3_max_d_value,
            group4_max_d_value,
            group5_max_d_value,
            group6_max_d_value,
            group7_max_d_value
        ])
    ]

    where_data_importance = data_feature_importances[
        data_feature_importances["feature_set"].apply(lambda x: x == list_max_f1)]

    # group_by_feature_set = data_feature_importances.groupby('feature').mean()

    data_feature_importances['rank'] = data_feature_importances.groupby('feature')['importance'].rank()

    group_by_feature_set_rank = data_feature_importances.groupby('feature')['rank'].agg(['mean', 'max', 'min'])
    group_by_feature_set_rank.columns = ['rank_mean', 'rank_max', 'rank_min']

    group_by_feature_set = data_feature_importances.groupby('feature')['importance'].agg(['mean', 'max', 'min'])
    group_by_feature_set.columns = ['mean', 'max', 'min']
    group_by_feature_set.reset_index(inplace=True)

    data_smell_time = data_original[['java:S1192_created',
                                     'java:S2176_created',
                                     'java:S3398_created',
                                     'java:S2177_created', 'total_time']]

    groups_factor = [group_by_feature_set[group_by_feature_set["feature"].isin(data_mian_group[i])] for i in range(7)]
    groups_factor_set1, groups_factor_set2, groups_factor_set3, groups_factor_set4, groups_factor_set5, groups_factor_set6, groups_factor_set7 = groups_factor

    for group in groups_factor:
        group.sort_values(by='mean', ascending=False, inplace=True)

    list_max_feacter = [group['feature'].iloc[0] for group in groups_factor]

    where_data_facter_importance = data_group_smell[
        data_group_smell["features"].apply(lambda x: x == ['java:S117_created',
                                                           'java:S1191_created',
                                                           'java:S4449_created',
                                                           'java:S2157_created',
                                                           'java:S4034_created',
                                                           'java:S2175_created',
                                                           'java:S5659_created'])
    ]
