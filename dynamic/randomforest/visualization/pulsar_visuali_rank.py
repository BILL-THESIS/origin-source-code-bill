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
    plt.savefig(f'../../output/seatunnel_f1_score_scatter{name}.png')
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
    plt.savefig(f'../../output/seatunnel_f1_score_bar{title}.png')
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
        plt.savefig(f'../../output/seatunnel_scatter_group{idx + 1}.png')
        # plt.show()

        time.sleep(2)


if __name__ == "__main__":
    file_significant = "../../output/pulsar_all_status_significant.pkl"
    file_feature_importances = '../../output/pulsar_feature_importances.pkl'
    file_group_smell = "../../output/pulsar_rdf_quantile_all.pkl"
    file_each_smell = "../../output/pulsar_rdf_quantile_each_smell.pkl"
    file_main_group = "../../output/pulsar_correlation_main_group_7.pkl"

    data_qr1 = pd.read_pickle(file_significant)
    data_feature_importances = pd.read_pickle(file_feature_importances)
    data_feature_importances["feature_set"] = data_feature_importances["feature_set"].apply(literal_eval)
    data_group_smell = pd.read_pickle(file_group_smell)
    data_each_smell = pd.read_pickle(file_each_smell)
    data_mian_group = pd.read_pickle(file_main_group)
    data_original = pd.read_pickle("../../output/pulsar_compare.pkl")

    group_by_feature_set = data_feature_importances.groupby('feature')['importance'].agg(['mean', 'max', 'min'])
    group_by_feature_set.columns = ['mean', 'max', 'min']
    group_by_feature_set.reset_index(inplace=True)

    groups_factor = [group_by_feature_set[group_by_feature_set["feature"].isin(data_mian_group[i])] for i in range(7)]
    groups_factor_set1, groups_factor_set2, groups_factor_set3, groups_factor_set4, groups_factor_set5, groups_factor_set6, groups_factor_set7 = groups_factor

    # groups_factor_set7.sort_values(by='mean', ascending=False, inplace=True)

    groups = [data_each_smell[data_each_smell["features"].isin(data_mian_group[i])] for i in
              range(len(data_mian_group))]
    groups = [group.merge(data_qr1, left_on='features', right_on='metric', how='left') for group in groups]
    group1, group2, group3, group4, group5, group6, group7 = groups

    for group in [group1, group2, group3, group4, group5, group6, group7]:
        group['rank_f1'] = group['test_f1'].rank(ascending=False)
        group['rank_roc'] = group['test_roc_auc'].rank(ascending=False)
        group['rank_d'] = group['d_value'].rank(ascending=False)

    group1 = group1[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group2 = group2[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group3 = group3[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group4 = group4[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group5 = group5[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group6 = group6[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group7 = group7[['features', 'rank_f1', 'rank_roc', 'rank_d']]

    group_join = pd.concat([group1, group2, group3, group4, group5, group6, group7], axis=0)

    grop_70_percen = data_group_smell.loc[data_group_smell['test_f1'] >= 0.7]

    data_rank_group_f1 = grop_70_percen['features'].apply(
        lambda s: [group_join.set_index('features').loc[x, 'rank_f1'] for x in s])
    data_rank_group_f1 = pd.concat([grop_70_percen['features'], grop_70_percen['test_f1'], data_rank_group_f1], axis=1)
    df_split_f1 = pd.DataFrame(data_rank_group_f1.iloc[:, 2].tolist(), columns=["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
                               index=data_rank_group_f1.index)
    data_rank_group_f1 = pd.concat([data_rank_group_f1, df_split_f1], axis=1)
    data_rank_group_f1['sum'] = data_rank_group_f1['sum'] = data_rank_group_f1[["G1", "G2", "G3", "G4", "G5", "G6", "G7"]].sum(axis=1)

    data_rank_group_auc = grop_70_percen['features'].apply(
        lambda s: [group_join.set_index('features').loc[x, 'rank_roc'] for x in s])
    data_rank_group_auc = pd.concat([grop_70_percen['features'], grop_70_percen['test_roc_auc'], data_rank_group_auc],
                                    axis=1)
    df_split_auc = pd.DataFrame(data_rank_group_auc.iloc[:, 2].tolist(), columns=["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
                                index=data_rank_group_auc.index)
    data_rank_group_auc = pd.concat([data_rank_group_auc, df_split_auc], axis=1)

    data_rank_group_d = grop_70_percen['features'].apply(
        lambda s: [group_join.set_index('features').loc[x, 'rank_d'] for x in s])
    data_rank_group_d = pd.concat([grop_70_percen['features'], data_rank_group_d], axis=1)
    df_split_d = pd.DataFrame(data_rank_group_d.iloc[:, 1].tolist(), columns=["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
                              index=data_rank_group_d.index)
    data_rank_group_d = pd.concat([data_rank_group_d, df_split_d], axis=1)
