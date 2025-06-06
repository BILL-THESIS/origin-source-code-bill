import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ฟังก์ชันเลือกสี
# กำหนดสีตามเงื่อนไข
def get_color_rf(f1):
    return 'green' if f1 > 0.7 else 'orange'


def get_color_lgbm(f1):
    return 'green' if f1 > 0.7 else 'pink'


def clean_created(feature_list):
    return [item.replace('_created', '') for item in feature_list]

def plot_compare_model(df, title, project_name):

    x = np.arange(len(df['feature_group']))

    # กำหนดสีสำหรับแต่ละจุด
    colors_rf = [get_color_rf(f) for f in df['f1_rf']]
    colors_lgbm = [get_color_lgbm(f) for f in df['f1_lgbm']]

    # วาดจุด
    plt.figure(figsize=(10, 6))
    plt.scatter(x, df['f1_rf'], color=colors_rf, label='Random Forest', s=100, marker='o')
    plt.scatter(x, df['f1_lgbm'], color=colors_lgbm, label='LightGBM', s=100, marker='x')

    # รายละเอียดกราฟ
    plt.xticks(x, df['feature_group'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature Group')
    plt.ylabel('F1 Score')
    plt.title(f'{title} F1 Score Comparison')
    plt.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # เส้นบอกค่าขีด 0.7
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # แสดงชื่อ feature_group เฉพาะจุดสีเขียว
    for i in range(len(x)):
        # จุด RF
        if df['f1_rf'][i] > 0.7:
            plt.text(x[i] - 0.15, df['f1_rf'][i] + 0.02, df['feature_group'][i], color='green', fontsize=9)
        # จุด LGBM
        if df['f1_lgbm'][i] > 0.7:
            plt.text(x[i] + 0.05, df['f1_lgbm'][i] + 0.02, df['feature_group'][i], color='green', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{project_name}_f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_compare_model_more(df, title, project_name):

    x = np.arange(len(df['feature_group']))

    # กำหนดสีสำหรับแต่ละจุด
    colors_rf = [get_color_rf(f) for f in df['f1_rf']]
    colors_lgbm = [get_color_lgbm(f) for f in df['f1_lgbm']]

    # วาดจุด
    plt.figure(figsize=(10, 6))
    plt.scatter(x, df['f1_rf'], color=colors_rf, label='Random Forest', s=100, marker='o')
    plt.scatter(x, df['f1_lgbm'], color=colors_lgbm, label='LightGBM', s=100, marker='x')

    # รายละเอียดกราฟ
    plt.xticks(x, df['feature_group'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature Group')
    plt.ylabel('F1 Score')
    plt.title(f'{title} F1 Score Comparison')
    plt.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # เส้นบอกค่าขีด 0.7
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()
    plt.savefig(f'{project_name}_f1_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_pulsar(dataframe):
    summary = dataframe.groupby("sum_rf").agg(
        total_group_features=('feature_group', 'nunique'),
        total_features_above_80=('f1_rf', lambda x: (x > 0.6).sum()),
        percentage=('f1_rf', lambda x: (x > 0.6).sum() / len(x)),
        f1_min=('f1_rf', 'min')  # ค่า F1 ต่ำสุดในแต่ละกลุ่ม
    ).reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    summary.columns = [
        "ผลรวมลำดับ",
        "จำนวนฟีเจอร์ของลำดับ",
        "จำนวนฟีเจอร์ที่คาดหวัง",
        "เปอร์เซ็นต์ที่คาดหวัง",
        "F1 ต่ำสุดของผลรวมลำดับ"
    ]

    # เพิ่มคอลัมน์ "Metricx" = "F1"
    summary.insert(0, "Metricx", "F1")

    # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    summary = summary.sort_values(by="ผลรวมลำดับ")

    return summary

def generate_summary(dataframe):
    summary = dataframe.groupby("sum_lgbm").agg(
        total_group_features=('feature_group', 'nunique'),
        total_features_above_80=('f1_lgbm', lambda x: (x > 0.7).sum()),
        percentage=('f1_lgbm', lambda x: (x > 0.7).sum() / len(x)),
        f1_min=('f1_lgbm', 'min')  # ค่า F1 ต่ำสุดในแต่ละกลุ่ม
    ).reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    summary.columns = [
        "ผลรวมลำดับ",
        "จำนวนฟีเจอร์ของลำดับ",
        "จำนวนฟีเจอร์ที่คาดหวัง",
        "เปอร์เซ็นต์ที่คาดหวัง",
        "F1 ต่ำสุดของผลรวมลำดับ"
    ]

    # เพิ่มคอลัมน์ "Metricx" = "F1"
    summary.insert(0, "Metricx", "F1")

    # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    summary = summary.sort_values(by="ผลรวมลำดับ")

    return summary


def generate_summary_rf(dataframe):
    dataframe['rank_rf'] = dataframe['rank_rf'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    dataframe['feature_group'] = dataframe['feature_group'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    summary = dataframe.groupby("rank_rf").agg(
        total_group_features=('feature_group', 'nunique'),
        total_features_above_80=('f1_rf', lambda x: (x > 0.7).sum()),  # จำนวนที่ F1 > 0.7
        percentage=('f1_rf', lambda x: (x > 0.7).sum() / len(x)),  # เปอร์เซ็นต์ที่ F1 > 0.7
        f1_min=('f1_rf', 'min')  # ค่า F1 ต่ำสุดในแต่ละกลุ่ม
    ).reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    summary.columns = [
        "ผลรวมลำดับ",
        "จำนวนฟีเจอร์ของลำดับ",
        "จำนวนฟีเจอร์ที่คาดหวัง",
        "เปอร์เซ็นต์ที่คาดหวัง",
        "F1 ต่ำสุดของผลรวมลำดับ"
    ]

    # เพิ่มคอลัมน์ "Metricx" = "F1"
    summary.insert(0, "Metricx", "F1")

    # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    summary = summary.sort_values(by="ผลรวมลำดับ")

    return summary

def generate_summary_lgbm(dataframe):
    dataframe['rank_rf'] = dataframe['rank_rf'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    dataframe['feature_group'] = dataframe['feature_group'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    summary = dataframe.groupby("rank_lgbm").agg(
        total_group_features=('feature_group', 'nunique'),
        total_features_above_80=('f1_lgbm', lambda x: (x > 0.7).sum()),
        percentage=('f1_lgbm', lambda x: (x > 0.7).sum() / len(x)),
        f1_min=('f1_lgbm', 'min')  # ค่า F1 ต่ำสุดในแต่ละกลุ่ม
    ).reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    summary.columns = [
        "ผลรวมลำดับ",
        "จำนวนฟีเจอร์ของลำดับ",
        "จำนวนฟีเจอร์ที่คาดหวัง",
        "เปอร์เซ็นต์ที่คาดหวัง",
        "F1 ต่ำสุดของผลรวมลำดับ"
    ]

    # เพิ่มคอลัมน์ "Metricx" = "F1"
    summary.insert(0, "Metricx", "F1")

    # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    summary = summary.sort_values(by="ผลรวมลำดับ")

    return summary


if __name__ == '__main__':
    ozone_rf = pd.read_pickle("rank_f1/output_rank/ozone_optuna_result_rank_rf.pkl")
    ozone_lgbm = pd.read_pickle("rank_f1/output_rank/ozone_optuna_result_rank_lgbm.pkl")

    pulsar_rf = pd.read_pickle("rank_f1/output_rank/pulsar_optuna_result_rank_rf.pkl")
    pulsar_lgbm = pd.read_pickle("rank_f1/output_rank/pulsar_optuna_result_rank_lgbm.pkl")
    # pulsar_rf['feature_group'] = pulsar_rf['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)
    # pulsar_lgbm['feature_group'] = pulsar_lgbm['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)


    # ozone = pd.merge(ozone_rf, ozone_lgbm, how='left', on='feature_group', suffixes=('_rf', '_lgbm'))
    # pulsar = pd.merge(pulsar_rf, pulsar_lgbm, how='left', on='feature_group', suffixes=('_rf', '_lgbm'))

    # ozone_rank_rf = ozone_rf[ozone_rf['f1'] >= 0.7]
    # ozone_rank_lgbm = ozone_lgbm[ozone_lgbm['f1'] >= 0.7]

    #
    # pulsar_rank_rf = pulsar_rf[pulsar_rf['f1'] >= 0.7]
    # pulsar_rank_lgbm = pulsar_lgbm[pulsar_lgbm['f1'] >= 0.7]
    # pulsar_rank = pulsar[(pulsar['f1_rf'] >= 0.7) & (pulsar['f1_lgbm'] >= 0.7)]

    # pulsar_sum_rf = generate_summary_pulsar(pulsar)
    # pulsar_sum_lgbm = generate_summary_pulsar_lgbm(pulsar)
    #
    # ozone_sum_rf = generate_summary_rf(ozone)
    # ozone_sum_lgbm = generate_summary_lgbm(ozone)
    #
    # seatunnel_sum_rf = generate_summary_rf(seatunnel)
    # seatunnel_sum_lgbm = generate_summary_lgbm(seatunnel)

    # seatunnel['rank_rf'] = seatunnel['rank_rf'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    # seatunnel['feature_group'] = seatunnel['feature_group'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    # summary = seatunnel.groupby("rank_rf").agg(
    #     total_group_features=('feature_group', 'nunique'),
    #     total_features_above_80=('f1_rf', lambda x: (x > 0.6).sum()),
    #     percentage=('f1_rf', lambda x: (x > 0.6).sum() / len(x)),
    #     f1_min=('f1_rf', 'min')
    # ).reset_index()
    #
    # # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    # summary.columns = [
    #     "ผลรวมลำดับ",
    #     "จำนวนฟีเจอร์ของลำดับ",
    #     "จำนวนฟีเจอร์ที่คาดหวัง",
    #     "เปอร์เซ็นต์ที่คาดหวัง",
    #     "F1 ต่ำสุดของผลรวมลำดับ"
    # ]
    #
    # # เพิ่มคอลัมน์ "Metricx" = "F1"
    # summary.insert(0, "Metricx", "F1")
    #
    # # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    # summary = summary.sort_values(by="ผลรวมลำดับ")