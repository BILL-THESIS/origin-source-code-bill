import ast
import itertools

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


def generate_summary(dataframe):
    summary = dataframe.groupby("sum").agg(
        total_group_features=('feature_group', 'nunique'),
        total_features_above_80=('f1', lambda x: (x > 0.7).sum()),
        percentage=('f1', lambda x: (x > 0.7).sum() / len(x)),
        f1_min=('f1', 'min'),  # ค่า F1 ต่ำสุดในแต่ละกลุ่ม
        f1_max=('f1', 'max'),
        feature_group_not_duplicate=('feature_group', lambda x: list(set(x)))
    ).reset_index()

    # เปลี่ยนชื่อคอลัมน์ให้ตรงกับภาพ
    summary.columns = [
        "ผลรวมลำดับ",
        "จำนวนฟีเจอร์ของลำดับ",
        "จำนวนฟีเจอร์ที่คาดหวัง",
        "เปอร์เซ็นต์ที่คาดหวัง",
        "F1 ต่ำสุดของผลรวมลำดับ",
        "F1 สูงสุดของผลรวมลำดับ",
        "ฟีเจอร์ของลำดับ",
    ]

    # เพิ่มคอลัมน์ "Metricx" = "F1"
    summary.insert(0, "Metricx", "F1")

    # จัดเรียงตาม sum_rank (ถ้ายังไม่ได้เรียง)
    summary = summary.sort_values(by="ผลรวมลำดับ")

    return summary

if __name__ == '__main__':
    ozone_rf = pd.read_pickle("rank_f1/output_rank/ozone_optuna_result_rank_rf.pkl")
    ozone_lgbm = pd.read_pickle("rank_f1/output_rank/ozone_optuna_result_rank_lgbm.pkl")
    ozone_rf['feature_group'] = ozone_rf['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)
    ozone_lgbm['feature_group'] = ozone_lgbm['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)

    pulsar_rf = pd.read_pickle("rank_f1/output_rank/pulsar_optuna_result_rank_rf.pkl")
    pulsar_lgbm = pd.read_pickle("rank_f1/output_rank/pulsar_optuna_result_rank_lgbm.pkl")
    pulsar_rf['feature_group'] = pulsar_rf['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)
    pulsar_lgbm['feature_group'] = pulsar_lgbm['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)

    seatunnel_rf = pd.read_pickle("rank_f1/output_rank/seatunnal_optuna_result_rank_rf.pkl")
    seatunnel_lgbm = pd.read_pickle("rank_f1/output_rank/seatunnel_optuna_result_rank_lgbm.pkl")
    seatunnel_rf['feature_group'] = seatunnel_rf['feature_group'].apply(lambda x: str(x) if isinstance(x, list) else x)
    seatunnel_lgbm['feature_group'] = seatunnel_lgbm['feature_group'].apply(
        lambda x: str(x) if isinstance(x, list) else x)

    # s_rf = generate_summary(seatunnel_rf).head(20).round(4)
    # s_lgbm = generate_summary(seatunnel_lgbm).head(20).round(4)
    s_rf = generate_summary(seatunnel_rf).round(4)
    s_lgbm = generate_summary(seatunnel_lgbm).round(4)

    p_rf = generate_summary(pulsar_rf).round(4)
    p_lgbm = generate_summary(pulsar_lgbm).round(4)

    o_rf = generate_summary(ozone_rf).round(4)
    o_lgbm = generate_summary(ozone_lgbm).round(4)

    



