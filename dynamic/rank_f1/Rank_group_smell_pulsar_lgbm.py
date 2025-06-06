import joblib
import pandas as pd


def clean_created(feature_list):
    return [item.replace('_created', '') for item in feature_list]


if __name__ == '__main__':

    # Load the data
    pulsar = pd.read_pickle('../output/pulsar_cut_time.pkl')
    pulsar_group = pd.read_pickle("../01.variable_clustering/output_variable/pulsar_correlation_main_group.pkl")
    pulsar_optuna = joblib.load("../lightgbm/output_lightgbm/pulsar_optuna_result_each_smell.pkl")
    pulsar_optuna_all_case = joblib.load("../lightgbm/output_lightgbm/pulsar_optuna_result_combinations_new.pkl")

    # Clean the feature names
    pulsar_optuna['feature_group'] = pulsar_optuna['feature_group'].apply(lambda x: x[0] if isinstance(x, list) and x else '')
    groups = [pulsar_optuna[pulsar_optuna['feature_group'].isin(pulsar_group[i])] for i in range(len(pulsar_group))]

    g1, g2, g3, g4 = groups
    g1['rank'] = g1['result'].rank(ascending=False)
    g2['rank'] = g2['result'].rank(ascending=False)
    g3['rank'] = g3['result'].rank(ascending=False)
    g4['rank'] = g4['result'].rank(ascending=False)

    g_all = pd.concat([g1, g2, g3, g4], axis=0)

    df_rank = pulsar_optuna_all_case['feature_group'].apply(
        lambda s: [g_all.set_index('feature_group').loc[x, 'rank'] for x in s])

    df_rank = pd.DataFrame(df_rank)

    df_rank_new = pd.DataFrame(df_rank['feature_group'].tolist(), columns=["G1", "G2", "G3", "G4"],
                                           index=df_rank.index)

    df_rank_new['feature_group'] = pulsar_optuna_all_case['feature_group']
    df_rank_new['sum'] = df_rank_new[['G1', 'G2', 'G3', "G4"]].sum(axis=1)
    df_rank_new['f1'] = pulsar_optuna_all_case['result']

    df_rank_new.to_pickle("output_rank/pulsar_optuna_result_rank_lgbm.pkl")

#     plot the rank with F1 by seaborn

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # หาค่ามากสุด/น้อยสุด
    max_idx = df_rank_new["f1"].idxmax()
    min_idx = df_rank_new["f1"].idxmin()

    # วาดกราฟ
    plt.figure(figsize=(8, 6))
    plt.scatter(df_rank_new["sum"], df_rank_new["f1"], color='blue')
    plt.scatter(df_rank_new.loc[max_idx, "sum"], df_rank_new.loc[max_idx, "f1"], color='green', label='f1 max', s=100)
    plt.scatter(df_rank_new.loc[min_idx, "sum"], df_rank_new.loc[min_idx, "f1"], color='red', label='f1 min', s=100)
    plt.xlabel("sum")
    plt.ylabel("f1 score")
    plt.title("Scatter Plot: sum vs f1")
    plt.legend()
    plt.grid(True)
    plt.show()
