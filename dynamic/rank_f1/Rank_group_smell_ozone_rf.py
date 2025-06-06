import joblib
import pandas as pd


def clean_created(feature_list):
    return [item.replace('_created', '') for item in feature_list]


if __name__ == '__main__':

    # Load the data
    pulsar = pd.read_pickle('../output/ozone_cut_time.pkl')
    pulsar_group = pd.read_pickle("../01.variable_clustering/output_variable/ozone_correlation_main_group.pkl")
    pulsar_optuna = joblib.load("../randomforest/output_randomforest/ozone_optuna_result_each_smell_rdf.pkl")
    pulsar_optuna_all_case = joblib.load("../randomforest/output_randomforest/ozone_optuna_result_rank_rf_combinations_new.pkl")

    # Clean the feature names
    pulsar_optuna['feature_group'] = pulsar_optuna['feature_group'].apply(lambda x: x[0] if isinstance(x, list) and x else '')
    groups = [pulsar_optuna[pulsar_optuna['feature_group'].isin(pulsar_group[i])] for i in range(len(pulsar_group))]

    g1, g2 = groups
    g1['rank'] = g1['result'].rank(ascending=False)
    g2['rank'] = g2['result'].rank(ascending=False)

    g_all = pd.concat([g1, g2], axis=0)

    df_rank = pulsar_optuna_all_case['feature_group'].apply(
        lambda s: [g_all.set_index('feature_group').loc[x, 'rank'] for x in s])

    df_rank = pd.DataFrame(df_rank)

    df_rank_new = pd.DataFrame(df_rank['feature_group'].tolist(), columns=["G1", "G2"],
                                           index=df_rank.index)

    df_rank_new['feature_group'] = pulsar_optuna_all_case['feature_group']
    df_rank_new['sum'] = df_rank_new[['G1', 'G2']].sum(axis=1)
    df_rank_new['f1'] = pulsar_optuna_all_case['f1']

    df_rank_new.to_pickle("output_rank/ozone_optuna_result_rank_rf.pkl")
