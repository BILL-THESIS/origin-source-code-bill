import joblib
import pandas as pd


def clean_created(feature_list):
    return [item.replace('_created', '') for item in feature_list]


if __name__ == '__main__':

    # Load the data
    df = pd.read_pickle('../output/pulsar_cut_time.pkl')
    df_group = pd.read_pickle("../01.variable_clustering/output_variable/pulsar_correlation_main_group.pkl")
    optuna_all_case = joblib.load("../randomforest/output_randomforest/ozone_optuna_result_each_smell_rdf.pkl")

    # Clean the feature names
    df_rank = pd.DataFrame()
    df_rank['feature_group'] = optuna_all_case['feature_group']
    df_rank['f1'] = optuna_all_case['result']
    df_rank['rank'] = optuna_all_case['result'].rank(ascending=False)

    df_rank.to_pickle("output_rank/pulsar_optuna_result_rank_each_smell_rf.pkl")
