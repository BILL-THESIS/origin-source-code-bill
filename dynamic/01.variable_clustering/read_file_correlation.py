import joblib
import pandas as pd

if __name__ == "__main__":
    seatunnel_spearman_rank = pd.read_pickle("output_variable/seatunnal_spearman_rank_all_case.pkl")
    pulsar_spearman_rank = pd.read_pickle("output_variable/pulsar_spearman_rank_all_case.pkl")
    ozone_spearman_rank = pd.read_pickle("output_variable/ozone_spearman_rank_all_case.pkl")

    seatunnel_group = pd.read_pickle("output_variable/seatunnel_correlation_main_group.pkl")
    pulsar_group = pd.read_pickle("output_variable/pulsar_correlation_main_group.pkl")
    ozone_group = joblib.load("output_variable/ozone_correlation_main_group.pkl")

    seatunnel_combinations = pd.read_pickle("output_variable/seatunnel_combinations.pkl")
    pulsar_combinations = pd.read_pickle("output_variable/pulsar_combinations.pkl")
    ozone_combinations =  pd.read_pickle("output_variable/ozone_combinations.pkl")

    seatunnel_combinations_new = pd.read_pickle("output_variable/seatunnel_combinations_new.pkl")
    pulsar_combinations_new = pd.read_pickle("output_variable/pulsar_combinations_new.pkl")
    ozone_combinations_new = pd.read_pickle("output_variable/ozone_combinations_new.pkl")


    # se_new = se[['url', 'base.sha', 'created_at', 'merged_at', 'java:S100_created',
    #          'java:S101_created','java:S106_created','total_time']].head(4)