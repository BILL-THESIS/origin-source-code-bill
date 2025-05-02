import pandas as pd

best_para_files = [
    "../../output_resample/pulsar/pulsar_best_params_20250203_043532_batch1.pkl",
    "../../output_resample/pulsar/pulsar_best_params_20250203_043532_batch2.pkl",
    "../../output_resample/pulsar/pulsar_best_params_20250203_043532_batch3.pkl",
    "../../output_resample/pulsar/pulsar_best_params_20250203_043532_batch4.pkl",
    "../../output_resample/pulsar/pulsar_best_params_20250203_043532_batch5.pkl"
]

result_tun = [
    "../../output_resample/pulsar/pulsar_rdf_quantile_all_20250203_043532_batch1.pkl",
    "../../output_resample/pulsar/pulsar_rdf_quantile_all_20250203_043532_batch2.pkl",
    "../../output_resample/pulsar/pulsar_rdf_quantile_all_20250203_043532_batch3.pkl",
    "../../output_resample/pulsar/pulsar_rdf_quantile_all_20250203_043532_batch4.pkl",
    "../../output_resample/pulsar/pulsar_rdf_quantile_all_20250203_043532_batch5.pkl"
]
best_para = pd.concat([pd.read_pickle(file) for file in best_para_files], ignore_index=True)
result = pd.concat([pd.read_pickle(file) for file in result_tun], ignore_index=True)

check  = pd.merge(best_para, result, on='features', how='inner')

df_s = pd.read_pickle('../../output/output_resample/seatunnel_rdf_quantile_all.pkl')
df_s_1 = pd.read_pickle('../../output/seatunnel/seatunnel_rdf_quantile_all_20250204_014054.pkl')
df_s_2 = pd.read_pickle('../../output/seatunnel/seatunnel_rdf_quantile_all_20250204_025340.pkl')
