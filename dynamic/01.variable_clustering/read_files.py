import joblib
import pandas
import pandas as pd

seatannel = pd.read_pickle("../output/output/seatunnel_correlation_main_group_4.pkl")
pulasr = pd.read_pickle("../output/output/pulsar_correlation_main_group_7.pkl")
ozone = pd.read_pickle("../output/output/ozone_correlation_main_group.pkl")

ozone_resanping =joblib.load("/dynamic/output/resample_data/ozone_resampled_data_20250318_090635.pkl")
result_se = joblib.load("/dynamic/output/seatunnel/seatunnel_optuna_results_20250314_043120.pkl")

result_se_list = result_se[0] + result_se[1] + result_se[2] + result_se[3]
result_se_list_to_df = pd.DataFrame(result_se_list)

result_se_list_to_df_80_percent = result_se_list_to_df[result_se_list_to_df['result'] > 0.8]


pulasr_df = pd.DataFrame(pulasr)
pulasr_df = pulasr_df.T
pulasr_df = pulasr_df.fillna("")

ozone_df = pd.DataFrame(ozone)
ozone_df = ozone_df.T
ozone_df = ozone_df.fillna("")