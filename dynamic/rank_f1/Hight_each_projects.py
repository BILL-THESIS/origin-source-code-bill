import pandas as pd

pu_lg = pd.read_pickle('output_rank/pulsar_optuna_result_rank_lgbm.pkl')
se_lg = pd.read_pickle('output_rank/seatunnel_optuna_result_rank_lgbm.pkl')
oz_lg = pd.read_pickle('output_rank/ozone_optuna_result_rank_lgbm.pkl')