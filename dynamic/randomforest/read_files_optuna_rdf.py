import joblib
import pandas as pd
import sys

ozne_rf = joblib.load("output_randomforest/ozone_optuna_rf_combinations_new.pkl")
pulsar_rf = joblib.load("output_randomforest/pulsar_resampled_combinations_new.pkl")
sea = joblib.load("output_randomforest/seatunnel_optuna_result_combinations_new_rf.pkl")

ozone_each = joblib.load("output_randomforest/ozone_optuna_result_each_smell_rdf.pkl")
pulsar_each = joblib.load("output_randomforest/pulsar_optuna_result_each_smell_rdf.pkl")
sea_each = joblib.load("output_randomforest/seatunnel_optuna_result_each_smell_rdf.pkl")
