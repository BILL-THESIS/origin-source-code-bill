import joblib
import pandas as pd
import sys

ozne_rf = joblib.load("output_randomforest/ozone_optuna_result_rdf.pkl")
pulsar_rf = joblib.load("output_randomforest/pulsar_optuna_result_rdf.pkl")
seatunnal_rf = joblib.load("output_randomforest/seatunnal_optuna_result_rdf.pkl")

