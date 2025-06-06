import joblib
import pandas as pd
import sys


s= joblib.load("output_lightgbm/seatunnal_optuna_result_combinations_new.pkl")
o =joblib.load("output_lightgbm/ozone_optuna_result_combinations_new.pkl")
p= joblib.load("output_lightgbm/pulsar_optuna_result_combinations_new.pkl")

s_s =joblib.load("output_lightgbm/seatunnel_optuna_result_each_smell.pkl")
o_s = joblib.load("output_lightgbm/ozone_optuna_result_each_smell.pkl")
p_s = joblib.load("output_lightgbm/pulsar_optuna_result_each_smell.pkl")