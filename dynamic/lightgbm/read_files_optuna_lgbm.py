import joblib
import pandas as pd
import sys

sea = joblib.load("output_lightgbm/seatunnel_optuna_result.pkl")
ozone = joblib.load("output_lightgbm/ozone_optuna_result.pkl")
