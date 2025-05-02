import joblib
import pandas as pd
import sys

sea = joblib.load("output/seatunnel_optuna_result.pkl")
ozone = joblib.load("output/ozone_optuna_result.pkl")
