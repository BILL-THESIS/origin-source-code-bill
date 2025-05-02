import joblib
import pandas as pd

seat = joblib.load("output/seatunnal_resampled.pkl")
ozone = joblib.load("output/ozone_resampled_data.pkl")
pulsar = joblib.load("output/pulsar_resampled_data.pkl")