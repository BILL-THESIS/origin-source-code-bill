import joblib
import pandas as pd

seat = joblib.load("output_resample/seatunnal_resampled.pkl")
ozone = joblib.load("output_resample/ozone_resampled_data.pkl")
pulsar = joblib.load("output_resample/pulsar_resampled_data.pkl")