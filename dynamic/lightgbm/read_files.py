import joblib
import pandas as pd
import sys

sea = joblib.load("/Users/bill/origin-source-code-bill/dynamic/output/resample_data/seatunnel_resampled_data_20250221_215909.pkl")
# check size of the list
size_file = sys.getsizeof(sea)

pusalr = joblib.load("/Users/bill/origin-source-code-bill/dynamic/output/resample_data/pulsar_resamples_list_2_chunks6.pkl")
# check size of the list
size_file_pusalr = sys.getsizeof(pusalr)

pusalr_size = pusalr[0][:10]

memory = pusalr_size[0].memory_usage(deep=True).sum()

