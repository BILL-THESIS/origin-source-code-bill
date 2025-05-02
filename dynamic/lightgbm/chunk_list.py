import joblib
import numpy as np

def chunk_list(lst, n_chunks):
    #Split a list into n roughly equal-sized chunks
    chunk_size = int(np.ceil(len(lst) / n_chunks))
    return [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]

if __name__ == '__main__':
    lst = joblib.load('../02.resample_data/output_resample/pulsar_resampled_data.pkl')

    pulsar = chunk_list(lst, 18)
