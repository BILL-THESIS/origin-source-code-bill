import itertools
import logging
import multiprocessing
import platform
import random
import time

import joblib
import pandas as pd
import threadpoolctl
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def chunk_list(data_combi, num_chunks, shuffle=False):
    if shuffle:
        random.shuffle(data_combi)
    avg_chunk_size = len(data_combi) // num_chunks
    return [data_combi[i * avg_chunk_size:(i + 1) * avg_chunk_size] for i in range(num_chunks)]



def fit_scaler(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)


def kmeans_cluster(scaled_df, max_threads=3):
    results = []
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    try:
        threadpoolctl.threadpool_limits(limits=max_threads)
    except Exception as e:
        logging.warning(f'Error setting thread limits: {e}')

    for n_clusters in range(2, 5):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        with joblib.parallel_backend('loky'):
            kmeans.fit(scaled_df)
            score = silhouette_score(scaled_df, kmeans.labels_)
            results.append((scaled_df.columns.to_list(), n_clusters, score, kmeans.labels_))
    return results



def process_combinations(df, combinations):
    kmeans_results = []
    for combo in combinations:
        sub_df = df[list(combo)]
        scaled_df = fit_scaler(sub_df)
        kmeans_results.append(kmeans_cluster(scaled_df))
    return kmeans_results


if __name__ == '__main__':

    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to normalize cluster at: {start_time_gmt}")

    # cpus = multiprocessing.cpu_count()
    cpus = 3
    # Test the function
    df = pd.read_parquet('../../output/seatunnal_20col.parquet')
    # drop the columns that are not needed
    df = df.drop(
        columns=['D_change', 'B_change', 'CP_change', 'C_change', 'OOA_change'])

    # Generate all column combinations
    combinations = [combo for r in range(2, len(df.columns) + 1) for combo in itertools.combinations(df.columns, r)]

    # Chunk the combinations list
    chunks = chunk_list(combinations, 8)

    # Process each chunk of combinations
    for chunk in chunks:
        results = process_combinations(df, chunk)
        print(results)
    print('Done')

    # Platform-specific multiprocessing
    system = platform.system()
    if system == 'Linux':
        logging.info('Running on Linux')
        with multiprocessing.pool.Pool(processes=cpus, maxtasksperchild=1) as pool:
            parsed_split = pool.map(process_combinations(df, chunk), chunk, 3)
            parsed_split.to_parquet(f'chunk_cluster_{start_time_gmt}.parquet')

        # with multiprocessing.Pool(processes=cpus, maxtasksperchild=1) as pool:
        #     results = pool.map(lambda chunk: process_combinations(df, chunk), chunks)
            logging.info("Results on Linux: %s", parsed_split)
    elif system == 'Darwin':
        logging.info('Running on macOS')
        with multiprocessing.Pool(processes=cpus, maxtasksperchild=1) as pool:
            results = pool.map(lambda chunk: process_combinations(df, chunk), chunks)
            logging.info("Results on macOS: %s", results)
    else:
        # Code to execute if running on another OS
        print("Running on an unknown or unsupported OS")

    end = time.time()
    total_time = end - start_time
    time_minutes = total_time / 60
    time_hours = total_time / 3600

    print("total time", total_time)
    print("total_time {:.2f} minutes".format(time_minutes))
    print("Total time {:.2f} hours".format(time_hours))