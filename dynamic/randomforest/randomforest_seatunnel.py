import pandas as pd

if __name__ == "__main__":
    # File paths
    df_original = pd.read_pickle("../output/seatunnel_compare.pkl")
    df_compare = pd.read_pickle("../output/seatunnel_significant.pkl")

    # Check if values in 'metric' column of df_compare are column names in df_original with suffix '_ended'
    df_original_columns_ended = [col for col in df_original.columns if col.endswith('_ended')]

    df_compare['metric'] = df_compare['metric'].apply(lambda i: i.replace('_ended', ''))

    df_check = df_compare['metric'].apply(lambda i: i in df_original_columns_ended)

