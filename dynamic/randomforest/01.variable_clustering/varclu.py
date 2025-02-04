import pandas as pd


if __name__ == "__main__":
    # File paths
    input_filepath = "../../output/output/seatunnel_compare.pkl"
    data = pd.read_pickle(input_filepath)


    prefix = "java:"
    suffix = "_created"

    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].dropna()

    print(selected_cols)