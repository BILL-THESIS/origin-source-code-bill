import pandas as pd
from varclushi import VarClusHi


if __name__ == "__main__":
    # File paths
    input_filepath = "../../output/seatunnel_compare.pkl"
    data = pd.read_pickle(input_filepath)

    prefix = "java:"
    suffix = "_created"

    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].dropna()

    # demo1_vc = VarClusHi(selected_cols, maxeigval2=0.7, maxclus=None)
    demo1_vc = VarClusHi(selected_cols)
    demo1_vc.varclus()

    show = demo1_vc.rsquare