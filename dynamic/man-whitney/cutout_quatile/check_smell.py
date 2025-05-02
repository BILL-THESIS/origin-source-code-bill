import pandas as pd


def fill_col(df: pd.DataFrame):
    cols_to_fill = [col for col in df.columns if 'java:' in col and '_created' in col]
    return cols_to_fill


if __name__ == "__main__":
    se = pd.read_pickle("../../output/seatunnal_cut_time.pkl")
    pu = pd.read_pickle("../../output/pulsar_cut_time.pkl")
    oz = pd.read_pickle("../../output/ozone_cut_time.pkl")

    se_col = fill_col(se)
    pu_col = fill_col(pu)
    oz_col = fill_col(oz)

    df_se = se[se_col]
    df_pu = pu[pu_col]
    df_oz = oz[oz_col]

    df_se_drop = df_se.dropna(axis=1, how='all')
    df_pu_drop = df_pu.dropna(axis=1, how='all')
    df_oz_drop = df_oz.dropna(axis=1, how='all')

    col_se = df_se_drop.columns
    col_pu = df_pu_drop.columns
    col_oz = df_oz_drop.columns

    all_cols = list(col_se)

    df_cols = pd.DataFrame([all_cols[i:i + 5] for i in range(0, len(all_cols), 5)])
    df_sorted = df_cols.apply(lambda col: sorted(col.dropna(), reverse=False) + [''] * (col.isna().sum()))
    df_cleaned = df_cols.applymap(lambda x: x.replace("_created", "") if isinstance(x, str) else x)

