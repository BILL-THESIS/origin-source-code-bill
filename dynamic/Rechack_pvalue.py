import pandas as pd


def get_status(df):
    df['status'] = pd.cut(
        df['d_value'],
        bins=[-float('inf'), 0.14, 0.33, 0.474, float('inf')],
        labels=["negligible", "small", "medium", "large"]
    )
    return df


if __name__ == "__main__":
    ozone = pd.read_pickle("man-whitney/output_man/ozone_man_whitney.pkl")
    seatunnel = pd.read_pickle("man-whitney/output_man/seatunnal_man_whitney.pkl")
    pulsar = pd.read_pickle("man-whitney/output_man/pulsar_man_whitney.pkl")

    # sea_man = seatunnel[seatunnel["p_value"] < 0.05]
    # pulsar_man = pulsar[pulsar["p_value"] < 0.05]
    # ozone_man = ozone[ozone["p_value"] < 0.05]
    #
    # sea_man = get_status(sea_man)
    # pulsar_man = get_status(pulsar_man)
    # ozone_man = get_status(ozone_man)
    #
    # # count status
    # status_counts = sea_man["status"].value_counts()
    # print(status_counts)
    # pulsar_counts = pulsar_man["status"].value_counts()
    # print(pulsar_counts)
    # ozone_counts =ozone_man["status"].value_counts()
    # print(ozone_counts)
    #
    # # result count
    # df = pd.concat([ozone_counts, pulsar_counts, status_counts], axis=1)
    # df.columns = ['ozone', 'pulsar', 'seatunnel']
    # df.sort_values("status", ascending=False, inplace=True)
    #
    # sea_man_final = sea_man[sea_man["status"].isin(["large", "medium"])]
    # pulsar_man_final = pulsar_man[pulsar_man["status"].isin(["large", "medium"])]
    # ozone_man_final = ozone_man[ozone_man["status"].isin(["large", "medium"])]
