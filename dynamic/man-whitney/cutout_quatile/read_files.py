import pandas as pd


def verify_importance(df: pd.DataFrame):
    df_importance = df[df['p_value'] < 0.05]
    return df_importance

if __name__ == "__main__":
    seatannel = pd.read_pickle("../output_man/seatunnal_man_whitney.pkl")
    pulsar = pd.read_pickle("../output_man/pulsar_man_whitney.pkl")
    ozone =  pd.read_pickle("../output_man/ozone_man_whitney.pkl")

    s_significant = seatannel['significant'].value_counts()
    p_significant = pulsar['significant'].value_counts()
    o_significant = ozone['significant'].value_counts()

    seatannel_importance = verify_importance(seatannel)
    # seatannel_importance.to_pickle("../output_man/seatannel_importance.pkl")
    pulsar_importance = verify_importance(pulsar)

    # pulsar_importance.to_pickle("../output_man/pulsar_importance.pkl")
    ozone_importance = verify_importance(ozone)
    # pulsar_importance.to_pickle("../output_man/ozone_importance.pkl")

    seatannel_importance_conut = seatannel_importance['eff_size'].value_counts()
    pulsar_importance_conut = pulsar_importance['eff_size'].value_counts()
    ozone_importance_conut = ozone_importance['eff_size'].value_counts()

    df_importance_conut = pd.concat([seatannel_importance_conut, pulsar_importance_conut, ozone_importance_conut], axis=1)
    df_importance_conut.columns = ['seatannel', 'pulsar', 'ozone']
    df_importance_conut.sort_values("eff_size", ascending=True, inplace=True)
    df_importance_conut.fillna("0", inplace=True)