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

    df_significant = pd.concat([s_significant, p_significant, o_significant], axis=1)
    df_significant.columns = ['seatannel', 'pulsar', 'ozone']

    seatannel_importance = verify_importance(seatannel)
    seatannel_importance.to_pickle("../output_man/seatannel_importance.pkl")

    pulsar_importance = verify_importance(pulsar)
    pulsar_importance = pulsar_importance[pulsar_importance['eff_size'] == 'small']
    pulsar_importance.to_pickle("../output_man/pulsar_importance.pkl")

    ozone_importance = verify_importance(ozone)
    ozone_importance.to_pickle("../output_man/ozone_importance.pkl")

    seatannel_importance_conut = seatannel_importance['eff_size'].value_counts()
    pulsar_importance_conut = pulsar_importance['eff_size'].value_counts()
    ozone_importance_conut = ozone_importance['eff_size'].value_counts()


    max_len = max(len(seatannel_importance['metric']), len(pulsar_importance['metric']),
                  len(ozone_importance['metric']))
    df = pd.DataFrame({
        'seatannel': seatannel_importance['metric'].tolist() + [""] * (max_len - len(seatannel_importance['metric'])),
        'pulsar': pulsar_importance['metric'].tolist() + [""] * (max_len - len(pulsar_importance['metric'])),
        'ozone': ozone_importance['metric'].tolist() + [""] * (max_len - len(ozone_importance['metric']))
    })


    df_importance_conut = pd.concat([seatannel_importance_conut, pulsar_importance_conut, ozone_importance_conut], axis=1)
    df_importance_conut.columns = ['seatannel', 'pulsar', 'ozone']
    df_importance_conut.sort_values("eff_size", ascending=True, inplace=True)
    df_importance_conut.fillna("0", inplace=True)