import pandas as pd


def time(data: pd.DataFrame):
    # แปลง total_time เป็น timedelta
    data['total_time'] = pd.to_timedelta(data['total_time'])

    # กรองเฉพาะรายการที่ไม่เกิน 90 วัน
    filtered_df = data[data['total_time'] <= pd.Timedelta(days=90)]

    return filtered_df


if __name__ == '__main__':
    ozone = pd.read_pickle("../output/ozone_compare.pkl")
    pulsar = pd.read_pickle("../output/pulsar_compare.pkl")
    seatunnal = pd.read_pickle("../output/seatunnel_compare.pkl")

    ozone_cut_time = time(ozone)
    ozone_cut_time.to_pickle("../output/ozone_cut_time.pkl")
    pulsar_cut_time = time(pulsar)
    pulsar_cut_time.to_pickle("../output/pulsar_cut_time.pkl")
    seatunnal_cut_time = time(seatunnal)
    seatunnal_cut_time.to_pickle("../output/seatunnal_cut_time.pkl")
