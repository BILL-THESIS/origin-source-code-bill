import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    input_ozone = "../../output_resample/output_resample/ozone_compare.pkl"
    input_pulsar = "../../output_resample/output_resample/pulsar_compare.pkl"
    input_setannel = "../../output_resample/output_resample/seatunnel_compare.pkl"
    data_group_ozone = pd.read_pickle("../../output/output_resample/ozone_correlation_main_group.pkl")
    data_group_pulsar = pd.read_pickle("../../output/output_resample/pulsar_correlation_main_group_7.pkl")
    data_group_setannel = pd.read_pickle("../../output/output_resample/seatunnel_correlation_main_group_4.pkl")
    data_ozone = pd.read_pickle(input_ozone)
    data_pulsar = pd.read_pickle(input_pulsar)
    data_setannel = pd.read_pickle(input_setannel)

    return data_ozone, data_group_ozone, data_pulsar, data_group_pulsar, data_setannel, data_group_setannel

def preprocess_data(data):
    # polt time modify o
    data['total_time'] = pd.to_timedelta(data['total_time'])
    data['total_hours'] = data['total_time'].dt.total_seconds() / 3600
    # data['year_month'] = data['completed_date'].dt.to_period('M').astype(str)

    # แปลง datetime เป็น year-month แบบ Period
    data['year_month'] = data['completed_date'].dt.to_period('M')

    # เรียงข้อมูลตามเวลา
    data = data.sort_values('year_month')

    # แปลงเป็น string เพื่อใช้ plot
    data['year_month_str'] = data['year_month'].astype(str)
    return data

def plot_time(data, title):
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data['year_month_str'], data['total_hours'], marker='o', linestyle='-')
    # plt.title(f"ระยะเวลาที่ใช้แก้ไขปัญหาของ {title}")
    plt.title(f"Time to resolve issues for {title}")
    # plt.xlabel('ระยะเวลา (ปี-เดือน)')
    # plt.ylabel('ระยะเวลาที่ใช้แก้ไขปัญหา (ชั่วโมง)')
    plt.xlabel("Time (Year-Month)")
    plt.ylabel("Time to resolve issues (Hours)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title}_time_modify.png")
    plt.show()

def sort_smell(smell_list, name):
    data_list = pd.DataFrame(smell_list, columns=[name])
    data_list['smell_num'] = data_list[name].str.extract(r'S(\d+)_')[0].astype(int)
    data_list = data_list.sort_values('smell_num').reset_index(drop=True)
    data_list = data_list.drop(columns=['smell_num'])
    return data_list

def select_cols(data):
    prefix = "java:"
    suffix = "_created"
    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].fillna(0)
    return selected_cols


if __name__ == "__main__":
    data_ozone, data_group_ozone, data_pulsar, data_group_pulsar, data_setannel, data_group_setannel = load_data()

    data_ozone = preprocess_data(data_ozone)
    data_pulsar = preprocess_data(data_pulsar)
    data_setannel = preprocess_data(data_setannel)
    #
    # plot_time(data_ozone, "Ozone")
    # plot_time(data_pulsar, "Pulsar")
    # plot_time(data_setannel, "Seatunnel")

    all_smell_ozone = data_group_ozone[0] + data_group_ozone[1] + data_group_ozone[2] + data_group_ozone[3]
    all_smell_pulsar = data_group_pulsar[0] + data_group_pulsar[1] + data_group_pulsar[2] + data_group_pulsar[3] + data_group_pulsar[4] + data_group_pulsar[5] + data_group_pulsar[6]
    all_smell_setannel = data_group_setannel[0] + data_group_setannel[1] + data_group_setannel[2] + data_group_setannel[3]


    df_smell_ozone = sort_smell(all_smell_ozone, "ozone")
    df_smell_pulsar = sort_smell(all_smell_pulsar, "pulsar")
    df_smell_setannel = sort_smell(all_smell_setannel, "setannel")

    all_smell = pd.concat([df_smell_ozone, df_smell_pulsar, df_smell_setannel], axis=1).fillna("")


