import os
import signal

import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def read_file_joblib(path_file):
    with open(os.path.join(path_file), 'rb') as f:
        file_test = joblib.load(f)
    return file_test


def sort_values_time(df, percen_data: float):
    for index, row in df.iterrows():
        time = row[['time0', 'time1', 'time2']]
        sort_time = time.sort_values(ascending=False)
        print(sort_time.iloc[0])

        # The most value in the time0, time1, time2
        most_value = sort_time.iloc[0]

        # The difference between the second value and the there value
        sumtime = abs(sort_time.iloc[1] + sort_time.iloc[2])

        lower_brown = sumtime * (1 - percen_data)
        upper_brown = sumtime * (1 + percen_data)
        # lower_brown = most_value * (1 - percen_data)
        # upper_brown = most_value + (1 - percen_data)

        difference_percentage = (abs(most_value - sumtime) / most_value) * 100

        if most_value > upper_brown:
            diff_time_value = abs(upper_brown - most_value)
        else:
            diff_time_value = abs(most_value - lower_brown)

        diff_time = lower_brown < most_value < upper_brown

        # add new columns sum
        df.loc[index, 'most_value'] = most_value
        df.loc[index, 'sum_time'] = sumtime
        df.loc[index, 'diff_percen'] = difference_percentage
        df.loc[index, f'lower_brown_{percen_data}'] = lower_brown
        df.loc[index, f'upper_brown_{percen_data}'] = upper_brown
        df.loc[index, 'diff_time'] = diff_time
        df.loc[index, f'diff_time_value_{percen_data}'] = diff_time_value

    return df

def drop_col(df):
    df.drop(columns=['f1_score_class0', 'f1_score_class1', 'f1_score_class2',
                     'f1_smote_class0', 'f1_smote_class1', 'f1_smote_class2', 'std_counts',
                     'std_f1', 'std_f1_smote'], axis=1, inplace=True)
    return df

def polt_3d_html(df, z_label, project_name):
    # Prepare the data for 3D plotting
    X = df['index_time01']
    Y = df['index_time12']
    Z = z_label

    # Plotting
    fig = go.Figure(data=[
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=12, color=Z, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(
        title=f"3D Plot of {project_name} split {z_label.name}",
        scene=dict(xaxis_title=f'{X.name}', yaxis_title=f'{Y.name}', zaxis_title=f'{z_label.name}'))

    fig.select_xaxes('time01 (hours)')
    fig.select_yaxes('time12 (hours)')
    fig.write_html(f"../matplotlib/output/{project_name}_split_{z_label.name}.html", include_plotlyjs=True,
                   full_html=True)
    return fig.show()


if __name__ == '__main__':
    ozone_class3_gbc = read_file_joblib('../../models/output/ozone_test_model_timeclass3_16Sep.parquet')
    ozone_class3_gbc.drop(columns=['report_dict'], axis=1, inplace=True)
    ozone_class3_svc = read_file_joblib('../../models/output/ozone_test_model_svc_timeclass3_28Sep.parquet')
    ozone_class3_svc.drop(columns=['report_dict'], axis=1, inplace=True)
    ozone_class3_rfc = read_file_joblib('../../models/output/ozone_test_model_rfc_timeclass3_28Sep.parquet')
    ozone_class3_rfc.drop(columns=['report_dict'], axis=1, inplace=True)

    seatunnal_class3_gbc = read_file_joblib('../../models/output/seatunnel_test_model_timeclass3_16Sep.parquet')
    seatunnal_class3_gbc.drop(columns=['report_dict'], axis=1, inplace=True)

    pulsar_class3_gbc = read_file_joblib('../../models/output/pulsarl_test_model_timeclass3_16Sep.parquet')
    pulsar_class3_gbc.drop(columns=['report_dict'], axis=1, inplace=True)
    pulsar_class3_svc = read_file_joblib('../../models/output/pulsar_test_model_svc_timeclass3_28Sep.parquet')
    pulsar_class3_svc.drop(columns=['report_dict'], axis=1, inplace=True)

    ozone_check_values_gbc = sort_values_time(ozone_class3_gbc, 0.3)
    ozone_check_values_svc = sort_values_time(ozone_class3_svc, 0.3)
    ozone_check_values_rfc = sort_values_time(ozone_class3_rfc, 0.3)

    seatunnal_check_values_gbc = sort_values_time(seatunnal_class3_gbc, 0.3)

    pulsar_check_values_gbc = sort_values_time(pulsar_class3_gbc, 0.3)
    pulsar_check_values_svc = sort_values_time(pulsar_class3_svc, 0.3)

    ozone_30percen_gbc_ture = ozone_check_values_gbc[ozone_check_values_gbc['diff_time'] == True]
    ozone_30percensvc_ture = ozone_check_values_svc[ozone_check_values_svc['diff_time'] == True]
    ozone_30percen_rfc_ture = ozone_check_values_rfc[ozone_check_values_rfc['diff_time'] == True]

    pulsar_30percen_gbc_ture = pulsar_check_values_gbc[pulsar_check_values_gbc['diff_time'] == True]
    pulsar_30percen_svc_ture = pulsar_check_values_svc[pulsar_check_values_svc['diff_time'] == True]