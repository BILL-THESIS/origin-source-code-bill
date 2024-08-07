import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import StepPatch


def chenge_time_to_hours(df, time01, time12):
    # Change the time to hours
    df['time01'] = df[time01].dt.total_seconds() / 3600
    df['time12'] = df[time12].dt.total_seconds() / 3600

    return df


def distance_difference_value(time01, time12, df):
    # Calculate the difference between the two times
    df['time_diff'] = df[time12] - df[time01]

    # Calculate the absolute difference between the time difference and the percentile
    df['time_diff_abs'] = abs(df['time_diff'])

    return df


def plot_time_difference(df, time01, time12, time_diff, project_name):

    # Set plot style and figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data from the DataFrame
    x = df.index
    y_time01 = df[time01]
    y_time12 = df[time12]
    y_time_diff = df[time_diff]

    # Plot time01
    ax.plot(y_time01, x, label=time01, marker='o', linestyle='-', color='red')
    # Plot time12
    ax.plot(y_time12, x, label=time12, marker='x', linestyle='-', color='blue')
    # Plot time difference
    ax.plot(y_time_diff, x, label=time_diff, marker='s', linestyle='-', color='green')

    # Add labels, title, and legend
    ax.set_xlabel('Index')
    ax.set_ylabel('Time')
    ax.set_title(f'Time Difference {project_name}')
    ax.legend()

    # Add grid lines
    ax.grid(True)
    return plt.show()


if __name__ == '__main__':
    pulsar_percentile = pd.read_pickle('../output/pulsar_percentile.pkl')
    pulsar_percentile_combi = pd.read_pickle('../output/pulsar_split_f1_smote_time_class3.pkl')
    print("pulsar", pulsar_percentile_combi.describe())
    print("\n")

    ozone_percentile = pd.read_pickle('../output/ozone_percentile.pkl')
    ozone_percentile_combi = pd.read_pickle('../output/ozone_split_f1_smote_time_class3.pkl')
    print("ozone", ozone_percentile_combi.describe())

    pulsar_hour = chenge_time_to_hours(pulsar_percentile_combi, 'time01', 'time12')
    pulsar_distance = distance_difference_value('time01', 'time12', pulsar_hour)
    # pulsar_distance_plot = plot_time_difference(pulsar_distance, 'time01', 'time12','time_diff' ,
    #                                             'Pulsar')
