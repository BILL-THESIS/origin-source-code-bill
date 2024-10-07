import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.dot import label
from hvplot import parallel_coordinates


def custom_time_hour_clustering(df):
    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 12, 18, 24, 48, 72, 96, 120, np.inf]
    labels = list(range(len(bins) - 1))  # [0, 1, 2, ..., 14]

    # Use pd.cut to assign values to the appropriate bins
    df['time_hour_class'] = pd.cut(df['total_time_hours'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def percentage_smell(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the percentage change for each smell type."""
    rename_dict = {
        'created_Dispensables': 'created_d',
        'created_Bloaters': 'created_b',
        'created_Change Preventers': 'created_cp',
        'created_Couplers': 'created_c',
        'created_Object-Orientation Abusers': 'created_ooa',
        'created_Uncategorized': 'created_u',
        'ended_Dispensables': 'ended_d',
        'ended_Bloaters': 'ended_b',
        'ended_Change Preventers': 'ended_cp',
        'ended_Couplers': 'ended_c',
        'ended_Object-Orientation Abusers': 'ended_ooa',
        'ended_Uncategorized': 'ended_u'
    }
    df = df.rename(columns=rename_dict)

    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'diff_{col.lower()}'] = df[f'ended_{col}'] - df[f'created_{col}']
        df[f'percentage_{col.lower()}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

    return df


def calculate_hourly_sums(ozone_outlier_hour, hour_class):
    """Calculates sums for the specified hour class."""
    ozone_hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == hour_class]
    if ozone_hour.empty:  # Handle case where there are no records for the hour class
        return {f'Hour Class {hour_class}': 'No Data'}

    # creactng the data to data frame

    return {
        'number_row_class': ozone_hour.shape[0],
        'created_d': ozone_hour['created_d'].sum(),
        'created_b': ozone_hour['created_b'].sum(),
        'created_cp': ozone_hour['created_cp'].sum(),
        'created_c': ozone_hour['created_c'].sum(),
        'created_ooa': ozone_hour['created_ooa'].sum(),
        'created_u': ozone_hour['created_u'].sum(),
        'ended_d': ozone_hour['ended_d'].sum(),
        'ended_b': ozone_hour['ended_b'].sum(),
        'ended_cp': ozone_hour['ended_cp'].sum(),
        'ended_c': ozone_hour['ended_c'].sum(),
        'ended_ooa': ozone_hour['ended_ooa'].sum(),
        'ended_u': ozone_hour['ended_u'].sum(),
        'diff_d': ozone_hour['diff_d'].sum(),
        'diff_b': ozone_hour['diff_b'].sum(),
        'diff_cp': ozone_hour['diff_cp'].sum(),
        'diff_c': ozone_hour['diff_c'].sum(),
        'diff_ooa': ozone_hour['diff_ooa'].sum(),
        'diff_u': ozone_hour['diff_u'].sum()
    }


def summarize_hourly_data(ozone_outlier_hour):
    """Summarizes data for each hour class."""
    summary = {}
    unique_classes = ozone_outlier_hour['time_hour_class'].unique()
    for hour_class in unique_classes:
        summary[hour_class] = calculate_hourly_sums(ozone_outlier_hour, hour_class)
    return summary


def create_dataframe(data_dict):
    df = pd.DataFrame()
    for hour_class, data in data_dict.items():
        for key, value in data.items():
            df.loc[hour_class, key] = value
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Hour Class'}, inplace=True)
    df = df.sort_values('Hour Class', ignore_index=True)
    return df


def cumulative_diff(df):
    # Calculates the cumulative difference for each smell type
    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'cumulative_diff_{col.lower()}'] = df[f'diff_{col.lower()}'].diff()
        df[f'cumulative_diff_sum_{col.lower()}'] = df[f'diff_{col.lower()}'].diff().cumsum()
    return df


def separate_cumulative_differences(df):
    positive = []
    negative = []
    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        cumulative_diff_positive = df[f'cumulative_diff_{col.lower()}'].where(df[f'cumulative_diff_{col.lower()}'] > 0,
                                                                              0)
        cumulative_diff_negative = df[f'cumulative_diff_{col.lower()}'].where(df[f'cumulative_diff_{col.lower()}'] < 0,
                                                                              0)

        positive.append(cumulative_diff_positive)
        negative.append(cumulative_diff_negative)

    # Combine the results into DataFrames (optional)
    positive_df = pd.concat(positive, axis=1)
    negative_df = pd.concat(negative, axis=1)

    return positive_df, negative_df


def merged_hour_class(data_class, data_separate):
    data_separate = data_separate.reset_index()
    df = pd.merge(data_class[['Hour Class', 'number_row_class']], data_separate, left_on='Hour Class',
                  right_on='index')
    return df


def plot_cumulative_diff(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_diff_d': 'smell d',
        'cumulative_diff_b': 'smell b',
        'cumulative_diff_c': 'smell c',
        'cumulative_diff_cp': 'smell ooa',
        'cumulative_diff_ooa': 'smell cp',
        'cumulative_diff_u': 'smell u'
    }
    print(df.columns)

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)  # X = 'Hour Class', Y = column values

    # Set custom x-ticks and labels based on unique values in 'Hour Class'
    plt.xticks(
        ticks=range(len(df)),  # Adjust ticks to the length of 'Hour Class'
        labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96', '96-120',
                '> 120'],
        rotation=45
    )

    # Add labels and title
    plt.xlabel('Hours Time')
    plt.ylabel('Number of cumulative diff')
    plt.title(f'{project_name} - Cumulative Differences by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    # plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference.png'))
    return plt.show()


def plot_cumulative_diff_sum(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_diff_sum_d': 'smell d',
        'cumulative_diff_sum_b': 'smell b',
        'cumulative_diff_sum_c': 'smell c',
        'cumulative_diff_sum_cp': 'smell ooa',
        'cumulative_diff_sum_ooa': 'smell cp',
        'cumulative_diff_sum_u': 'smell u'
    }
    print(df.columns)

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)  # X = 'Hour Class', Y = column values

    # Set custom x-ticks and labels based on unique values in 'Hour Class'
    plt.xticks(
        ticks=range(len(df)),  # Adjust ticks to the length of 'Hour Class'
        labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96', '96-120',
                '> 120'],
        rotation=45
    )

    # Add labels and title
    plt.xlabel('Hours Time')
    plt.ylabel('Number of cumulative diff')
    plt.title(f'{project_name} - Cumulative Differences Sum by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    # plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_sum.png'))
    return plt.show()


def liners_regression(df, project_name):
    # Set up the figure and axes for the plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Cumulative Difference of {project_name}')

    axs[0, 0].plot(df['Hour Class'], df['cumulative_diff_d'], marker='o', color='b')
    axs[0, 0].set_title('cumulative diff smell d')
    axs[0, 0].set_xlabel('Hour Class')
    axs[0, 0].set_ylabel('cumulative_diff_d')

    axs[0, 1].plot(df['Hour Class'], df['cumulative_diff_b'], marker='o', color='g')
    axs[0, 1].set_title('cumulative diff smell b')
    axs[0, 1].set_xlabel('Hour Class')
    axs[0, 1].set_ylabel('cumulative_diff_b')

    axs[1, 0].plot(df['Hour Class'], df['cumulative_diff_cp'], marker='o', color='r')
    axs[1, 0].set_title('cumulative diff smell cp')
    axs[1, 0].set_xlabel('Hour Class')
    axs[1, 0].set_ylabel('cumulative_diff_cp')

    axs[1, 1].plot(df['Hour Class'], df['cumulative_diff_ooa'], marker='o', color='m')
    axs[1, 1].set_title('cumulative diff smell ooa')
    axs[1, 1].set_xlabel('Hour Class')
    axs[1, 1].set_ylabel('cumulative_diff_ooa')

    axs[2, 0].plot(df['Hour Class'], df['cumulative_diff_c'], marker='o', color='g')
    axs[2, 0].set_title('cumulative diff smell c')
    axs[2, 0].set_xlabel('Hour Class')
    axs[2, 0].set_ylabel('cumulative diff c')

    axs[2, 1].plot(df['Hour Class'], df['cumulative_diff_u'], marker='o', color='g')
    axs[2, 1].set_title('cumulative diff smell u')
    axs[2, 1].set_xlabel('Hour Class')
    axs[2, 1].set_ylabel('cumulative_diff_u')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference.png'))
    return plt.show()

if __name__ == '__main__':
    ozone_original = pd.read_pickle('../../Github/output/ozone_filtered_final_api_new.pkl')
    pulsar_original = pd.read_pickle('../../Github/output/pulsar_filtered_final_api_new.pkl')
    seatunnal_original = pd.read_pickle('../../Github/output/seatunnel_filtered_final_api_new.pkl')

    ozone_outlier = pd.read_parquet('../../models/output/ozone_prepare_to_train_newversion_9Sep.parquet')
    pulsar_outlier = pd.read_parquet('../../models/output/pulsar_prepare_to_train_newversion_9Sep.parquet')
    seatunnal_outlier = pd.read_parquet('../../models/output/seatunnel_prepare_to_train_newversion_9Sep.parquet')

    # Apply the custom clustering function and percentage calculation
    ozone_outlier_hour = percentage_smell(custom_time_hour_clustering(ozone_outlier))
    pulsar_outlier_hour = percentage_smell(custom_time_hour_clustering(pulsar_outlier))
    seatunnal_outlier_hour = percentage_smell(custom_time_hour_clustering(seatunnal_outlier))

    # Summarize data for each hour class
    ozone_summary = summarize_hourly_data(ozone_outlier_hour)
    pulsar_summary = summarize_hourly_data(pulsar_outlier_hour)
    seatunnal_summary = summarize_hourly_data(seatunnal_outlier_hour)

    # Create an empty DataFrame
    ozone_df = create_dataframe(ozone_summary)
    pulsar_df = create_dataframe(pulsar_summary)
    seatunnal_df = create_dataframe(seatunnal_summary)

    # Calculate the cumulative difference for each smell type
    ozone_df = cumulative_diff(ozone_df)
    pulsar_df = cumulative_diff(pulsar_df)
    seatunnal_df = cumulative_diff(seatunnal_df)

    # Separate data frame
    ozone_positive, ozone_negative = separate_cumulative_differences(ozone_df)
    # ozone_positive_merged = merged_hour_class(ozone_df, ozone_positive)
    pulsar_positive, pulsar_negative = separate_cumulative_differences(pulsar_df)
    seatunnal_positive, seatunnal_negative = separate_cumulative_differences(seatunnal_df)

    # # Plot the cumulative difference for each smell type
    # liners_regression(ozone_df, 'Ozone')
    # liners_regression(pulsar_df, 'pulsar')
    # liners_regression(seatunnal_df, 'seatunnal')

    plot_cumulative_diff(ozone_positive, 'Ozone positive')
    plot_cumulative_diff(ozone_negative,'Ozone negative')

    plot_cumulative_diff(pulsar_positive, 'Pulsar positive')
    plot_cumulative_diff(pulsar_negative, 'Pulsar negative')

    plot_cumulative_diff(seatunnal_positive, 'Seatunnal positive')
    plot_cumulative_diff(seatunnal_negative, 'Seatunnal negative')

    plot_cumulative_diff_sum(ozone_df, 'Ozone')
    plot_cumulative_diff_sum(pulsar_df, 'Pulsar')
    plot_cumulative_diff_sum(seatunnal_df, 'Seatunnal')


    # Count the number of instances in each class
    ozone_outlier_hour_counts = ozone_outlier_hour['time_hour_class'].value_counts().sort_index()
    pulsar_outlier_hour_counts = pulsar_outlier_hour['time_hour_class'].value_counts().sort_index()
    seatunnal_outlier_hour_counts = seatunnal_outlier_hour['time_hour_class'].value_counts().sort_index()
