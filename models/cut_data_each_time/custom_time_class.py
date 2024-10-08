import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.ImagePalette import negative
from dask.dot import label
from hvplot import parallel_coordinates


def custom_time_hour_clustering(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 12, 18, 24, 48, 72, 96, 120, np.inf]
    labels = list(range(len(bins) - 1))

    # Use pd.cut to assign values to the appropriate bins
    df['time_hour_class'] = pd.cut(df['total_time_hours'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def percentage_smell(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the percentage change for each smell type
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


def calculate_hourly_sums(df_outlier, hour_class):
    # Calculates sums for the specified hour class
    df_hour = df_outlier[df_outlier['time_hour_class'] == hour_class]
    if df_hour.empty:  # Handle case where there are no records for the hour class
        return {f'Hour Class {hour_class}': 'No Data'}

    # creactng the data to data frame

    return {
        'number_row_class': df_hour.shape[0],
        'created_d': df_hour['created_d'].sum(),
        'created_b': df_hour['created_b'].sum(),
        'created_cp': df_hour['created_cp'].sum(),
        'created_c': df_hour['created_c'].sum(),
        'created_ooa': df_hour['created_ooa'].sum(),
        'created_u': df_hour['created_u'].sum(),

        'ended_d': df_hour['ended_d'].sum(),
        'ended_b': df_hour['ended_b'].sum(),
        'ended_cp': df_hour['ended_cp'].sum(),
        'ended_c': df_hour['ended_c'].sum(),
        'ended_ooa': df_hour['ended_ooa'].sum(),
        'ended_u': df_hour['ended_u'].sum(),

        'diff_d': df_hour['diff_d'].sum(),
        'diff_b': df_hour['diff_b'].sum(),
        'diff_cp': df_hour['diff_cp'].sum(),
        'diff_c': df_hour['diff_c'].sum(),
        'diff_ooa': df_hour['diff_ooa'].sum(),
        'diff_u': df_hour['diff_u'].sum()
    }


def summarize_hourly_data(df_outlier : pd.DataFrame) -> pd.DataFrame:
    # Summarizes data for each hour class
    summary = {}
    unique_classes = df_outlier['time_hour_class'].unique()
    for hour_class in unique_classes:
        summary[hour_class] = calculate_hourly_sums(df_outlier, hour_class)
    return summary

def verity_values_positive_negative(data_dict: dict):
    df = pd.DataFrame()
    for hour_class, data in data_dict.items():
        for col, value in data.items():
            df.loc[hour_class, col] = value

        positive = {f'positive_{col}': data[f'diff_{col.lower()}'] if data[f'diff_{col.lower()}'] > 0 else 0 for col in
                    ['d', 'b', 'cp', 'c', 'ooa', 'u']}
        negative = {f'negative_{col}': data[f'diff_{col.lower()}'] if data[f'diff_{col.lower()}'] < 0 else 0 for col in
                    ['d', 'b', 'cp', 'c', 'ooa', 'u']}

        for col, value_positive in positive.items():
            df.loc[hour_class, col] = value_positive
        for col, value_negative in negative.items():
            df.loc[hour_class, col] = value_negative

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'hour_class'}, inplace=True)
    df = df.sort_values('hour_class', ignore_index=True)

    return df

def cumulative_diff(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the cumulative difference for each smell type
    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'cumulative_positive_{col.lower()}'] = df[f'positive_{col.lower()}'].diff().cumsum()
        df[f'cumulative_positive_sum_{col.lower()}'] = df[f'positive_{col.lower()}'].cumsum()
        df[f'cumulative_negative_{col.lower()}'] = df[f'negative_{col.lower()}'].diff().cumsum()
        df[f'cumulative_negative_sum_{col.lower()}'] = df[f'negative_{col.lower()}'].cumsum()
        df[f'cumulative_sum_{col.lower()}'] = df[f'diff_{col.lower()}'].diff().cumsum()
    return df


# def separate_cumulative_differences(df):
#     positive = []
#     negative = []
#     for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
#         cumulative_diff_positive = df[f'diff_{col.lower()}'].where(df[f'diff_{col.lower()}'] > 0,0)
#         cumulative_diff_negative = df[f'diff_{col.lower()}'].where(df[f'diff_{col.lower()}'] < 0,0)
#
#         positive.append(cumulative_diff_positive)
#         negative.append(cumulative_diff_negative)
#
#     # Combine the results into DataFrames (optional)
#     positive_df = pd.concat(positive, axis=1)
#     negative_df = pd.concat(negative, axis=1)
#
#     return positive_df, negative_df


def plot_cumulative_diff_positive(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_positive_d': 'smell d',
        'cumulative_positive_b': 'smell b',
        'cumulative_positive_c': 'smell c',
        'cumulative_positive_cp': 'smell ooa',
        'cumulative_positive_ooa': 'smell cp',
        'cumulative_positive_u': 'smell u'
    }
    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)

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
    plt.title(f'{project_name} - Cumulative Differences Positive by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_Positive.png'))
    return plt.show()

def plot_cumulative_diff_negative(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_negative_d': 'smell d',
        'cumulative_negative_b': 'smell b',
        'cumulative_negative_c': 'smell c',
        'cumulative_negative_cp': 'smell ooa',
        'cumulative_negative_ooa': 'smell cp',
        'cumulative_negative_u': 'smell u'
    }

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)

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
    plt.title(f'{project_name} - Cumulative Differences Negative by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_Negative.png'))
    return plt.show()


def plot_cumulative_diff_sum(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_sum_d': 'smell d',
        'cumulative_sum_b': 'smell b',
        'cumulative_sum_c': 'smell c',
        'cumulative_sum_cp': 'smell ooa',
        'cumulative_sum_ooa': 'smell cp',
        'cumulative_sum_u': 'smell u'
    }

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)

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
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_Sum.png'))
    return plt.show()


def plot_cumulative_diff_positive_sum(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_positive_sum_d': 'smell d',
        'cumulative_positive_sum_b': 'smell b',
        'cumulative_positive_sum_c': 'smell c',
        'cumulative_positive_sum_cp': 'smell ooa',
        'cumulative_positive_sum_ooa': 'smell cp',
        'cumulative_positive_sum_u': 'smell u'
    }

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)

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
    plt.title(f'{project_name} - Positive Sum by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_Positive_sum.png'))
    return plt.show()

def plot_cumulative_diff_negative_sum(df, project_name):
    # Map DataFrame columns to their labels
    columns_labels = {
        'cumulative_negative_sum_d': 'smell d',
        'cumulative_negative_sum_b': 'smell b',
        'cumulative_negative_sum_c': 'smell c',
        'cumulative_negative_sum_cp': 'smell ooa',
        'cumulative_negative_sum_ooa': 'smell cp',
        'cumulative_negative_sum_u': 'smell u'
    }

    # Plot the data with custom x-axis labels
    plt.figure(figsize=(10, 6))

    # Plot each cumulative_diff column
    for col, label in columns_labels.items():
        plt.plot( df[col], marker='.', label=label)

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
    plt.title(f'{project_name} - Negative Sum by Smell Type')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(f'../matplotlib/output/{project_name}_Cumulative_Difference_Negative_sum.png'))
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

    # Verity data positive values and Negative values by time class
    ozone_verity = verity_values_positive_negative(ozone_summary)
    pulsar_verity = verity_values_positive_negative(pulsar_summary)
    seatunnal_verity = verity_values_positive_negative(seatunnal_summary)


    # Calculate the cumulative difference for each smell type
    ozone_df = cumulative_diff(ozone_verity)
    pulsar_df = cumulative_diff(pulsar_verity)
    seatunnal_df = cumulative_diff(seatunnal_verity)

    # Separate data frame
    # ozone_positive, ozone_negative = separate_cumulative_differences(ozone_outlier_hour)
    # pulsar_positive, pulsar_negative = separate_cumulative_differences(pulsar_outlier_hour)
    # seatunnal_positive, seatunnal_negative = separate_cumulative_differences(seatunnal_outlier_hour)

    # Plot the cumulative difference for each smell type
    # plot_cumulative_diff_positive(ozone_df, 'Ozone positive')
    # plot_cumulative_diff_negative(ozone_df,'Ozone negative')
    #
    # plot_cumulative_diff_positive(pulsar_df, 'Pulsar positive')
    # plot_cumulative_diff_negative(pulsar_df, 'Pulsar negative')
    # #
    # plot_cumulative_diff_positive(seatunnal_df, 'Seatunnal positive')
    # plot_cumulative_diff_negative(seatunnal_df, 'Seatunnal negative')
    #
    # plot_cumulative_diff_sum(ozone_df, 'Ozone')
    # plot_cumulative_diff_sum(pulsar_df, 'Pulsar')
    # plot_cumulative_diff_sum(seatunnal_df, 'Seatunnal')

    plot_cumulative_diff_positive_sum(ozone_df, 'Ozone')
    plot_cumulative_diff_positive_sum(pulsar_df, 'Pulsar')
    plot_cumulative_diff_positive_sum(seatunnal_df, 'Seatunnal')

    plot_cumulative_diff_negative_sum(ozone_df, 'Ozone')
    plot_cumulative_diff_negative_sum(pulsar_df, 'Pulsar')
    plot_cumulative_diff_negative_sum(seatunnal_df, 'Seatunnal')

    # Count the number of instances in each class
    # ozone_outlier_hour_counts = ozone_outlier_hour['time_hour_class'].value_counts().sort_index()
    # pulsar_outlier_hour_counts = pulsar_outlier_hour['time_hour_class'].value_counts().sort_index()
    # seatunnal_outlier_hour_counts = seatunnal_outlier_hour['time_hour_class'].value_counts().sort_index()
