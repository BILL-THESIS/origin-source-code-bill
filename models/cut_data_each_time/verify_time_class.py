import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def custom_time_hour_clustering(df: pd.DataFrame) -> pd.DataFrame:
    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 12, 18, 24, 48, 72, 96, 120, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

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


def separate_smell_integer(df):
    # Check the number of rows for each time_hour_class
    positive = {}
    negative = {}
    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        positive_list = df[df[f'diff_{col.lower()}'] >= 0].drop(
            columns=[f'diff_{c}' for c in ['d', 'b', 'cp', 'c', 'ooa', 'u'] if c != col])
        negative_list = df[df[f'diff_{col.lower()}'] <= 0].drop(
            columns=[f'diff_{c}' for c in ['d', 'b', 'cp', 'c', 'ooa', 'u'] if c != col])

        positive[col] = positive_list
        negative[col] = negative_list

    return positive, negative


def separate_smell_classes(df: pd.DataFrame) -> dict:
    # Separate the smell classes
    data = []

    for key, df in df.items():
        print(f"Positive & Negative {key}")
        print(f"shape: {df.shape[0]}")
        print(f"sum: {df[f'diff_{key}'].sum()}")

        unique_classes = df['time_hour_class'].unique()
        for h in unique_classes:
            class_df = df[df['time_hour_class'] == h]
            data.append({
                'smell_key': key,
                'time_hour_class': h,
                'sum_diff': class_df[f'diff_{key}'].sum(),
                f'shape': class_df.shape[0]
            })

    new_df = pd.DataFrame(data)

    return new_df


def separant_calculate_smell(df_positive: pd.DataFrame, df_negative: pd.DataFrame):
    df_positive['percentage_positive'] = df_positive['sum_diff'] / df_positive['shape']
    df_negative['percentage_negative'] = df_negative['sum_diff'] / df_negative['shape']

    df_merge = pd.merge(df_positive, df_negative, on=['time_hour_class', 'smell_key'], how='outer')

    return df_merge

def plot_compare_interge(df: pd.DataFrame, project_name: str):

    # Filter the data for each smell type
    smell_types = ['b', 'd', 'c', 'cp', 'ooa', 'u']
    smell_data = {smell: df[df['smell_key'] == smell] for smell in smell_types}

    # Plotting positive and negative values for each smell type over time hour classes
    for smell, data in smell_data.items():
        plt.figure(figsize=(12, 6))

        # Bar plot for percentage_positive (green) and percentage_negative (red) for each time_hour_class
        plt.bar(data['time_hour_class'], data['percentage_positive'],
                color='green', label=f'Smell {smell.upper()} Positive',
                width=0.4
                , alpha=0.5
                # ,align='center'
                )
        plt.bar(data['time_hour_class'], data['percentage_negative'],
                color='red', label=f'Smell {smell.upper()} Negative',
                width=0.4
                , alpha=0.5
                # ,align='edge'
                )

        plt.title(f'{project_name} - Smell {smell.upper()} Positive and Negative Percentages Over Time Hour Classes')

        # Set custom x-ticks and labels based on unique values in 'Hour Class'
        plt.xticks(
            ticks=range(len(data)),  # Adjust ticks to the length of 'Hour Class'
            labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96', '96-120',
                    '> 120'],
            rotation=45
        )

        plt.xlabel('Time Hour Class')
        plt.ylabel('Percentage')
        plt.xticks(data['time_hour_class'])
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        # plt.savefig(os.path.join(f'../matplotlib/output/ozone/{project_name}_compare_integer_{smell}.png'))
        plt.show()



def plot_compare_smell(df: pd.DataFrame, project_name: str):

    # Filter the data for each smell type
    smell_types = ['b', 'd', 'c', 'cp', 'ooa', 'u']
    smell_data = {smell: df[df['smell_key'] == smell] for smell in smell_types}

    # Plotting positive and negative values for each smell type over time hour classes
    for smell, data in smell_data.items():
        plt.figure(figsize=(12, 6))

        # Bar plot for percentage_positive (green) and percentage_negative (red) for each time_hour_class
        plt.bar(data['time_hour_class'], data['sum_diff_x'],
                color='green', label=f'Smell {smell.upper()} Positive',
                width=0.4
                , alpha=0.5
                # ,align='center'
                )
        plt.bar(data['time_hour_class'], data['sum_diff_y'],
                color='red', label=f'Smell {smell.upper()} Negative',
                width=0.4
                , alpha=0.5
                # ,align='edge'
                )

        plt.title(f'{project_name} - Smell {smell.upper()} Positive and Negative Total Smell Time Hour Classes')

        # Set custom x-ticks and labels based on unique values in 'Hour Class'
        plt.xticks(
            ticks=range(len(data)),  # Adjust ticks to the length of 'Hour Class'
            labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96', '96-120',
                    '> 120'],
            rotation=45
        )

        plt.xlabel('Time Hour Class')
        plt.ylabel('Total Smell')
        plt.xticks(data['time_hour_class'])
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        # plt.savefig(os.path.join(f'../matplotlib/output/ozone/{project_name}_compare_smell_{smell}.png'))
        plt.show()



def plot_compare_shape(df: pd.DataFrame, project_name: str):

    # Filter the data for each smell type
    smell_types = ['b', 'd', 'c', 'cp', 'ooa', 'u']
    smell_data = {smell: df[df['smell_key'] == smell] for smell in smell_types}

    # Plotting positive and negative values for each smell type over time hour classes
    for smell, data in smell_data.items():

        barWidth = 0.25
        width = 0.25
        plt.subplots(figsize=(12, 8))


        plt.bar(data['time_hour_class'], data['shape_x'],
                color='green', label=f'Smell {smell.upper()} Positive',
                width=barWidth,
                edgecolor='grey'
                , alpha=0.5
                # ,align='center'
                )
        plt.bar(data['time_hour_class'] + width, data['shape_y'],
                color='red', label=f'Smell {smell.upper()} Negative',
                width= barWidth ,
                edgecolor='grey'
                , alpha=0.5
                # ,align='edge'
                )

        plt.title(f'{project_name} - Smell {smell.upper()} Positive and Negative Total Shape Time Hour Classes')

        # Set custom x-ticks and labels based on unique values in 'Hour Class'
        plt.xticks(
            ticks=range(len(data)),  # Adjust ticks to the length of 'Hour Class'
            labels=['< 0', '1', '2', '3', '4', '5', '6', '7-12', '12-18', '18-24', '24-48', '48-72', '72-96', '96-120',
                    '> 120'],
            rotation=45
        )

        plt.xlabel('Time Hour Class')
        plt.ylabel('Total Shape',)
        plt.xticks(data['time_hour_class'])
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        # plt.savefig(os.path.join(f'../matplotlib/output/ozone/{project_name}_compare_shape_{smell}.png'))
        plt.show()


if __name__ == '__main__':
    ozone_original = pd.read_pickle('../../Github/output/ozone_filtered_final_api_new.pkl')
    pulsar_original = pd.read_pickle('../../Github/output/pulsar_filtered_final_api_new.pkl')
    seatunnal_original = pd.read_pickle('../../Github/output/seatunnel_filtered_final_api_new.pkl')

    ozone_outlier = pd.read_parquet('../../models/output/ozone_prepare_to_train_newversion_9Sep.parquet')
    pular_outlier = pd.read_parquet('../../models/output/pulsar_prepare_to_train_newversion_9Sep.parquet')
    seatunnal_outlier = pd.read_parquet('../../models/output/seatunnel_prepare_to_train_newversion_9Sep.parquet')

    ozone_outlier = custom_time_hour_clustering(percentage_smell(ozone_outlier))
    pular_outlier = custom_time_hour_clustering(percentage_smell(pular_outlier))
    seatunnal_outlier = custom_time_hour_clustering(percentage_smell(seatunnal_outlier))

    ozone_positive, ozone_negative = separate_smell_integer(ozone_outlier)
    pulsar_positive, pulsar_negative = separate_smell_integer(pular_outlier)
    seatunnal_positive, seatunnal_negative = separate_smell_integer(seatunnal_outlier)

    ozone_positive_class = separate_smell_classes(ozone_positive)
    ozone_negative_class = separate_smell_classes(ozone_negative)

    pulsar_positive_class = separate_smell_classes(pulsar_positive)
    pulsar_negative_class = separate_smell_classes(pulsar_negative)

    seatunnal_positive_class = separate_smell_classes(seatunnal_positive)
    seatunnal_negative_class = separate_smell_classes(seatunnal_negative)

    ozone = separant_calculate_smell(ozone_positive_class, ozone_negative_class)
    pulsar = separant_calculate_smell(pulsar_positive_class, pulsar_negative_class)
    seatunnal = separant_calculate_smell(seatunnal_positive_class, seatunnal_negative_class)

    plot_compare_interge(ozone, 'Ozone')
    plot_compare_interge(pulsar, 'Pulsar')
    plot_compare_interge(seatunnal, 'Seatunnal')
    plot_compare_smell(ozone, 'Ozone')
    plot_compare_smell(pulsar, 'Pulsar')
    plot_compare_smell(seatunnal, 'Seatunnal')
    plot_compare_shape(ozone, 'Ozone')
    plot_compare_shape(pulsar, 'Pulsar')
    plot_compare_shape(seatunnal, 'Seatunnal')



