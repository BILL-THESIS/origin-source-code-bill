import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def custom_time_hour_clustering(df):

    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    # Use pd.cut to assign values to the appropriate bins
    df['time_hour_class'] = pd.cut(df['total_time_hours'], bins=bins, labels=labels, right=False, include_lowest=True)

    return df


def custom_time_day_clustering(df):
    # Ensure the 'timestamp' is in datetime format
    df['total_time'] = pd.to_timedelta(df['total_time'])

    # Extract hour and day of the week
    df['hour'] = df['total_time'].dt.components.hours
    df['day_of_week'] = df['total_time'].dt.dayofweek  # Monday=0, Sunday=6

    # Define custom time bins for hours (e.g., Morning, Afternoon, Evening, Night)
    hour_bins = [0, 6, 12, 18, 24]
    hour_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['time_of_day'] = pd.cut(df['hour'], bins=hour_bins, labels=hour_labels, right=False)

    # Define custom labels for day of the week (Weekday vs Weekend)
    day_labels = ['Weekday' if day < 5 else 'Weekend' for day in df['day_of_week']]
    df['day_class'] = day_labels

    return df

if __name__ == '__main__':
    ozone_original = pd.read_pickle('../../Github/output/ozone_filtered_final_api_new.pkl')
    pulsar_original = pd.read_pickle('../../Github/output/pulsar_filtered_final_api_new.pkl')
    seatunnal_original = pd.read_pickle('../../Github/output/seatunnel_filtered_final_api_new.pkl')

    # Apply the custom clustering function
    ozone_hour = custom_time_hour_clustering(ozone_original)
    pulsar_hour = custom_time_hour_clustering(pulsar_original)
    seatunnal_hour = custom_time_hour_clustering(seatunnal_original)

    ozone_day = custom_time_day_clustering(ozone_original)
    pulsar_day = custom_time_day_clustering(pulsar_original)
    seatunnal_day = custom_time_day_clustering(seatunnal_original)

    # Count the number of instances in each class
    ozone_class_counts = ozone_hour['time_hour_class'].value_counts().sort_index()
    pulsar_class_counts = pulsar_hour['time_hour_class'].value_counts().sort_index()
    seatunnal_class_counts = seatunnal_hour['time_hour_class'].value_counts().sort_index()

    ozone_day_counts = ozone_day['day_class'].value_counts().sort_index()
    pulsar_day_counts = pulsar_day['day_class'].value_counts().sort_index()
    seatunnal_day_counts = seatunnal_day['day_class'].value_counts().sort_index()


