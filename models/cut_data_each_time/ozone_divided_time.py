import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def custom_time_hour_clustering(df):

    # Define the bin edges and labels
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 12, 18, 24, 48, 72, 96, 120,  np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

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


if __name__ == '__main__':
    ozone_original = pd.read_pickle('../../Github/output/ozone_filtered_final_api_new.pkl')
    pulsar_original = pd.read_pickle('../../Github/output/pulsar_filtered_final_api_new.pkl')
    seatunnal_original = pd.read_pickle('../../Github/output/seatunnel_filtered_final_api_new.pkl')

    ozone_outlier = pd.read_parquet('../../models/output/ozone_prepare_to_train_newversion_9Sep.parquet')
    pular_outlier = pd.read_parquet('../../models/output/pulsar_prepare_to_train_newversion_9Sep.parquet')
    seatunnal_outlier = pd.read_parquet('../../models/output/seatunnel_prepare_to_train_newversion_9Sep.parquet')

    # Apply the custom clustering function
    ozone_outlier_hour = custom_time_hour_clustering(ozone_outlier)
    ozone_outlier_hour = percentage_smell(ozone_outlier_hour)

    ozone_outlier_0hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 0]
    ozone_outlier_1hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 1]
    ozone_outlier_2hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 2]
    ozone_outlier_3hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 3]
    ozone_outlier_4hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 4]
    ozone_outlier_5hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 5]
    ozone_outlier_6hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 6]
    ozone_outlier_7hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 7]
    ozone_outlier_8hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 8]
    ozone_outlier_9hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 9]
    ozone_outlier_10hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 10]
    ozone_outlier_11hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 11]
    ozone_outlier_12hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 12]
    ozone_outlier_13hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 13]
    ozone_outlier_14hour = ozone_outlier_hour[ozone_outlier_hour['time_hour_class'] == 14]

    ozone_outlier_0hour_d = ozone_outlier_0hour['created_d'].sum()
    ozone_outlier_0hour_b = ozone_outlier_0hour['created_b'].sum()
    ozone_outlier_0hour_cp = ozone_outlier_0hour['created_cp'].sum()
    ozone_outlier_0hour_c = ozone_outlier_0hour['created_c'].sum()
    ozone_outlier_0hour_ooa = ozone_outlier_0hour['created_ooa'].sum()
    ozone_outlier_0hour_u = ozone_outlier_0hour['created_u'].sum()

    ozone_outlier_0hour_d_end = ozone_outlier_0hour['ended_d'].sum()
    ozone_outlier_0hour_b_end = ozone_outlier_0hour['ended_b'].sum()
    ozone_outlier_0hour_cp_end = ozone_outlier_0hour['ended_cp'].sum()
    ozone_outlier_0hour_c_end = ozone_outlier_0hour['ended_c'].sum()
    ozone_outlier_0hour_ooa_end = ozone_outlier_0hour['ended_ooa'].sum()
    ozone_outlier_0hour_u_end = ozone_outlier_0hour['ended_u'].sum()

    ozone_outlier_0hour_d_diff = ozone_outlier_0hour['diff_d'].sum()
    ozone_outlier_0hour_b_diff = ozone_outlier_0hour['diff_b'].sum()
    ozone_outlier_0hour_cp_diff = ozone_outlier_0hour['diff_cp'].sum()
    ozone_outlier_0hour_c_diff = ozone_outlier_0hour['diff_c'].sum()
    ozone_outlier_0hour_ooa_diff = ozone_outlier_0hour['diff_ooa'].sum()
    ozone_outlier_0hour_u_diff = ozone_outlier_0hour['diff_u'].sum()


    pular_outlier_hour = custom_time_hour_clustering(pular_outlier)
    pular_outlier_hour = percentage_smell(pular_outlier_hour)
    seatunnal_outlier_hour = custom_time_hour_clustering(seatunnal_outlier)
    seatunnal_outlier_hour = percentage_smell(seatunnal_outlier_hour)


    # Count the number of instances in each class
    ozone_outlier_hour_counts = ozone_outlier_hour['time_hour_class'].value_counts().sort_index()
    pular_outlier_hour_counts = pular_outlier_hour['time_hour_class'].value_counts().sort_index()
    seatunnal_outlier_hour_counts = seatunnal_outlier_hour['time_hour_class'].value_counts().sort_index()





