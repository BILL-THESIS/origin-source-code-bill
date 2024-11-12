import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calculate_smell_bug(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the percentage change for each smell type
    rename_dict = {
        'Dispensables_created': 'created_d',
        'Bloaters_created': 'created_b',
        'Change Preventers_created': 'created_cp',
        'Couplers_created': 'created_c',
        'Object-Orientation Abusers_created': 'created_ooa',
        'Uncategorized_created': 'created_u',
        'Dispensables_ended': 'ended_d',
        'Bloaters_ended': 'ended_b',
        'Change Preventers_ended': 'ended_cp',
        'Couplers_ended': 'ended_c',
        'Object-Orientation Abusers_ended': 'ended_ooa',
        'Uncategorized_ended': 'ended_u'
    }
    df = df.rename(columns=rename_dict)
    df['total_time'] = pd.to_datetime(df['merged_at']) - pd.to_datetime(df['created_at'])
    df['completed_date'] = pd.to_datetime(df['created_at']) + df['total_time']
    df['value_ended'] = pd.to_numeric(df['value_ended'], errors='coerce')
    df['value_created'] = pd.to_numeric(df['value_created'], errors='coerce')
    df['diff_bug'] = df['value_ended'] - df['value_created']
    df['percentage_bug'] = ((df['value_ended'] - df['value_created']) / df['value_created']) * 100

    for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'diff_{col}'] = df[f'ended_{col}'] - df[f'created_{col}']
        df[f'percentage_{col}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

    # Add a new column for year-month
    df['year'] = pd.to_datetime(df['completed_date']).dt.to_period('Y')
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')
    df['year_month_day'] = pd.to_datetime(df['completed_date']).dt.to_period('D')
    return df


def robust_outlier(df: pd.DataFrame) -> pd.DataFrame:
    columns = df['total_time']

    median = np.median(columns)
    print("Median: ", median)

    med = np.abs(columns - median).median()
    print("MAD: ", med)

    MADN = (med / 0.6745)
    print("MADN: ", MADN)

    threshold = 2.24

    outlier = (columns - median).abs() / MADN > threshold
    print("Sum outliers :", outlier.sum())

    # lower = pd.Timedelta(max(np.abs(median - threshold * MADN).total_seconds(), 0), unit='s')
    # lower = pd.Timedelta(days=1)
    # lower = pd.Timedelta(hours=1)
    lower = pd.Timedelta(minutes=30)
    upper = median + threshold * MADN

    print(
        "The value of the lower outliers: ", lower,
        "The value of the upper outliers: ", upper)

    # df_outliers = df[~outlier]
    df_lower = df[df['total_time'] <= lower]
    df_upper = df[df['total_time'] >= upper]

    df_normal = df[~df.index.isin(df_lower.index) & ~df.index.isin(df_upper.index)]

    return df_normal, df_lower, df_upper, lower, upper


def separate_data(df: pd.DataFrame) -> pd.DataFrame:
    # Selecting only the "diff" columns
    diff_columns = [col for col in df.columns if 'diff_' in col]

    # Separate positive and negative values for each "diff" column
    for col in diff_columns:
        df[f'{col}_positive'] = df[col].apply(lambda x: x if x > 0 else 0)
        df[f'{col}_negative'] = df[col].apply(lambda x: x if x < 0 else 0)

    return df


def cumulative_diff(df: pd.DataFrame) -> pd.DataFrame:
    # Calculates the cumulative difference for each smell type
    for col in ['bug', 'd', 'b', 'cp', 'c', 'ooa', 'u']:
        df[f'cumulative_positive_{col.lower()}'] = df[f'diff_{col.lower()}_positive'].diff().cumsum()
        df[f'cumulative_negative_{col.lower()}'] = df[f'diff_{col.lower()}_negative'].diff().cumsum()
        # Ensure cumulative_negative_* only includes negative values
        df[f'cumulative_negative_{col.lower()}'] = df[f'cumulative_negative_{col.lower()}'].apply(lambda x: x if x < 0 else 0)
    return df

# separate years
def separate_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df_2021 = df[df['completed_date'].dt.year == 2021]
    df_2022 = df[df['completed_date'].dt.year == 2022]
    df_2023 = df[df['completed_date'].dt.year == 2023]
    df_2024 = df[df['completed_date'].dt.year == 2024]
    return df_2021, df_2022, df_2023, df_2024


def group_data_by_year(df_normal, df_lower, df_upper):
    group_normal = df_normal.groupby('year')
    group_low = df_lower.groupby('year')
    group_up = df_upper.groupby('year')

    data_group_years = pd.DataFrame({
        'year': group_normal['year'].first(),
        'Normal': group_normal.size(),
        'Lower outliers': group_low.size(),
        'Upper outliers': group_up.size()
    }).fillna(0)

    return data_group_years


def group_data_by_year_month(df_normal, df_lower, df_upper):
    group_normal = df_normal.groupby('year_month')
    group_low = df_lower.groupby('year_month')
    group_up = df_upper.groupby('year_month')

    data_group_years_month = pd.DataFrame({
        'year_month_normal': group_normal['year_month'].first(),
        'year_moth_lower': group_low['year_month'].first(),
        'year_moth_upper': group_up['year_month'].first(),
        'Normal': group_normal.size(),
        'Lower outliers': group_low.size(),
        'Upper outliers': group_up.size()
    }).fillna(0)

    return data_group_years_month.reset_index()


def plot_cumulative_bars_overview(df, project_name,file_suffix):
    columns = [
        'cumulative_positive_bug', 'cumulative_negative_bug',
        'cumulative_positive_d', 'cumulative_negative_d',
        'cumulative_positive_b', 'cumulative_negative_b',
        'cumulative_positive_cp', 'cumulative_negative_cp',
        'cumulative_positive_c', 'cumulative_negative_c',
        'cumulative_positive_ooa', 'cumulative_negative_ooa',
        'cumulative_positive_u', 'cumulative_negative_u'
    ]

    colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'brown',
        'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'lime'
    ]

    df_grouped = df.groupby('year_month')[columns].sum().sort_values(by='year_month')
    row_counts = df['year_month'].value_counts().sort_index()
    ax = df_grouped.plot(kind='bar', figsize=(15, 10), color=colors)

    # Annotate the bars with the number of rows
    for p, count in zip(ax.patches, row_counts):
        ax.annotate(str(count), (p.get_x() + p.get_width() / 2., p.get_height() * 1.005), ha='center')

    plt.title(f'Cumulative Differences for {file_suffix} Shape {df["year_month"].shape}')
    plt.xlabel('Year-Month')
    plt.ylabel('Cumulative Difference')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.savefig(os.path.join(f'{file_suffix}_Cumulative_Differences_Bar.png'))
    plt.show()


def plot_years(years_list, outlier=False):
    for df_year in years_list:
        if df_year.empty:
            continue

        # Find the variable name for the current df_year
        var_name = [name for name, value in globals().items() if value is df_year]
        if var_name:
            suffix = '_outlier' if outlier else ''
            # file_suffix = f"{var_name[0]}{suffix}"
            file_suffix = f"{var_name[0]}"
            plot_cumulative_bars_overview(df_year, f'seatunnel_{df_year.iloc[0]["year_month"].year}{suffix}',
                                          file_suffix)

def plot_show_shape_outlier_year_and_month(data: pd.DataFrame, lower: pd.Timedelta, upper: pd.Timedelta,
                                           project_name: str):
    plt.figure(figsize=(15, 10))
    bar_width = 0.25
    r1 = np.arange(len(data['year_month']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, data['Lower outliers'], color='blue', width=bar_width, edgecolor='grey', label='Lower outliers')
    plt.bar(r2, data['Normal'], color='green', width=bar_width, edgecolor='grey', label='Normal')
    plt.bar(r3, data['Upper outliers'], color='red', width=bar_width, edgecolor='grey', label='Upper outliers')

    for i in range(len(data)):
        plt.text(r1[i], data['Lower outliers'][i], str(data['Lower outliers'][i]), ha='center', va='bottom')
        plt.text(r2[i], data['Normal'][i], str(data['Normal'][i]), ha='center', va='bottom')
        plt.text(r3[i], data['Upper outliers'][i], str(data['Upper outliers'][i]), ha='center', va='bottom')

    plt.xlabel('year_month', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(data['year_month']))], data['year_month'], rotation=45)
    plt.title(f'Outliers for {project_name} from each year less than {lower} and greater than {upper}')
    plt.ylabel('Count the number of rows for each year that contain outlier data.')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(f"{project_name}_{lower}_{data['year_month'].name}_Outliers.png"))
    plt.show()


def plot_show_shape_outlier_year(data: pd.DataFrame, lower: pd.Timedelta, upper: pd.Timedelta, project_name: str):
    plt.figure(figsize=(15, 10))
    bar_width = 0.25
    r1 = np.arange(len(data['year']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, data['Lower outliers'], color='blue', width=bar_width, edgecolor='grey', label='Lower outliers')
    plt.bar(r2, data['Normal'], color='green', width=bar_width, edgecolor='grey', label='Normal')
    plt.bar(r3, data['Upper outliers'], color='red', width=bar_width, edgecolor='grey', label='Upper outliers')

    for i in range(len(data)):
        plt.text(r1[i], data['Lower outliers'][i], str(data['Lower outliers'][i]), ha='center', va='bottom')
        plt.text(r2[i], data['Normal'][i], str(data['Normal'][i]), ha='center', va='bottom')
        plt.text(r3[i], data['Upper outliers'][i], str(data['Upper outliers'][i]), ha='center', va='bottom')

    plt.xlabel('year', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(data['year']))], data['year'], rotation=45)
    plt.title(f'Outliers for {project_name} from each year less than {lower} and greater than {upper}')
    plt.ylabel('Count the number of rows for each year that contain outlier data.')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(f"{project_name}_{lower}_{data['year'].name}_Outliers.png"))
    plt.show()


if __name__ == '__main__':
    seatunnal = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')

    # Apply the custom clustering function and percentage calculation
    df_calulate_smell_bug = calculate_smell_bug(seatunnal)

    df_normal, df_lower, df_upper, lower, upper = robust_outlier(df_calulate_smell_bug)

    data_years = group_data_by_year(df_normal, df_lower, df_upper)
    data_years_month = group_data_by_year_month(df_normal, df_lower, df_upper)

    # plot_show_shape_outlier_year(data_years, lower, upper, 'seatunnal')
    # plot_show_shape_outlier_year_and_month(data_years_month, lower, upper, 'seatunnal')

    seatunnal_verity = separate_data(df_calulate_smell_bug)
    seatunnal_verity_outlier_upper = separate_data(df_upper)
    seatunnal_verity_outlier_lower = separate_data(df_lower)
    seatunnal_verity_outlier_normal = separate_data(df_normal)


    # Calculate cumulative diff of positive and negative integers
    seatunnal_cumulative = cumulative_diff(seatunnal_verity)
    seatunnal_cumulative_outlier_normal = cumulative_diff(seatunnal_verity_outlier_normal)
    seatunnal_cumulative_outlier_upper = cumulative_diff(seatunnal_verity_outlier_upper)
    seatunnal_cumulative_outlier_lower = cumulative_diff(seatunnal_verity_outlier_lower)

    # Separate the data by year and month
    year_2021, year_2022, year_2023, year_2024 = separate_year_month(seatunnal_cumulative)
    year_2021_normal_outlier, year_2022_normal_outlier, year_2023_normal_outlier, year_2024_normal_outlier = separate_year_month(
        seatunnal_cumulative_outlier_normal)
    year_2021_lower_outlier, year_2022_lower_outlier, year_2023_lower_outlier, year_2024_lower_outlier = separate_year_month(seatunnal_cumulative_outlier_lower)
    year_2021_upper_outlier, year_2022_upper_outlier, year_2023_upper_outlier, year_2024_upper_outlier = separate_year_month(seatunnal_cumulative_outlier_upper)


    years = [year_2021, year_2022, year_2023, year_2024]
    years_outlier_naormal = [year_2021_normal_outlier, year_2022_normal_outlier, year_2023_normal_outlier, year_2024_normal_outlier]
    years_outlier_lower = [year_2021_lower_outlier, year_2022_lower_outlier, year_2023_lower_outlier, year_2024_lower_outlier]
    years_outlier_upper = [year_2021_upper_outlier, year_2022_upper_outlier, year_2023_upper_outlier, year_2024_upper_outlier]

    # Plot the cumulative differences for each year
    # plot_years(years)
    # plot_years(years_outlier_naormal, outlier=True)
    # plot_years(years_outlier_lower, outlier=True)
    # plot_years(years_outlier_upper, outlier=True)


