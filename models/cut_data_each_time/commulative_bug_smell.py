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
    lower = pd.Timedelta(minutes=9)
    upper = median + threshold * MADN

    print(
        "The value of the lower outliers: ", lower,
        "The value of the upper outliers: ", upper)

    # df_outliers = df[~outlier]
    df_lower = df[df['total_time'] < lower]
    df_upper = df[df['total_time'] > upper]

    df_outliers = df[~df.index.isin(df_lower.index) & ~df.index.isin(df_upper.index)]

    return df_outliers, df_lower, df_upper , lower, upper


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
    return df


# supareat years and months
def supareat_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df_2021 = df[df['completed_date'].dt.year == 2021]
    df_2022 = df[df['completed_date'].dt.year == 2022]
    df_2023 = df[df['completed_date'].dt.year == 2023]
    df_2024 = df[df['completed_date'].dt.year == 2024]
    return df_2021, df_2022, df_2023, df_2024


def plot_cumulative_bars_overview(df, project_name):
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

    plt.title(f'Cumulative Differences for {project_name} Shape {df["year_month"].shape}')
    plt.xlabel('Year-Month')
    plt.ylabel('Cumulative Difference')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(f'{project_name}_Cumulative_Differences_Bar.png'))
    plt.show()


def plot_years(years_list, outlier=False):
    for df_year in years_list:
        if df_year.empty:
            continue
        suffix = '_outlier' if outlier else ''
        plot_cumulative_bars_overview(df_year, f'seatunnal_{df_year.iloc[0]["year_month"].year}{suffix}')


def plot_show_shape_outlier(dict_results: dict, lower, upper, project_name: str):
    summary = {
        'Period': [],
        'Outliers': [],
        'Lower Outliers': [],
        'Upper Outliers': []
    }

    for period, (df_outliers, df_lower, df_upper, lower, upper) in dict_results.items():
        print(f"Period: {period}")
        print(
            f"Outliers: {df_outliers.shape[0]}, Lower Outliers: {df_lower.shape[0]}, Upper Outliers: {df_upper.shape[0]}")
        summary['Period'].append(period)
        summary['Outliers'].append(df_outliers.shape[0])
        summary['Lower Outliers'].append(df_lower.shape[0])
        summary['Upper Outliers'].append(df_upper.shape[0])

    summary_df = pd.DataFrame(summary)
    print(summary_df)

    plt.figure(figsize=(15, 10))
    bar_width = 0.25
    r1 = np.arange(len(summary_df['Period']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, summary_df['Lower Outliers'], color='blue', width=bar_width, edgecolor='grey', label='Lower Outliers')
    plt.bar(r2, summary_df['Outliers'], color='green', width=bar_width, edgecolor='grey', label='Outliers')
    plt.bar(r3, summary_df['Upper Outliers'], color='red', width=bar_width, edgecolor='grey', label='Upper Outliers')

    for i in range(len(summary_df)):
        plt.text(r1[i], summary_df['Lower Outliers'][i], str(summary_df['Lower Outliers'][i]), ha='center', va='bottom')
        plt.text(r2[i], summary_df['Outliers'][i], str(summary_df['Outliers'][i]), ha='center', va='bottom')
        plt.text(r3[i], summary_df['Upper Outliers'][i], str(summary_df['Upper Outliers'][i]), ha='center', va='bottom')

    plt.xlabel('Period', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(summary_df['Period']))], summary_df['Period'], rotation=45)
    # plt.title(f'Outliers for {project_name} (Lower: {lower}, Upper: {upper})')
    plt.title(f'Outliers for {project_name} 9 minutes')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(f'{project_name}{lower}_Outliers.png'))
    plt.show()


if __name__ == '__main__':
    seatunnal = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')

    # Apply the custom clustering function and percentage calculation
    seatunnal_calulate_smell_bug = calculate_smell_bug(seatunnal)

    seatunnal_outlier, seatunnal_outliers_low, seatunnal_outliers_up, lower, upper = robust_outlier(seatunnal_calulate_smell_bug)

    grouped = seatunnal_calulate_smell_bug.groupby('year_month')
    results = {period: robust_outlier(group) for period, group in grouped}

    df_save = pd.DataFrame({})
    for period, (df_outliers, df_lower, df_upper, lower, upper) in results.items():
        save = pd.DataFrame({})
        save['Period'] = [period]
        save['Outliers'] = [df_outliers.shape[0]]
        save['Lower shape'] = [df_lower.shape[0]]
        save['lower date'] = [lower]
        save['Upper shape'] = [df_upper.shape[0]]
        save['upper date'] = [upper]
        df_save = pd.concat([df_save, save])

    plot_show_shape_outlier(results, lower,upper,'seatunnal')

    seatunnal_verity = separate_data(seatunnal_calulate_smell_bug)
    seatunnal_verity_outlier = separate_data(seatunnal_outlier)

    # Calculate cumulative diff of positive and negative integers
    seatunnal_cumulative = cumulative_diff(seatunnal_verity)
    seatunnal_cumulative_outlier = cumulative_diff(seatunnal_verity_outlier)

    # Separate the data by year and month
    year_2021, year_2022, year_2023, year_2024 = supareat_year_month(seatunnal_cumulative)
    year_2021_outlier, year_2022_outlier, year_2023_outlier, year_2024_outlier = supareat_year_month(
        seatunnal_cumulative_outlier)

    years = [year_2021, year_2022, year_2023, year_2024]
    years_outlier = [year_2021_outlier, year_2022_outlier, year_2023_outlier, year_2024_outlier]

    # Plot the cumulative differences for each year
    # plot_years(years)
    # plot_years(years_outlier, outlier=True)
