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
    # df['value_ended'] = pd.to_numeric(df['value_ended'], errors='coerce')
    # df['value_created'] = pd.to_numeric(df['value_created'], errors='coerce')
    # df['diff_bug'] = df['value_ended'] - df['value_created']
    # df['percentage_bug'] = ((df['value_ended'] - df['value_created']) / df['value_created']) * 100

    # for col in ['d', 'b', 'cp', 'c', 'ooa', 'u']:
    #     df[f'diff_{col}'] = df[f'ended_{col}'] - df[f'created_{col}']
    #     df[f'percentage_{col}'] = ((df[f'ended_{col}'] - df[f'created_{col}']) / df[f'created_{col}']) * 100

    # Add a new column for year-month
    df['year'] = pd.to_datetime(df['completed_date']).dt.to_period('Y')
    df['year_month'] = pd.to_datetime(df['completed_date']).dt.to_period('M')
    df['year_month_day'] = pd.to_datetime(df['completed_date']).dt.to_period('D')
    return df





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



if __name__ == '__main__':
    seatunnal = pd.read_pickle('../../Sonar/output/tag_bug/seatunnal_bug_comapare_time.pkl')
    pulsar = pd.read_pickle('/Users/bill/origin-source-code-bill/Github/output/pulsar_filtered_issue_bug.pkl')


    # Apply the custom clustering function and percentage calculation
    df_calulate_smell_bug = calculate_smell_bug(seatunnal)
    df_calulate_smell_bug_pulsar = calculate_smell_bug(pulsar)

    # Count the occurrences of each year-month
    year_month_counts = df_calulate_smell_bug['year_month'].value_counts().sort_index()
    year_month_counts_pulsar = df_calulate_smell_bug_pulsar['year_month'].value_counts().sort_index()
    print(year_month_counts)

    # year_month_counts.plot(kind='line', figsize=(10, 5))
    # plt.title('Year-Month Counts')
    # plt.xlabel('Year-Month')
    # plt.ylabel('Counts')
    # # plt.grid(True)
    # plt.show()


    # year_month_counts_pulsar.plot(kind='line', figsize=(10, 5))
    # plt.title('Year-Month Counts')
    # plt.xlabel('Year-Month')
    # plt.ylabel('Counts')
    # for i, v in enumerate(year_month_counts_pulsar):
    #     plt.text(i, v, str(v), ha='center', va='bottom')
    # plt.savefig('pulsar_year_month_counts.png')
    # plt.grid(True)
    # plt.show()

    year_month_counts_pulsar.plot(kind='line', figsize=(10, 5))
    plt.title('Year-Month Counts')
    plt.xlabel('Year-Month')
    plt.ylabel('Counts')
    # plt.xticks(np.arange(len(year_month_counts_pulsar)), year_month_counts_pulsar.index, rotation=45)
    for i, v in enumerate(year_month_counts_pulsar):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.savefig('pulsar_year_month_counts.png')
    plt.grid(True)
    plt.show()


    top_n = 5
    top_year_month_counts_pulsar = year_month_counts_pulsar.nlargest(top_n)

    top_year_month_counts_pulsar.plot(kind='bar', figsize=(10, 5))
    plt.title(f'Top {top_n} Year-Month Counts')
    plt.xlabel('Year-Month')
    plt.ylabel('Counts')
    for i, v in enumerate(top_year_month_counts_pulsar):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.savefig('pulsar_top_year_month_counts.png')
    plt.show()


    
    
    
    




