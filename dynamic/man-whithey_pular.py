import pandas as pd
from scipy.stats import mannwhitneyu
import itertools
import numpy as np


def cliffs_delta(x, y):
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    greater = np.sum(x_i > y_j for x_i in x for y_j in y)
    less = np.sum(x_i < y_j for x_i in x for y_j in y)
    delta = (greater - less) / (n_x * n_y)
    return delta


def analyze_mann_whitney(df, p_value_threshold=0.01):
    # Load data
    data = df

    # Filter relevant columns
    java_columns = [col for col in data.columns if col.startswith('java:') and col.endswith('_ended')]
    java_columns_created = [col for col in data.columns if col.startswith('java:') and col.endswith('_created')]
    java_columns_diff = [col for col in data.columns if col.startswith('java:') and col.endswith('_diff')]
    print("The number of columns ended", len(java_columns))
    print("The number of columns created", len(java_columns_created))
    print("The number of columns diff", len(java_columns_diff))
    time_col = ['total_time']

    # Generate column combinations
    combinations = list(itertools.product(java_columns, time_col))

    # Prepare data for analysis
    col_list = []
    for col_combi_list in combinations:
        selection_col = data[[col_combi_list[0], col_combi_list[1]]]
        greater_than_zero = selection_col[selection_col[col_combi_list[0]] > 0]
        equal_to_zero = selection_col[selection_col[col_combi_list[0]] == 0]
        col_list.append((greater_than_zero, equal_to_zero))

    # Perform Mann-Whitney U Test and collect results
    results = []
    for greater_than_zero, equal_to_zero in col_list:
        if not greater_than_zero.empty and not equal_to_zero.empty:
            u_statistic, p_value = mannwhitneyu(
                greater_than_zero['total_time'],
                equal_to_zero['total_time'],
                alternative='two-sided'
            )

            cliff_delta = cliffs_delta(
                greater_than_zero['total_time'].values,
                equal_to_zero['total_time'].values
            )

            # Store results
            results.append({
                'metric': greater_than_zero.columns[0],
                'u_statistic': u_statistic,
                'p_value': p_value,
                'd_value': cliff_delta,
                'smell_count': greater_than_zero.iloc[:, 0].count(),
                'no_smell_count': equal_to_zero.iloc[:, 0].count(),
                'smell_sum': greater_than_zero.iloc[:, 0].sum(),
                'no_smell_sum': equal_to_zero.iloc[:, 0].sum(),
                'time_modify_smell': greater_than_zero.iloc[:, 1].mean(),
                'time_modify_smell_min': greater_than_zero.iloc[:, 1].min(),
                'time_modify_smell_max': greater_than_zero.iloc[:, 1].max(),
                'time_modify_no_smell': equal_to_zero.iloc[:, 1].mean(),
                'time_modify_no_smell_min': equal_to_zero.iloc[:, 1].min(),
                'time_modify_no_smell_max': equal_to_zero.iloc[:, 1].max()
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def remove_keywords_from_metric(results_df, keywords_to_remove):
    for keyword in keywords_to_remove:
        results_df['metric'] = results_df['metric'].str.replace(keyword, '', regex=False)
    return results_df


def map_rule_category(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal):
    category_mapping = {
        **dict.fromkeys(rule_smell_bug['key'], 'bug'),
        **dict.fromkeys(rule_smell_vulnerability['key'], 'vulnerability'),
        **dict.fromkeys(rule_smell_normal['key'], 'normal')
    }
    results_df['category'] = results_df['metric'].map(category_mapping).fillna('nan')
    return results_df


def calculate_effect_size(results_df):
    results_df['eff_size'] = results_df['d_value'].apply(
        lambda i: 'small' if 0.147 < i < 0.33 else
        'medium' if 0.33 < i < 0.474 else
        'large' if i > 0.474 else
        'negligible'
    )
    return results_df


def calculate_significance(results_df):
    results_df['significant'] = results_df['p_value'].apply(lambda i: 'significant' if i < 0.05 else 'not significant')
    return results_df


def analyze_high_low_delta(results_df, high_percentiles=[0.1, 0.15, 0.2, 0.25, 0.3], low_percentiles=[0.1, 0.15, 0.2, 0.25, 0.3]):

    # เรียงลำดับจากมากไปน้อย
    sorted_more = results_df.sort_values(by='d_value', ascending=False)
    sorted_less = results_df.sort_values(by='d_value', ascending=True)

    high_groups = {}
    low_groups = {}

    # สำหรับแต่ละเปอร์เซ็นต์ที่กำหนดใน high_percentiles
    for high_percentile in high_percentiles:
        high_groups[high_percentile] = sorted_more.sample(frac=high_percentile)

    # สำหรับแต่ละเปอร์เซ็นต์ที่กำหนดใน low_percentiles
    for low_percentile in low_percentiles:
        low_groups[low_percentile] = sorted_less.sample(frac=low_percentile)

    return high_groups, low_groups



if __name__ == "__main__":
    # File paths
    file_path = "output/pulsar_compare.pkl"
    data = pd.read_pickle(file_path)
    data = data.dropna()
    # rule category
    rule_smell_bug = pd.read_pickle('../Sonar/output/sonar_rules_bug_version9.9.6.pkl')
    rule_smell_vulnerability = pd.read_pickle('../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl')
    rule_smell_normal = pd.read_pickle('../Sonar/output/sonar_rules_version9.9.6.pkl')

    # Perform analysis
    results_df = analyze_mann_whitney(data)

    # Remove specific keywords from the 'metric' column
    keywords_to_remove = ['_ended']
    results_df = remove_keywords_from_metric(results_df, keywords_to_remove)
    results_df = map_rule_category(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)
    results_df = calculate_effect_size(results_df)
    results_df = calculate_significance(results_df)

    results = results_df[(results_df['significant'] == 'significant') & (results_df['eff_size'] == 'negligible')]
    r_significant = results_df[results_df['significant'] == 'significant']

    high_group, low_group = analyze_high_low_delta(r_significant)


