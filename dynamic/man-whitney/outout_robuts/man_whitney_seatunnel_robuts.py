import itertools
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta for two distributions.
    """
    x, y = np.array(x), np.array(y)
    n_x, n_y = len(x), len(y)
    greater = np.sum(x_i > y_j for x_i in x for y_j in y)
    less = np.sum(x_i < y_j for x_i in x for y_j in y)
    delta = (greater - less) / (n_x * n_y)
    return delta

def load_data(file_path, rule_paths):
    """
    Load main dataset and rule category datasets.
    """
    df = pd.read_pickle(file_path)
    rule_smell_bug = pd.read_pickle(rule_paths['bug'])
    rule_smell_vulnerability = pd.read_pickle(rule_paths['vulnerability'])
    rule_smell_normal = pd.read_pickle(rule_paths['normal'])
    return df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal

# def split_data_by_quantiles(df, column, lower_quantile, upper_quantile):
#     """
#     Split data into lower and upper quantiles.
#     """
#     sorted_df = df.sort_values(by=column, ascending=True)
#     q1 = sorted_df[column].quantile(lower_quantile)
#     q3 = sorted_df[column].quantile(upper_quantile)
#     return sorted_df[sorted_df[column] <= q1], sorted_df[sorted_df[column] >= q3]

def calculate_mad_outliers(data, threshold=3):
    # คำนวณค่ามัธยฐาน
    median = np.median(data)
    # คำนวณ MAD
    mad = np.median(np.abs(data - median))
    # คำนวณขอบเขต
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    # หาค่า Outlier
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return lower_bound, upper_bound, outliers

    lower, upper, outliers = calculate_mad_outliers(data)

    return lower, upper

def analyze_combinations(q_data):
    """
    Analyze combinations of metrics and compute statistical results.
    """
    results = []
    target_columns = [col for col in q_data.columns if col.startswith('java:') and col.endswith('_created')]
    combinations = list(itertools.product(target_columns, ['total_time']))

    for col in combinations:
        selected_col = q_data[[col[0], col[1]]]
        group1 = selected_col[selected_col[col[0]] > 1]
        group2 = selected_col[selected_col[col[0]] <= 0]

        if group1.empty or group2.empty:
            continue

        u_statistic, p_val = mannwhitneyu(group1[col[1]], group2[col[1]])
        cliff_delta = cliffs_delta(group1[col[1]], group2[col[1]])

        results.append({
            'metric': col[0],
            'u_statistic': u_statistic,
            'p_value': p_val,
            'd_value': cliff_delta,
            'smell_count': group1.iloc[:, 0].count(),
            'no_smell_count': group2.iloc[:, 0].count(),
            'smell_sum': group1.iloc[:, 0].sum(),
            'no_smell_sum': group2.iloc[:, 0].sum(),
            'time_modify_smell': group1.iloc[:, 1].mean(),
            'time_modify_smell_min': group1.iloc[:, 1].min(),
            'time_modify_smell_max': group1.iloc[:, 1].max(),
            'time_modify_no_smell': group2.iloc[:, 1].mean(),
            'time_modify_no_smell_min': group2.iloc[:, 1].min(),
            'time_modify_no_smell_max': group2.iloc[:, 1].max()
        })

    return results

def map_categories(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal):
    """
    Map metrics to categories and classify effect sizes and significance.
    """
    keywords_to_remove = ['_created']
    for keyword in keywords_to_remove:
        results_df['key'] = results_df['metric'].str.replace(keyword, '', regex=False)

    category_mapping = {
        **dict.fromkeys(rule_smell_bug['key'], 'bug'),
        **dict.fromkeys(rule_smell_vulnerability['key'], 'vulnerability'),
        **dict.fromkeys(rule_smell_normal['key'], 'normal')
    }
    results_df['category'] = results_df['key'].map(category_mapping).fillna('nan')

    results_df['eff_size'] = results_df['d_value'].apply(
        lambda i: 'small' if 0.147 < i < 0.33 else
        'medium' if 0.33 < i < 0.474 else
        'large' if i > 0.474 else
        'negligible'
    )

    results_df['significant'] = results_df['p_value'].apply(
        lambda i: 'significant' if i < 0.01 else 'not significant')

    return results_df

if __name__ == "__main__":
    # Test the functions
    # Define file paths
    file_path = "../../output/seatunnel_compare.pkl"
    rule_paths = {
        'bug': '../../Sonar/output/sonar_rules_bug_version9.9.6.pkl',
        'vulnerability': '../../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl',
        'normal': '../../Sonar/output/sonar_rules_version9.9.6.pkl'
    }

    # Load data
    df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal = load_data(file_path, rule_paths)

    # Split data into quantiles
    q1_data, q3_data = calculate_mad_outliers(df, )

    # Analyze combinations
    results_q1 = analyze_combinations(q1_data)
    results_q3 = analyze_combinations(q3_data)

    # Create results DataFrame
    results_df_q1 = pd.DataFrame(results_q1)
    results_df_q3 = pd.DataFrame(results_q3)

    # Map categories and classify results
    results_df_q1 = map_categories(results_df_q1, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)
    results_df_q3 = map_categories(results_df_q3, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)

    print(results_df_q1.head())
    print(results_df_q3.head())