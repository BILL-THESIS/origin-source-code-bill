import pandas as pd
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
import numpy as np



def load_data(file_path, rule_paths):
    df = pd.read_pickle(file_path)
    rule_smell_bug = pd.read_pickle(rule_paths['bug'])
    rule_smell_vulnerability = pd.read_pickle(rule_paths['vulnerability'])
    rule_smell_normal = pd.read_pickle(rule_paths['normal'])
    return df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal


def analyze_mann_whitney(data_upper, data_lower):
    data_list = []
    # Iterate through all combinations of sublists in data_upper and data_lower
    for i_upper, i_lower in zip(data_upper, data_lower):
        results = []
        # Identify target columns with specific naming pattern
        target_columns_upper = [col for col in i_upper.columns if col.startswith('java:') and col.endswith('_created')]
        target_columns_lower = [col for col in i_lower.columns if col.startswith('java:') and col.endswith('_created')]

        # Find common columns between upper and lower datasets
        common_columns = set(target_columns_upper) & set(target_columns_lower)

        # Process each common column
        for col in common_columns:
            try:
                # Filter and drop NaN values for the specific column
                data_upper_col = i_upper[[col, 'total_time']].dropna()
                data_lower_col = i_lower[[col, 'total_time']].dropna()

                # Perform Mann-Whitney U test and calculate Cliff's delta
                u_statistic, p_val = mannwhitneyu(data_upper_col[col], data_lower_col[col])
                cliff_delta = cliffs_delta(data_upper_col[col], data_lower_col[col])

                # Collect metrics
                results.append({
                    'metric': col,
                    'u_statistic': u_statistic,
                    'p_value': p_val,
                    'd_value': cliff_delta[0],
                    'percen': i_upper['percentile'].values[0],
                    'smell_count_upper': len(data_upper_col),
                    'smell_count_lower': len(data_lower_col),
                    'smell_sum_upper': data_upper_col[col].sum(),
                    'smell_sum_lower': data_lower_col[col].sum(),
                    'time_modify_smell_upper_mean': data_upper_col['total_time'].mean(),
                    'time_modify_smell_upper_min': data_upper_col['total_time'].min(),
                    'time_modify_smell_upper_max': data_upper_col['total_time'].max(),
                    'time_modify_smell_lower_mean': data_lower_col['total_time'].mean(),
                    'time_modify_smell_lower_min': data_lower_col['total_time'].min(),
                    'time_modify_smell_lower_max': data_lower_col['total_time'].max(),
                    'eff_size': cliff_delta[1]
                })
            except KeyError as ke:
                print(f"KeyError for column {col}: {ke}")
            except ValueError as ve:
                print(f"ValueError for column {col}: {ve}")
            except Exception as e:
                print(f"Unexpected error for column {col}: {e}")
            continue

        data_list.append(pd.DataFrame(results))
    return data_list


def remove_keywords_from_metric(results_dfs, keywords_to_remove):
    data_list = []
    for results_df in results_dfs:
        for keyword in keywords_to_remove:
            results_df['key'] = results_df['metric'].str.replace(keyword, '', regex=False)
        data_list.append(results_df)
    return data_list


def map_rule_category(results_dfs, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal):
    list = []
    for results_df in results_dfs:
        category_mapping = {
            **dict.fromkeys(rule_smell_bug['key'], 'bug'),
            **dict.fromkeys(rule_smell_vulnerability['key'], 'vulnerability'),
            **dict.fromkeys(rule_smell_normal['key'], 'normal')
        }
        results_df['category'] = results_df['key'].map(category_mapping).fillna('nan')
        list.append(results_df)
    return list


def calculate_significance(results_dfs):
    data_list = []
    for results_df in results_dfs:
        results_df['significant'] = results_df['p_value'].apply(
            lambda i: 'significant' if i >= 0.01 else 'not significant')
        data_list.append(results_df)
    return data_list


def analyze_high_low_data(results_df, percentiles=[0.1, 0.15, 0.2, 0.25, 0.3]):
    df_sorted = results_df.sort_values(by='total_time', ascending=False)
    top_percentiles = [
        df_sorted.iloc[:int(p * len(df_sorted))].reset_index(drop=True).assign(percentile=f"Top {int(p * 100)}%")
        for p in percentiles
    ]

    df_sorted_ascending = results_df.sort_values(by='total_time', ascending=True)
    bottom_percentiles = [
        df_sorted_ascending.iloc[:int(p * len(df_sorted_ascending))].reset_index(drop=True).assign(
            percentile=f"Bottom {int(p * 100)}%")
        for p in percentiles
    ]

    return top_percentiles, bottom_percentiles


if __name__ == "__main__":
    # Define file paths
    file_path = "../../output/pulsar_compare.pkl"
    rule_paths = {
        'bug': '../../Sonar/output/sonar_rules_bug_version9.9.6.pkl',
        'vulnerability': '../../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl',
        'normal': '../../Sonar/output/sonar_rules_version9.9.6.pkl'
    }

    # Load data
    df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal = load_data(file_path, rule_paths)

    # Remove specific keywords from the 'metric' column
    keywords_to_remove = ['_created']
    upper_group, low_group = analyze_high_low_data(df)

    # Perform analysis
    results = analyze_mann_whitney(upper_group, low_group)
    results = remove_keywords_from_metric(results, keywords_to_remove)
    results = map_rule_category(results, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)
    results = calculate_significance(results)


    s_data = [df[df['significant'] == 'significant'] for df in results]
