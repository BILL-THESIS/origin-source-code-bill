from cliffs_delta import cliffs_delta
import pandas as pd
from scipy.stats import mannwhitneyu


def load_data(file_path, rule_paths):
    df = pd.read_pickle(file_path)
    rule_smell_bug = pd.read_pickle(rule_paths['bug'])
    rule_smell_vulnerability = pd.read_pickle(rule_paths['vulnerability'])
    rule_smell_normal = pd.read_pickle(rule_paths['normal'])
    return df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal


def split_data_by_quantiles(df, column, lower_quantile, upper_quantile):
    # Split data into lower and upper quantiles.
    sorted_df = df.sort_values(by=column, ascending=True)
    q1 = sorted_df[column].quantile(lower_quantile)
    q3 = sorted_df[column].quantile(upper_quantile)
    return sorted_df[sorted_df[column] <= q1], sorted_df[sorted_df[column] >= q3]


def analyze_quartile(q1_data, q3_data):
    results = []

    target_columns_q1 = [col for col in q1_data.columns if col.startswith('java:') and col.endswith('_created')]
    target_columns_q3 = [col for col in q3_data.columns if col.startswith('java:') and col.endswith('_created')]
    common_columns = set(target_columns_q1) & set(target_columns_q3)

    for col in common_columns:
        try:
            data_q1 = q1_data[[col, 'total_time']].dropna()
            data_q3 = q3_data[[col, 'total_time']].dropna()

            u_statistic, p_val = mannwhitneyu(data_q1[col], data_q3[col])
            cliff_delta = cliffs_delta(data_q1[col], data_q3[col])

            results.append({
                'metric': col,
                'u_statistic': u_statistic,
                'p_value': p_val,
                'd_value': cliff_delta[0],
                'smell_count_q1': data_q1[col].count(),
                'smell_count_q3': data_q3[col].count(),
                'smell_sum_q1': data_q1[col].sum(),
                'smell_sum_q3': data_q3[col].sum(),
                'time_modify_smell_q1': data_q1['total_time'].mean(),
                'time_modify_smell_min_q1': data_q1['total_time'].min(),
                'time_modify_smell_max_q1': data_q1['total_time'].max(),
                'time_modify_smell_q3': data_q3['total_time'].mean(),
                'time_modify_smell_min_q3': data_q3['total_time'].min(),
                'time_modify_smell_max_q3': data_q3['total_time'].max(),
                'eff_size': cliff_delta[1]
            })

        except ValueError as e:
            print(f"Error performing statistical test for column {col}: {e}")
            continue

    return results


def map_categories(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal):
    keywords_to_remove = ['_created']
    for keyword in keywords_to_remove:
        results_df['key'] = results_df['metric'].str.replace(keyword, '', regex=False)

    category_mapping = {
        **dict.fromkeys(rule_smell_bug['key'], 'bug'),
        **dict.fromkeys(rule_smell_vulnerability['key'], 'vulnerability'),
        **dict.fromkeys(rule_smell_normal['key'], 'normal')
    }
    results_df['category'] = results_df['key'].map(category_mapping).fillna('nan')

    # results_df['eff_size'] = results_df['d_value'].apply(
    #     lambda i: 'small' if 0.147 < i < 0.33 else
    #     'medium' if 0.33 < i < 0.474 else
    #     'large' if i > 0.474 else
    #     'negligible'
    # )

    results_df['significant'] = results_df['p_value'].apply(
        lambda i: 'significant' if i < 0.01 else 'not significant')

    return results_df


if __name__ == "__main__":
    # Test the functions
    # Define file paths
    file_path = "../../output/pulsar_compare.pkl"
    rule_paths = {
        'bug': '../../../Sonar/output/sonar_rules_bug_version9.9.6.pkl',
        'vulnerability': '../../../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl',
        'normal': '../../../Sonar/output/sonar_rules_version9.9.6.pkl'
    }

    # Load data
    df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal = load_data(file_path, rule_paths)

    # Split data into quantiles
    q1_data, q3_data = split_data_by_quantiles(df, 'total_time', 0.25, 0.75)

    # Analyze combinations
    results = analyze_quartile(q1_data, q3_data)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Map categories and classify results
    results_df = map_categories(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)

    print(results_df.head())

    # s_data = results_df[(results_df['significant'] == 'significant') & (results_df['eff_size'] == 'large')]
    s_data_singifcant = results_df[results_df['significant'] == 'significant']
    s_data_singifcant.to_pickle('../../output/pulsar_quatile_significant.pkl')