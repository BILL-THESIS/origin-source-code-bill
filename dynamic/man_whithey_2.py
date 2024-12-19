import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np


def calculate_cliffs(data: pd.DataFrame, suffix: str):
    results = []
    for col in data.columns:
        if col.startswith('java:') and col.endswith(suffix):
            smell_data = data[data[col] == 1]['total_time']
            no_smell_data = data[data[col] == 0]['total_time']

            if not smell_data.empty and not no_smell_data.empty:
                # Perform Mann-Whitney U Test
                u_statistic, p_value = mannwhitneyu(smell_data, no_smell_data, alternative='two-sided')

                # Store results
                results.append({
                    'metric': col,
                    'u_statistic': u_statistic,
                    'p_value': p_value,
                    'smell_count': smell_data.count(),
                    'no_smell_count': no_smell_data.count(),
                    'time_modify_smell': smell_data.mean(),
                    'time_modify_smell_min': smell_data.min(),
                    'time_modify_smell_max': smell_data.max(),
                    'time_modify_no_smell': no_smell_data.mean(),
                    'time_modify_no_smell_min': no_smell_data.min(),
                    'time_modify_no_smell_max': no_smell_data.max()
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def interpret_results(results_df: pd.DataFrame, alpha=0.05):
    """ตีความผลลัพธ์และเพิ่มการตีความ Cliff's Delta"""
    results_df['significant'] = results_df['p_value'] < alpha
    results_df['interpretation'] = np.where(
        results_df['significant'],
        'statistically significant',
        'not statistically significant'
    )
    return results_df


def analyze_suffix(data: pd.DataFrame, suffix: str, alpha=0.05):
    results_df = calculate_cliffs(data, suffix)
    interpreted_results = interpret_results(results_df, alpha=alpha)
    return interpreted_results


if __name__ == "__main__":
    # Load data
    file_path = "seatunnel_compare.pkl"
    data = pd.read_pickle(file_path)

    # rule category
    rule_smell_bug = pd.read_pickle('../Sonar/output/sonar_rules_bug_version9.9.6.pkl')
    rule_smell_vulnerability = pd.read_pickle('../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl')
    rule_smell_normal = pd.read_pickle('../Sonar/output/sonar_rules_version9.9.6.pkl')

    # Analyze metrics for each suffix
    # suffixes = ['_created', '_ended', '_diff']
    suffixes = ['_ended']
    results = {}

    for suffix in suffixes:
        results[suffix] = analyze_suffix(data, suffix)

    # Combine and save results
    combined_results = pd.concat(results.values())

    # Remove specific keywords from the 'metric' column
    keywords_to_remove = ['_ended']
    for keyword in keywords_to_remove:
        combined_results['key'] = combined_results['metric'].str.replace(keyword, '', regex=False)

    # the rule category
    category_mapping = {
        **dict.fromkeys(rule_smell_bug['key'], 'bug'),
        **dict.fromkeys(rule_smell_vulnerability['key'], 'vulnerability'),
        **dict.fromkeys(rule_smell_normal['key'], 'normal')
    }

    combined_results['category'] = combined_results['key'].map(category_mapping).fillna('nan')

    # Display summary
    print("=== Analysis Summary ===")
    for suffix, result in results.items():
        print(f"\n--- {suffix} ---")
        print(result[['metric', 'u_statistic', 'p_value', 'interpretation']].head())
