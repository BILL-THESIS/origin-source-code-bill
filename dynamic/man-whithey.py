import pandas as pd
from scipy.stats import mannwhitneyu
import itertools

def analyze_mann_whitney(data_path):
    # Load data
    data = pd.read_pickle(data_path)

    # Filter relevant columns
    java_columns = [col for col in data.columns if col.startswith('java:') and col.endswith('_ended')]
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
            # Perform Mann-Whitney U Test
            u_statistic, p_value = mannwhitneyu(
                greater_than_zero['total_time'],
                equal_to_zero['total_time'],
                alternative='two-sided'
            )

            # Store results
            results.append({
                'metric': greater_than_zero.columns[0],
                'u_statistic': u_statistic,
                'p_value': p_value,
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

if __name__ == "__main__":
    # File paths
    file_path = "seatunnel_compare.pkl"
    bug_rules_path = '../Sonar/output/sonar_rules_bug_version9.9.6.pkl'
    vuln_rules_path = '../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl'
    normal_rules_path = '../Sonar/output/sonar_rules_version9.9.6.pkl'

    # Perform analysis
    results_mann_ehitney = analyze_mann_whitney(file_path)



