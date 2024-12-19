import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu


def calculate_cliffs(df: pd.DataFrame, suffix: str):
    smell_data = {}
    no_smell_data = {}
    u_statistics = {}
    p_values = {}

    for col in df.columns:
        if col.startswith('java:') and col.endswith(suffix):
            smell_data[col] = df[df[col] == 1]['total_time']
            no_smell_data[col] = df[df[col] == 0]['total_time']

            if not smell_data[col].empty and not no_smell_data[col].empty:
                u_statistic, p_value = mannwhitneyu(smell_data[col], no_smell_data[col], alternative='two-sided')
                u_statistics[col] = u_statistic
                p_values[col] = p_value

    u_statistics = pd.DataFrame(u_statistics, index=['u-statistic']).T
    p_values = pd.DataFrame(p_values, index=['p-value']).T
    smell_data = pd.DataFrame(smell_data)
    no_smell_data = pd.DataFrame(no_smell_data)

    return u_statistics, p_values, smell_data, no_smell_data


def cliffs_delta(lst1, lst2):
    greater = sum(x > y for x in lst1 for y in lst2)
    less = sum(x < y for x in lst1 for y in lst2)
    n = len(lst1) * len(lst2)
    delta = (greater - less) / n
    return delta


def interpret_results(p_values):
    alpha = 0.05
    for i in p_values.index:
        if p_values.loc[i, 'p-value'] < alpha:
            print(f"{i}: statistically significant")
            p_values.loc[i, 'significant'] = True
        else:
            print(f"{i}: not statistically significant")
            p_values.loc[i, 'significant'] = False

    return p_values


if __name__ == "__main__":
    file_path = "seatunnel_compare.pkl"
    data = pd.read_pickle(file_path)

    # Display basic information and preview the data
    data.info()
    data.head()

    # Select columns related to 'java' and 'total_time'
    # java_columns = [col for col in data.columns if col.startswith('java:') and col.endswith('_diff')]
    java_columns = [col for col in data.columns if 'java' in col.lower()]

    selected_columns = java_columns + ['total_time']

    # Filter the data to include only the selected columns
    filtered_data = data[selected_columns]

    # Create the 'total_time_days' column
    filtered_data['total_time_days'] = filtered_data['total_time'].dt.days

    # Drop the original 'total_time' as we now have it in seconds
    filtered_data = filtered_data.drop(columns=['total_time'])

    # Display the processed data
    filtered_data.info()
    filtered_data.head()

    # Define a function to calculate Pearson and Spearman correlations for each feature
    correlation_results = []

    for col in java_columns:
        try:
            # Calculate Pearson and Spearman correlation
            pearson_corr, pearson_pval = pearsonr(filtered_data[col], filtered_data['total_time_days'])
            spearman_corr, spearman_pval = spearmanr(filtered_data[col], filtered_data['total_time_days'])

            # Append results
            correlation_results.append({
                'feature': col,
                'pearson_corr': pearson_corr,
                'pearson_pval': pearson_pval,
                'spearman_corr': spearman_corr,
                'spearman_pval': spearman_pval
            })
        except Exception as e:
            # Handle cases where correlation cannot be computed
            correlation_results.append({
                'feature': col,
                'pearson_corr': None,
                'pearson_pval': None,
                'spearman_corr': None,
                'spearman_pval': None
            })

    # Convert results to a DataFrame for better readability
    correlation_df = pd.DataFrame(correlation_results)

    # Display the top features with the strongest Spearman correlation (absolute value)
    top_spearman = correlation_df.dropna().sort_values(by='spearman_corr', key=abs, ascending=False).head(10)
    top_spearman