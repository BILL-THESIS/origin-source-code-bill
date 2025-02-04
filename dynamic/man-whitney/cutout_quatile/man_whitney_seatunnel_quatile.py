import pandas as pd

# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,\
    f1_score, roc_auc_score, roc_curve, auc
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from cliffs_delta import cliffs_delta
from sklearn.metrics import mean_squared_error
import itertools

# Tree Visualisation

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
                'd_value': abs(cliff_delta[0]),
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

    results_df['significant'] = results_df['p_value'].apply(
        lambda i: 'significant' if i < 0.01 else 'not significant')

    return results_df


if __name__ == "__main__":
    # Test the functions
    # Define file paths
    file_path = "../../output/output/seatunnel_compare.pkl"
    rule_paths = {
        'bug': '../../../Sonar/output/sonar_rules_bug_version9.9.6.pkl',
        'vulnerability': '../../../Sonar/output/sonar_rules_VULNERABILITY_version9.9.6.pkl',
        'normal': '../../../Sonar/output/sonar_rules_version9.9.6.pkl'
    }
    f = pd.read_pickle('../../output/output/seatunnel_compare.pkl')

    # Load data
    df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal = load_data(file_path, rule_paths)

    # Split data into quantiles
    q1_data, q3_data = split_data_by_quantiles(df, 'total_time', 0.25, 0.75)

    # Analyze combinations
    results = analyze_quartile(q1_data, q3_data)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_pickle("../../output/seatunnel_all_status_significant.pkl")

    # Map categories and classify results
    results_df = map_categories(results_df, rule_smell_bug, rule_smell_vulnerability, rule_smell_normal)
    print(results_df.head())

    # Get significant and large effect size data
    s_data = results_df[(results_df['significant'] == 'significant') & (results_df['eff_size'] == 'large')]
    s_data_singifcant = results_df[results_df['significant'] == 'significant']
    s_data_singifcant.to_pickle("../../output/seatunnel_significant.pkl")

    # Get category counts and percentages
    category_counts = s_data_singifcant['eff_size'].value_counts()
    category_percentages_s = (category_counts / len(s_data_singifcant)) * 100
    category_percentages_a = (category_counts / len(results_df)) * 100

    # QR 2 What are the important factors that affect the time to modify a smell in the lower and upper quantiles?
    # Data collection q1_data, q3_data
    # Tag label 0 for q1_data and 1 for q3_data
    q1_data['time_class'] = 0
    q3_data['time_class'] = 1
    data_prepare = pd.concat([q1_data, q3_data])
    hours = pd.to_timedelta(data_prepare['total_time']).dt.total_seconds() / 3600

    #Select the factors that affect significant
    data_prepare_significant = data_prepare[data_prepare.columns.intersection(s_data_singifcant['metric'])].fillna(0)

    X = data_prepare_significant.astype(int)
    y = data_prepare['time_class']

    # trian the model
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('y train:', y_train.value_counts().to_markdown())

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('y test:', y_test.value_counts().to_markdown())


    # Evaluate the model all factors
    results_evaluated_all =  []
    results_evaluated_all.append({
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred,average='macro'),
        'recall': recall_score(y_test, y_pred,average='macro'),
        'f1_score': f1_score(y_test, y_pred,average='macro'),
        'confusion': confusion_matrix(y_test, y_pred)
    })

    results_evaluated_all = pd.DataFrame(results_evaluated_all)


    # Evaluate the model one factor at a time
    results_metrics_list = []

    # Loop through each feature (column) in X to calculate the metrics
    for column in X.columns:
        # For each feature, consider it as a target and calculate the metrics
        X_train_column = X_train[[column]]
        X_test_column = X_test[[column]]

        # Fit the model and predict
        model.fit(X_train_column, y_train)
        y_pred_column = model.predict(X_test_column)

        # Evaluate performance for this feature
        results_metrics_list.append({
            'metric': column,
            'accuracy': accuracy_score(y_test, y_pred_column),
            'precision': precision_score(y_test, y_pred_column,average='macro'),
            'recall': recall_score(y_test, y_pred_column,average='macro'),
            'f1_score': f1_score(y_test, y_pred_column,average='macro'),
            'confusion': confusion_matrix(y_test, y_pred_column)
        })

    # Convert the results into a DataFrame for easy inspection
    results_metrics_list = pd.DataFrame(results_metrics_list)
    print("The results for all feature:")
    print(results_evaluated_all.describe().to_markdown())
    print("The results for each feature:")
    print(results_metrics_list.describe().to_markdown())
