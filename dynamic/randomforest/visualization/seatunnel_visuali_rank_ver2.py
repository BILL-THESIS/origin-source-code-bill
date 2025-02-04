import pandas as pd


def load_data():
    files = {
        "significant": "../../output/output/seatunnel_all_status_significant.pkl",
        "group_smell": "../../output/output/seatunnel_rdf_quantile_all.pkl",
        "each_smell": "../../output/output/seatunnel_rdf_quantile_each_smell.pkl",
        "main_group": "../../output/output/seatunnel_correlation_main_group_4.pkl",
        "original": "../../output/output/seatunnel_compare.pkl"
    }
    return {key: pd.read_pickle(path) for key, path in files.items()}


def rank_features(data_each_smell):
    data_each_smell['rank'] = data_each_smell['test_f1'].rank(ascending=False)
    return data_each_smell


def group_and_merge(data_each_smell, data_main_group, data_qr1):
    groups = [data_each_smell[data_each_smell["features"].isin(data_main_group[i])] for i in range(4)]
    return [group.merge(data_qr1, left_on='features', right_on='metric', how='left') for group in groups]


def compute_ranks(groups):
    for group in groups:
        for col in ['test_f1', 'test_roc_auc', 'd_value']:
            group[f'rank_{col.split("_")[1]}'] = group[col].rank(ascending=False)
            print(group.columns)
    return [group[['features', 'rank_f1', 'rank_roc', 'rank_value']] for group in groups]


def filter_by_f1_threshold(data_group_smell, threshold=0.8):
    return data_group_smell.loc[data_group_smell['test_f1'] >= threshold]


def compute_rank_sums(filtered_data, group_join, rank_col):
    rank_data = filtered_data['features'].apply(
        lambda s: [group_join.set_index('features').loc[x, rank_col] for x in s])
    df_split = pd.DataFrame(rank_data.tolist(), columns=["G1", "G2", "G3", "G4"], index=filtered_data.index)

    rank_data = pd.concat([filtered_data[['features', rank_col.replace("rank_", "test_")]], df_split], axis=1)
    rank_data['sum'] = rank_data[['G1', 'G2', 'G3', 'G4']].sum(axis=1)
    return rank_data


def extract_remaining_features(filtered_data):
    remaining_features = set()
    for features in filtered_data['features']:
        remaining_features.update(features)
    return remaining_features


def main():
    data = load_data()
    data['each_smell'] = rank_features(data['each_smell'])

    groups = group_and_merge(data['each_smell'], data['main_group'], data['significant'])
    ranked_groups = compute_ranks(groups)
    group_join = pd.concat(ranked_groups, axis=0)

    filtered_f1 = filter_by_f1_threshold(data['group_smell'])

    rank_f1 = compute_rank_sums(filtered_f1, group_join, 'rank_f1')
    rank_roc = compute_rank_sums(filtered_f1, group_join, 'rank_roc')
    rank_d = compute_rank_sums(filtered_f1, group_join, 'rank_value')

    remaining_features_f1 = extract_remaining_features(rank_f1[rank_f1['sum'] <= 10])
    remaining_features_roc = extract_remaining_features(rank_roc[rank_roc['sum'] <= 10])
    remaining_features_d = extract_remaining_features(rank_d[rank_d['sum'] <= 10])

    return remaining_features_f1, remaining_features_roc, remaining_features_d


if __name__ == "__main__":
    data = load_data()
    data['each_smell'] = rank_features(data['each_smell'])

    groups = group_and_merge(data['each_smell'], data['main_group'], data['significant'])
    ranked_groups = compute_ranks(groups)
    group_join = pd.concat(ranked_groups, axis=0)

    filtered_f1 = filter_by_f1_threshold(data['group_smell'])

    rank_f1 = compute_rank_sums(filtered_f1, group_join, 'rank_f1')
    rank_roc = compute_rank_sums(filtered_f1, group_join, 'rank_roc')
    rank_d = compute_rank_sums(filtered_f1, group_join, 'rank_value')

    remaining_features_f1 = extract_remaining_features(rank_f1[rank_f1['sum'] <= 10])
    remaining_features_roc = extract_remaining_features(rank_roc[rank_roc['sum'] <= 10])
    remaining_features_d = extract_remaining_features(rank_d[rank_d['sum'] <= 10])

