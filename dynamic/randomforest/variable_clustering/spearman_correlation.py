import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # File paths
    input_filepath = "../../output/seatunnel_compare.pkl"
    data = pd.read_pickle(input_filepath)

    prefix = "java:"
    suffix = "_created"

    selected_cols = [col for col in data.columns if col.startswith(prefix) and col.endswith(suffix)]
    selected_cols = data[selected_cols].fillna(0)

    list_of_cols_x_y = [{col: data[col], 'total_time': data['total_time']} for col in selected_cols]

    list_spearman_correlation = []
    for cols in list_of_cols_x_y:
        x = cols[list(cols.keys())[0]].fillna(0).astype('float64')
        y = (cols['total_time'].dt.total_seconds() / 3600.0).astype('float64')

        rank_x = x.rank()
        print(rank_x)
        rank_y = y.rank()
        print(rank_y)
        spearman_rank, p_value =  stats.spearmanr(rank_x, rank_y)

        # spearman^2
        spearman_rank_squared = spearman_rank ** 2

        list_spearman_correlation.append({'col': x.name ,
                                          'spearman_rank': spearman_rank,
                                          'spearman_rank_squared': spearman_rank_squared,
                                          'p_value': p_value})

        df = pd.DataFrame(list_spearman_correlation).fillna(0)

    conditions_r = [
        (df['spearman_rank'] <= 0.1),
        (df['spearman_rank'] <= 0.3),
        (df['spearman_rank'] <= 0.5),
        (df['spearman_rank'] <= 0.7),
        (df['spearman_rank'] > 0.7)
    ]

    conditions_r_squared = [
        (df['spearman_rank_squared'] <= 0.0),
        (df['spearman_rank_squared'] <= 0.25),
        (df['spearman_rank_squared'] <= 0.64),
        (df['spearman_rank_squared'] > 0.64)
    ]

    choices_r = ['no correlation', 'low correlation', 'moderate correlation', 'high correlation', 'very high correlation']
    choices_r_squared = ['no correlation', 'moderate correlation', 'high correlation', 'very high correlation']
    df['group_r'] = np.select(conditions_r, choices_r)
    df['group_r_squared'] = np.select(conditions_r_squared, choices_r_squared)

    # percentage of correlation
    no_corr = df[df['group_r'] == 'no correlation'].shape[0] / df.shape[0] * 100
    low_corr = df[df['group_r'] == 'low correlation'].shape[0] / df.shape[0] * 100
    moderate_corr = df[df['group_r'] == 'moderate correlation'].shape[0] / df.shape[0] * 100
    high_corr = df[df['group_r'] == 'high correlation'].shape[0] / df.shape[0] * 100

    df_status = pd.DataFrame({'group_r': ['no correlation', 'low correlation', 'moderate correlation', 'high correlation'],
                              'percentage': [no_corr, low_corr, moderate_corr, high_corr],
                              'shape': [df[df['group_r'] == 'no correlation'].shape[0],
                                        df[df['group_r'] == 'low correlation'].shape[0],
                                        df[df['group_r'] == 'moderate correlation'].shape[0],
                                        df[df['group_r'] == 'high correlation'].shape[0]],
                              'mean_rank': [df[df['group_r'] == 'no correlation']['spearman_rank'].mean(),
                                        df[df['group_r'] == 'low correlation']['spearman_rank'].mean(),
                                        df[df['group_r'] == 'moderate correlation']['spearman_rank'].mean(),
                                        df[df['group_r'] == 'high correlation']['spearman_rank'].mean()],
                              'highest_rank': [df[df['group_r'] == 'no correlation']['spearman_rank'].max(),
                                            df[df['group_r'] == 'low correlation']['spearman_rank'].max(),
                                            df[df['group_r'] == 'moderate correlation']['spearman_rank'].max(),
                                            df[df['group_r'] == 'high correlation']['spearman_rank'].max()],
                              'lowest_rank': [df[df['group_r'] == 'no correlation']['spearman_rank'].min(),
                                               df[df['group_r'] == 'low correlation']['spearman_rank'].min(),
                                               df[df['group_r'] == 'moderate correlation']['spearman_rank'].min(),
                                               df[df['group_r'] == 'high correlation']['spearman_rank'].min()]
                              })

    # Histogram
    plt.hist(df['spearman_rank'], bins=20, alpha=0.5, color='b', label='spearman_rank')
    plt.legend(loc='upper right')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title(f'Spearman Rank Correlation Histogram')
    plt.tight_layout()
    plt.show()


