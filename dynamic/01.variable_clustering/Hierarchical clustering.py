import joblib
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


def polt_spearman(df, t=0.5, method='average', maxclust=None, project_name=str):
    # Pivot to correlation matrix
    corr = df.pivot(index='col1', columns='col2', values='spearman_rank')
    corr = corr.combine_first(corr.T)

    labels = sorted(set(df['col1']).union(set(df['col2'])))
    corr = corr.reindex(index=labels, columns=labels)

    for lab in labels:
        corr.loc[lab, lab] = 1.0
    corr = corr.fillna(0)

    dist = 1 - corr
    condensed = squareform(dist.values)

    link = linkage(condensed, method=method)

    # Cut dendrogram
    if maxclust:
        clusters = fcluster(link, t=maxclust, criterion='maxclust')
    else:
        clusters = fcluster(link, t=t, criterion='distance')

    cluster_df = pd.DataFrame({'feature': labels, 'cluster': clusters})

    # Plot dendrogram
    plt.figure(figsize=(8, 5))
    dendrogram(link, labels=labels, orientation='top')
    plt.xticks(rotation=90)
    plt.title(f'Hierarchical Clustering Dendrogram of Spearman Correlation by Feature of {project_name}')
    plt.ylabel('Distance (1 - Spearman)')
    plt.tight_layout()
    # plt.savefig(f'output_variable/{project_name}_dendrogram.png')
    # plt.show()

    return cluster_df


seatunnel = pd.read_pickle("output_variable/seatunnel_spearman_rank_high_case.pkl")
ozone = pd.read_pickle("output_variable/ozone_spearman_high_case.pkl")
pulsar = pd.read_pickle("output_variable/pulsar_spearman_high_case.pkl")

data_s = pd.read_pickle('../output/seatunnal_cut_time.pkl')
data_p = pd.read_pickle('../output/pulsar_cut_time.pkl')
data_o = pd.read_pickle('../output/ozone_cut_time.pkl')

pulsar_k = polt_spearman(pulsar, project_name='Pulsar')
ozone_k = polt_spearman(ozone, project_name="Ozone")
seatunnel_k = polt_spearman(seatunnel, project_name='Seatunnel')

p = pulsar_k['cluster'].value_counts()
o = ozone_k['cluster'].value_counts()
s = seatunnel_k['cluster'].value_counts()

ozone_k_list = list(ozone_k.groupby('cluster'))

ozone_k1 = list(ozone_k_list[0][1]['feature'].to_list())
ozone_k2 = list(ozone_k_list[1][1]['feature'].to_list())

o_ls = list([ozone_k1] + [ozone_k2])
with open('output_variable/ozone_correlation_main_group.pkl', 'wb') as f:
    joblib.dump(o_ls, f)

pulsar_k_list = list(pulsar_k.groupby('cluster'))
pulsar_k1 = list(pulsar_k_list[0][1]['feature'].to_list())
pulsar_k2 = list(pulsar_k_list[1][1]['feature'].to_list())
pulsar_k3 = list(pulsar_k_list[2][1]['feature'].to_list())
pulsar_k4 = list(pulsar_k_list[3][1]['feature'].to_list())

p_ls = list([pulsar_k1] + [pulsar_k2] + [pulsar_k3] + [pulsar_k4])

with open('output_variable/pulsar_correlation_main_group.pkl', 'wb') as f:
    joblib.dump(p_ls, f)

seatunnel_k_list = list(seatunnel_k.groupby('cluster'))

seatunnel_k1 = list(seatunnel_k_list[0][1]['feature'].to_list())
seatunnel_k2 = list(seatunnel_k_list[1][1]['feature'].to_list())
seatunnel_k3 = list(seatunnel_k_list[2][1]['feature'].to_list())
seatunnel_k4 = list(seatunnel_k_list[3][1]['feature'].to_list())
seatunnel_k5 = list(seatunnel_k_list[4][1]['feature'].to_list())
seatunnel_k6 = list(seatunnel_k_list[5][1]['feature'].to_list())
seatunnel_k7 = list(seatunnel_k_list[6][1]['feature'].to_list())

s_ls = list([seatunnel_k1] + [seatunnel_k2] + [seatunnel_k3] + [seatunnel_k4] + [seatunnel_k5] +
              [seatunnel_k6] + [seatunnel_k7])

with open('output_variable/seatunnel_correlation_main_group.pkl', 'wb') as f:
    joblib.dump(s_ls, f)

# o_combinations = list(product(data_o[ozone_k1], data_o[ozone_k2]))
# p_combinations = list(product(data_p[pulsar_k1], data_p[pulsar_k2], data_p[pulsar_k3], data_p[pulsar_k4]))
# s_combinations = list(product(data_s[seatunnel_k1], data_s[seatunnel_k2], data_s[seatunnel_k3],
#                               data_s[seatunnel_k4], data_s[seatunnel_k5], data_s[seatunnel_k6],
#                               data_s[seatunnel_k7]))
#
# with open('output_variable/pulsar_combinations_new.pkl', 'wb') as f:
#     joblib.dump(p_combinations, f)
#
# with open('output_variable/ozone_combinations_new.pkl', 'wb') as f:
#     joblib.dump(o_combinations, f)
#
# with open('output_variable/seatunnel_combinations_new.pkl', 'wb') as f:
#     joblib.dump(s_combinations, f)
#
# o_all_feature = [ozone_k1 + ozone_k2]
# p_all_feature = [pulsar_k1 + pulsar_k2 + pulsar_k3 + pulsar_k4]
# s_all_feature = [seatunnel_k1 + seatunnel_k2 + seatunnel_k3 + seatunnel_k4 + seatunnel_k5 + seatunnel_k6 +
#                  seatunnel_k7]
#
# with open('output_variable/pulsar_all_feature_each_smell.pkl', 'wb') as f:
#     joblib.dump(p_all_feature, f)
# with open('output_variable/ozone_all_feature_each_smell.pkl', 'wb') as f:
#     joblib.dump(o_all_feature, f)
# with open('output_variable/seatunnel_all_feature_each_smell.pkl', 'wb') as f:
#     joblib.dump(s_all_feature, f)
