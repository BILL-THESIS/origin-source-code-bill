import pandas as pd
import seaborn as sns
import plotly.express as px
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import time
from ast import literal_eval

from requests_toolbelt.multipart.encoder import total_len


def load_data():
    files = {
        "significant": "../../output/seatunnel_all_status_significant.pkl",
        "group_smell": "../../output/seatunnel_rdf_quantile_all.pkl",
        "each_smell": "../../output/seatunnel_rdf_quantile_each_smell.pkl",
        "main_group": "../../output/seatunnel_correlation_main_group_4.pkl",
        "original": "../../output/seatunnel_compare.pkl"
    }
    return {key: pd.read_pickle(path) for key, path in files.items()}



if __name__ == "__main__":
    file_significant = "../../output/seatunnel_all_status_significant.pkl"
    file_group_smell = "../../output/seatunnel_rdf_quantile_all.pkl"
    file_each_smell = "../../output/seatunnel_rdf_quantile_each_smell.pkl"
    file_main_group = "../../output/seatunnel_correlation_main_group_4.pkl"

    data_qr1 = pd.read_pickle(file_significant)
    data_group_smell = pd.read_pickle(file_group_smell)
    data_each_smell = pd.read_pickle(file_each_smell)
    data_mian_group = pd.read_pickle(file_main_group)
    data_original = pd.read_pickle("../../output/seatunnel_compare.pkl")

    data_each_smell['rank'] = data_each_smell['test_f1'].rank(ascending=False)
    groups = [data_each_smell[data_each_smell["features"].isin(data_mian_group[i])] for i in range(4)]
    groups = [group.merge(data_qr1, left_on='features', right_on='metric', how='left') for group in groups]
    group1, group2, group3, group4 = groups


    for group in [group1, group2, group3, group4]:
        group['rank_f1'] = group['test_f1'].rank(ascending=False)
        group['rank_roc'] = group['test_roc_auc'].rank(ascending=False)
        group['rank_d'] = group['d_value'].rank(ascending=False)


    group1 = group1[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group2 = group2[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group3 = group3[['features', 'rank_f1', 'rank_roc', 'rank_d']]
    group4 = group4[['features', 'rank_f1', 'rank_roc', 'rank_d']]

    group_join = pd.concat([group1, group2, group3, group4], axis=0)

    grop_80_percen = data_group_smell.loc[data_group_smell['test_f1'] >= 0.8]
    grop_less_80_percen = data_group_smell.loc[data_group_smell['test_f1'] < 0.8]

    data_rank_all_group = data_group_smell['features'].apply(lambda s: [group_join.set_index('features').loc[x, 'rank_f1'] for x in s])

    data_rank_all_group = pd.concat([data_group_smell['features'], data_group_smell['test_f1'], data_rank_all_group], axis=1)
    df_split_less_80_percen = pd.DataFrame(data_rank_all_group.iloc[:, 2].tolist(), columns=["G1", "G2", "G3", "G4"], index=data_rank_all_group.index)

    data_rank_all_group = pd.concat([data_rank_all_group, df_split_less_80_percen], axis=1)
    data_rank_all_group['sum'] = data_rank_all_group[['G1', 'G2', 'G3', 'G4']].sum(axis=1)

    filtered_less = data_rank_all_group[data_rank_all_group['sum'] < []]
    filtered_more_80_percen_f1 = filtered_less.loc[filtered_less['test_f1'] >= 0.8]



    # stage 2 by ROC

    data_rank_group_auc = grop_80_percen['features'].apply(lambda s: [group_join.set_index('features').loc[x, 'rank_roc'] for x in s])
    data_rank_group_auc = pd.concat([grop_80_percen['features'],grop_80_percen['test_roc_auc'] ,data_rank_group_auc], axis=1)
    df_split_auc = pd.DataFrame(data_rank_group_auc.iloc[:, 2].tolist(), columns=["G1", "G2", "G3", "G4"],
                                index=data_rank_group_auc.index)
    data_rank_group_auc = pd.concat([data_rank_group_auc, df_split_auc], axis=1)
    data_rank_group_auc['sum'] = data_rank_group_auc[['G1', 'G2', 'G3', 'G4']].sum(axis=1)

    filtered_auc = data_rank_group_auc[data_rank_group_auc['sum'] < 16]
    filtered_more_80_percen_auc = filtered_auc.loc[filtered_auc['test_roc_auc'] >= 0.8]


    # stage 3 by D-values

    data_rank_group_d = grop_80_percen['features'].apply(lambda s: [group_join.set_index('features').loc[x, 'rank_d'] for x in s])
    data_rank_group_d = pd.concat([grop_80_percen['features'] ,data_rank_group_d], axis=1)
    df_split_d = pd.DataFrame(data_rank_group_d.iloc[:, 1].tolist(), columns=["G1", "G2", "G3", "G4"],
                              index=data_rank_group_d.index)
    data_rank_group_d = pd.concat([data_rank_group_d, df_split_d], axis=1)
    data_rank_group_d['sum'] = data_rank_group_d[['G1', 'G2', 'G3', 'G4']].sum(axis=1)

    filtered_d = data_rank_group_d[data_rank_group_d['sum'] <= 20]
    filtered_more_80_percen_d = filtered_d.loc[filtered_d['test_f1'] >= 0.8]