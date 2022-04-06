
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

import seaborn as sns

# In[] variables + set the script

filename = 'dataset_corr.xlsx'

export_table = 1

scenario_list = ['duration_of_PD', 'LED', 'UPDRSIII', 'UPDRSIII-speech', 'H&Y']
score_list = ['coeff', 'p-value', 'FDR_correction']

# prepare output excel
if export_table == 1:
    writer_spear = pd.ExcelWriter('results/spearman.xlsx')
    writer_pears = pd.ExcelWriter('results/pearson.xlsx')

for scenario in scenario_list:

    # In[] load data

    df_data = pd.read_excel(filename, sheet_name=scenario, index_col=0)

    df_PD = df_data.loc[df_data['diagnosis'] == 'PD']

    df_feat = df_PD.drop(['diagnosis', scenario], axis=1)

    # get features names
    feature_list = list(df_feat)

    # prepare tables
    df_spear = pd.DataFrame(0.00, index=feature_list, columns=score_list)
    df_pears = pd.DataFrame(0.00, index=feature_list, columns=score_list)

    # In[] calculate Spearman and Pearson

    for feature_name in feature_list:
        # select vectors of features and clinical data
        feat_vector = np.array(df_feat.loc[:, feature_name])
        data_vector = np.array(df_PD.loc[:, scenario])

        # remove NaNs
        nan_mask_feat = np.isnan(feat_vector)
        feat_vector = feat_vector[~nan_mask_feat]
        data_vector = data_vector[~nan_mask_feat]

        nan_mask_data = np.isnan(data_vector)
        feat_vector = feat_vector[~nan_mask_data]
        data_vector = data_vector[~nan_mask_data]

        # Spearman's rank correlation
        coef_spear, p_spear = spearmanr(feat_vector.flatten(), data_vector.flatten())
        df_spear.loc[feature_name, score_list[0]] = round(coef_spear, 3)
        df_spear.loc[feature_name, score_list[1]] = round(p_spear, 3)

        reject_spear, p_cor_spear = fdrcorrection(np.array(list(df_spear['p-value'])), alpha=0.05, method='indep')
        df_spear.loc[:, 'FDR_correction'] = p_cor_spear

        # Pearson's rank correlation
        coef_pears, p_pears = pearsonr(feat_vector.flatten(), data_vector.flatten())
        df_pears.loc[feature_name, score_list[0]] = round(coef_pears, 3)
        df_pears.loc[feature_name, score_list[1]] = round(p_pears, 3)

        reject_pears, p_cor_pears = fdrcorrection(np.array(list(df_spear['p-value'])), alpha=0.05, method='indep')
        df_pears.loc[:, 'FDR_correction'] = p_cor_pears

    if export_table == 1:
        df_spear.to_excel(writer_spear, sheet_name=scenario)
        df_pears.to_excel(writer_pears, sheet_name=scenario)

# save and close excel
if export_table == 1:
    writer_spear.save()
    writer_pears.save()
