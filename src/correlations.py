import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

"""
Script for computing the Pearson and Spearman correlation of features and
clinical scores.
"""

# In[] variables + set the script

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'
export_table = True
correlation = 'spearman'  # or 'pearson'
output_filename = f'results/{correlation}.xlsx'
scenarios = ['duration_of_PD', 'LED', 'UPDRSIII', 'UPDRSIII-speech', 'H&Y']
scores = ['coeff', 'p-value', 'FDR_correction']

# load features
df_feat = pd.read_csv(features_file_name, sep=';', index_col=0)

# get feature names
features = df_feat.columns

# prepare empty df
df = pd.DataFrame(index=features, columns=pd.MultiIndex.from_product(
    [scenarios, scores], names=['Scenarios', 'Scores']))

for scenario in scenarios:

    # In[] load data

    df_clin = pd.read_csv(
        clinical_file_name, sep=';', index_col=0, usecols=['ID', scenario])
    df_data = df_feat.copy().join(df_clin)

    # In[] calculate correlation of each feature with scenario

    for feature_name in features:
        # select vectors of features and clinical data
        feat_vector = df_data.loc[:, feature_name].values
        clin_vector = df_data.loc[:, scenario].values

        # remove rows where the feature is NaN
        nan_mask_feat = np.isnan(feat_vector)
        feat_vector = feat_vector[~nan_mask_feat]
        clin_vector = clin_vector[~nan_mask_feat]

        # remove rows where the scenario is NaN
        nan_mask_clin = np.isnan(clin_vector)
        feat_vector = feat_vector[~nan_mask_clin]
        clin_vector = clin_vector[~nan_mask_clin]

        # compute correlation coefficient, p-value and fdr correction
        func = spearmanr if correlation == 'spearman' else pearsonr
        coef, pval = func(feat_vector.flatten(), clin_vector.flatten())
        df.loc[feature_name, scenario] = [coef, pval, np.nan]  # round(x, 3)

    # once the p-values for all featues are computed, compute fdrcorrections
    all_pvals = df.loc[:, (scenario, 'p-value')].values
    _, fdrcorrs = fdrcorrection(all_pvals, alpha=0.05, method='indep')
    df.loc[:, (scenario, 'FDR_correction')] = fdrcorrs

if export_table:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_excel(output_filename)

print('Script finished.')
