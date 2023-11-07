import os
import numpy as np
import pandas as pd

import pingouin as pg
from statsmodels.stats.multitest import fdrcorrection

"""
Script for computing the partial Pearson and Spearman correlation of features and
clinical scores.
"""

# In[] Set the script

export_table = True
correlation = 'spearman'  # or 'pearson'

# In[] Variables

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'
output_filename = f'results/partial_{correlation}.xlsx'

scenarios = ['duration_of_PD', 'LED', 'UPDRSIII', 'UPDRSIII-speech', 'H&Y']
scores = ['coeff', 'p-value', 'FDR_correction']

# In[] Load data
df_feat = pd.read_excel(features_file_name, index_col=0)

# get feature names
features = df_feat.columns

# prepare empty df
df = pd.DataFrame(index=features, columns=pd.MultiIndex.from_product(
    [scenarios, scores], names=['Scenarios', 'Scores']))

# In[] Loop through scenarios

for scenario in scenarios:

    # In[] load data

    df_clin = pd.read_excel(clinical_file_name, index_col=0, usecols=['ID', scenario, 'nationality'])
    df_data = df_feat.copy().join(df_clin)

    # In[] Calculate correlation of each feature with scenario

    for feature_name in features:

        df_data['nationality_factorized'] = pd.factorize(df_data['nationality'])[0]
        partial_corr = pg.partial_corr(data=df_data, x=feature_name, y=scenario,
                                                  covar='nationality_factorized', method=correlation)

        pval = partial_corr['p-val'].iloc[0]
        coef = partial_corr['r'].iloc[0]

        df.loc[feature_name, scenario] = [coef, pval, np.nan]  # round(x, 3)

    # Compute false discovery rate
    all_pvals = df.loc[:, (scenario, 'p-value')].values
    _, fdrcorrs = fdrcorrection(all_pvals, alpha=0.05, method='indep')
    df.loc[:, (scenario, 'FDR_correction')] = fdrcorrs

# In[] Export tables

if export_table:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_excel(output_filename)

# In[]

print('Script finished.')
