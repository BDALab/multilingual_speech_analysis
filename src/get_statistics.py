import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

"""
Script for computing:
 - MannwhitneyU test for all features
 - ratio of PD patients that lie outside of normal interval of healthy controls
"""

# TODO Daniel
# load data from `data/features_adjusted.csv` and `data/labels.csv` instead
# of `dataset_stats.xlsx`. Change output file names and types to csv.
# Use `df = pd.read_csv(path, sep=';')` and `df.to_csv(path, sep=';')`.
# Btw, with csv, you do not need any writer

# In[] Set the script

only_one_Feature = 0  # 1 = only one feature to process and show graph, 0 = process all features

# in the case of all features
only_one_scenario = 0  # 1 = one scenario, 0 = iterate over all scenarios and save dfs as different excel sheets
export_table = 1  # 1 = export results to excel file, 0 = do nothing

# In[] Variables (in the case of 1 feature and 1 scenario)

file_name = 'data/dataset_stats.xlsx'  # name of the excel file
scenario_list = ['CZ', 'US', 'IL', 'CO', 'IT', 'all']  # as names of sheets in excel file

scenario = 'IT'  # in the case of only_one_scenario = 1
feature_name = 'TSK2-EEVOL'  # in the case of 1 feature

# In[] Loop for scenarios

if only_one_scenario == 1:
    scenario_list = list([scenario])

if export_table == 1:
    writer_RAT = pd.ExcelWriter('results/stats_results.xlsx')

for scenario in scenario_list:

    df_feat = pd.read_excel(file_name, sheet_name=scenario, index_col=0)  # load table

    if only_one_Feature == 1:
        feature_list = list([feature_name])
    else:
        feature_list = list(df_feat.columns)
        label_list = df_feat.index.values.tolist()

    # In[] divide data to HC and PD

    df_HC = df_feat.loc[df_feat['diagnosis'] == 'HC']  # dataframe of Healthy controls
    df_PD = df_feat.loc[df_feat['diagnosis'] == 'PD']  # dataframe of Parkinson's disease

    # In[] prepare statistics and RAT-score dataframe

    score_list = ['Mann-Whitney', 'FDR_correction', 'significance',
                  'mean_HC', 'mean_PD', 'mean',
                  'med_HC', 'med_PD', 'med',
                  'std_HC', 'std_PD', 'std',
                  'RAT_L', 'RAT_H']

    if only_one_Feature == 0:
        feature_list.pop()  # drop the diagnosis column out

    df_out = pd.DataFrame(0.00, index=feature_list, columns=score_list)

    # In[] Select the feature and preprocess

    c = int(0)  # for monitoring the loop progression

    for feature_name in feature_list:

        # select vectors of HC and PD for the given feature
        feature_HC = np.array(df_HC.loc[:, feature_name])
        feature_PD = np.array(df_PD.loc[:, feature_name])

        # In[] remove NaN

        # HC
        nan_mask_HC = np.isnan(feature_HC)
        feature_noNaN_HC = feature_HC[~nan_mask_HC]

        # PD
        nan_mask_PD = np.isnan(feature_PD)
        feature_noNaN_PD = feature_PD[~nan_mask_PD]

        # In[] Statistics

        out_mean_HC = np.mean(feature_noNaN_HC)
        out_median_HC = np.median(feature_noNaN_HC)
        out_std_HC = np.std(feature_noNaN_HC)

        out_mean_PD = np.mean(feature_noNaN_PD)
        out_median_PD = np.median(feature_noNaN_PD)
        out_std_PD = np.std(feature_noNaN_PD)

        MW_U, MW_p_value = mannwhitneyu(feature_noNaN_HC, feature_noNaN_PD, method="auto")  # Mann-Whitney U test

        # In[] (norm data to HC) z-scores and calculate geometric features

        # ----- HC -----
        z_score_HC = (feature_noNaN_HC - np.mean(feature_noNaN_HC)) / np.std(feature_noNaN_HC)

        z_mean_HC = np.mean(z_score_HC)
        z_median_HC = np.median(z_score_HC)
        z_std_HC = np.std(z_score_HC)

        # ----- PD -----

        z_score_PD = (feature_noNaN_PD - np.mean(feature_noNaN_HC)) / np.std(feature_noNaN_PD)

        z_mean_PD = np.mean(z_score_PD)
        z_median_PD = np.median(z_score_PD)
        z_std_PD = np.std(z_score_PD)

        # In[] Get boundaries for outliers from HC distribution
        percentile25 = np.quantile(z_score_HC, 0.25)
        percentile75 = np.quantile(z_score_HC, 0.75)

        iqr = percentile75 - percentile25

        lower_limit = percentile25 - 1.5 * iqr
        upper_limit = percentile75 + 1.5 * iqr

        # In[] Investigate the distribution

        PD_out_L = z_score_PD[z_score_PD < lower_limit]  # vector of non-normal labels
        PD_out_R = z_score_PD[z_score_PD > upper_limit]  # vector of non-normal labels

        n_PD_out = PD_out_L.size + PD_out_R.size  # number of non-normal labels

        if n_PD_out > 0:
            # Left
            PD_out_L = z_score_PD[z_score_PD < lower_limit]
            n_PD_L = PD_out_L.size
            if PD_out_L.size == 0:
                ratio_PD_L = 0
            else:
                ratio_PD_L = (n_PD_L / z_score_PD.size) * 100

            # Right
            PD_out_R = z_score_PD[z_score_PD > upper_limit]
            n_PD_R = PD_out_R.size
            if PD_out_R.size == 0:
                ratio_PD_R = 0
            else:
                ratio_PD_R = (n_PD_R / z_score_PD.size) * 100
        else:
            n_PD_L = 0
            n_PD_R = 0
            ratio_PD_L = 0
            ratio_PD_R = 0

        # In[] investigate geometrics

        if out_mean_PD > out_mean_HC:
            change_mean = u'\u2191'  # (higher)
        else:
            change_mean = u'\u2193'  # (lower)

        if out_median_PD > out_median_HC:
            change_median = u'\u2191'
        else:
            change_median = u'\u2193'

        if out_std_PD > out_std_HC:
            change_std = u'\u2191'
        else:
            change_std = u'\u2193'

        # In[] save the result

        # geometrical
        df_out.loc[feature_name, 'mean_HC'] = out_mean_HC
        df_out.loc[feature_name, 'mean_PD'] = out_mean_PD
        df_out.loc[feature_name, 'mean'] = change_mean
        df_out.loc[feature_name, 'med_HC'] = out_median_HC
        df_out.loc[feature_name, 'med_PD'] = out_median_PD
        df_out.loc[feature_name, 'med'] = change_median
        df_out.loc[feature_name, 'std_HC'] = out_std_HC
        df_out.loc[feature_name, 'std_PD'] = out_std_PD
        df_out.loc[feature_name, 'std'] = change_std

        # statistical
        df_out.loc[feature_name, 'Mann-Whitney'] = MW_p_value
        df_out.loc[feature_name, 'RAT_L'] = ratio_PD_L
        df_out.loc[feature_name, 'RAT_H'] = ratio_PD_R

        # In[] show the loop progression
        c = c + 1
        print('Scenario: ' + scenario + ' - Features done: ' + str(c) + '/' + str(len(feature_list)))

    # In[] FDR correction for MW test

    rejected, p_values_cor = fdrcorrection(np.array(list(df_out['Mann-Whitney'])), alpha=0.05, method='indep')

    df_out.loc[:, 'significance'] = rejected*1
    df_out.loc[:, 'FDR_correction'] = p_values_cor

    # In[] export dataset

    if export_table == 1:
        df_out.to_excel(writer_RAT, sheet_name=scenario)

    # In[] plot the distribution (in the case of only one feature)

    sns.set_theme()

    if only_one_Feature == 1:
        plt.figure(1)
        plt.clf()
        # plt.subplot(211)

        sns.histplot(x=z_score_HC, stat='count', binwidth=0.2, color='b', zorder=2)
        sns.histplot(x=z_score_PD, stat='count', binwidth=0.2, color='peachpuff', zorder=1)

        limits = np.array(plt.ylim())

        plt.plot(np.array([z_mean_HC, z_mean_HC]),
                 np.array([0, limits[1] * 0.95 / 2]), 'b--',
                 np.array([z_mean_PD, z_mean_PD]),
                 np.array([0, limits[1] * 0.95 / 2]), 'r--',
                 np.array([z_median_HC, z_median_HC]),
                 np.array([0, limits[1] * 0.95]), 'b',
                 np.array([z_median_PD, z_median_PD]),
                 np.array([0, limits[1] * 0.95]), 'r')

        plt.scatter(PD_out_L.reshape(-1, 1), np.full((1, n_PD_L), 1),
                    color='r',
                    zorder=9)
        plt.scatter(PD_out_R.reshape(-1, 1), np.full((1, n_PD_R), 1),
                    color='r',
                    zorder=9)

        plt.plot(np.array([lower_limit, lower_limit]), np.array([0, limits[1] * 0.95]), 'k')

        plt.xlabel('z-score: %s (%s)' % (feature_name, scenario))
        plt.legend([('HC mean= %.3f' % z_mean_HC),
                    ('PD mean = %.3f' % z_mean_PD),
                    ('HC median= %.3f' % z_median_HC),
                    ('PD median = %.3f' % z_median_PD),

                    ('PD < Q1 - 1.5*IQR = %i (%.1f %%)' % (n_PD_L, ratio_PD_L)),
                    ('PD > Q3 + 1.5*IQR = %i (%.1f %%)' % (n_PD_R, ratio_PD_R)),'boundaries for outliers',
                    'HC', 'PD'])

        plt.plot(np.array([upper_limit, upper_limit]), np.array([0, limits[1] * 0.95]), 'k')

        plt.xlabel('z-score: %s (%s)' % (feature_name, scenario))
        plt.savefig('results/out_of_norm.pdf')
        plt.close()

# In[] save and close excel files
if export_table == 1:
    writer_RAT.save()

# In[]

print('Script finished.')
