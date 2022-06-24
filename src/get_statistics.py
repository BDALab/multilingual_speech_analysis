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

# In[] Set the script

only_one_Feature = False  # True = only one feature to process and show graph, False = process all features

# in the case of all features
only_one_scenario = False  # True = one scenario, False = all scenarios and save results as different excel sheets

export_table = True  # True = export results to excel file, False = do nothing

# In[] Variables

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'
output_filename_stats = 'results/stats_results.xlsx'
output_filename_pdf = 'results/out_of_norm.pdf'  # in the case of only_one_Feature = True

scenario_list = ['CZ', 'US', 'IL', 'CO', 'IT']  # languages
score_list = ['Mann-Whitney', 'FDR_correction', 'significance',
              'mean_HC', 'mean_PD', 'mean',
              'med_HC', 'med_PD', 'med',
              'std_HC', 'std_PD', 'std',
              'RAT_L', 'RAT_H']

scenario = 'IT'  # in the case of only_one_scenario = True
feature_name = 'TSK2-RFA2'  # in the case of only_one_Feature = True

# In[] Load data

df_feat = pd.read_csv(features_file_name, sep=';', index_col=0)
df_clin = pd.read_csv(clinical_file_name, sep=';', index_col=0, usecols=['ID', 'nationality', 'diagnosis'])

df_data = df_feat.copy().join(df_clin['diagnosis'])

if export_table:
    writer_RAT = pd.ExcelWriter(output_filename_stats)

# In[] Loop through scenarios

if only_one_scenario:
    scenario_list = list([scenario])

for scenario in scenario_list:

    df_scenario = df_data.loc[df_clin['nationality'] == scenario]

    if scenario == 'IT':
        df_scenario = df_scenario.drop('TSK7-relSDSD', axis=1)

    if only_one_Feature:
        feature_list = list([feature_name])
    else:
        feature_list = list(df_scenario.columns)
        feature_list.pop()

        # In[] Prepare table

    df_out = pd.DataFrame(0.00, index=feature_list, columns=score_list)

    # In[] Divide data to HC and PD

    df_HC = df_scenario.loc[df_scenario['diagnosis'] == 'HC']  # dataframe of Healthy controls
    df_PD = df_scenario.loc[df_scenario['diagnosis'] == 'PD']  # dataframe of Parkinson's disease

    # In[] Select the feature and preprocess

    c = int(0)  # for monitoring the loop progression

    for feature_name in feature_list:

        # select vectors of HC and PD for the given feature
        feature_HC = np.array(df_HC.loc[:, feature_name])
        feature_PD = np.array(df_PD.loc[:, feature_name])

        # In[] Remove NaN

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

        # In[] Normalise data to HC (z-scores) and calculate geometric features

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
        c += 1
        print('Scenario: ' + scenario + ' - Features done: ' + str(c) + '/' + str(len(feature_list)))

    # In[] FDR correction for MW test

    rejected, p_values_cor = fdrcorrection(np.array(list(df_out['Mann-Whitney'])), alpha=0.05, method='indep')

    df_out.loc[:, 'significance'] = rejected*1
    df_out.loc[:, 'FDR_correction'] = p_values_cor

    # In[] Export tables

    if export_table:
        df_out.to_excel(writer_RAT, sheet_name=scenario)

    # In[] Plot the distribution (in the case of only one feature)

    sns.set_theme()

    if only_one_Feature:
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
                    ('PD > Q3 + 1.5*IQR = %i (%.1f %%)' % (n_PD_R, ratio_PD_R)), 'boundaries for outliers',
                    'HC', 'PD'])

        plt.plot(np.array([upper_limit, upper_limit]), np.array([0, limits[1] * 0.95]), 'k')

        plt.xlabel('z-score: %s (%s)' % (feature_name, scenario))
        plt.savefig(output_filename_pdf)
        plt.close()

# In[] Save and close excel files

if export_table:
    writer_RAT.save()

# In[]

print('Script finished.')
