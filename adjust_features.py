
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

sns.set_theme()

# In[] Variables

feature_file_name = "extracted_features.xlsx"
clinical_file_name = "labels_data.xlsx"

only_one_feature = 0  # 1 = only one feature to process and show graph; 0 = adjust all features
feature_name = 'TSK3-DUV'  # in the case of only_one_feature = 1

export_table = 1  # 1 = export adjusted data and coefficients to excel

# In[] read excel and create dataframes
df_clin = pd.read_excel(clinical_file_name, index_col=0)
df_feat = pd.read_excel(feature_file_name, index_col=0)

if only_one_feature == 1:
    feature_list = list([feature_name])
else:
    feature_list = list(df_feat.columns)

label_list = df_feat.index.values.tolist()

# pd.options.display.float_format = '${:,.2f}'.format # to have the float in dataframe
df_out = pd.DataFrame(0, index=label_list, columns=feature_list)
df_age = pd.DataFrame(0, index=feature_list, columns=['age coef'])
df_sex = pd.DataFrame(0, index=feature_list, columns=['sex coef'])

# In[] Loc (get vectors)

age = np.array(df_clin.loc[:, 'age']).reshape(-1, 1)

sex = np.array(df_clin.loc[:, 'sex'])
is_female = np.array((pd.get_dummies(sex)).loc[:, 'F'])

diag = np.array(df_clin.loc[:, 'diagnosis'])
is_PD = np.array((pd.get_dummies(diag)).loc[:, 'PD'])

# In[] Add missing values

imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

age = imputer_mean.fit_transform(age)
age = age.flatten()

for feature_name in feature_list:
    feature = np.array(df_feat.loc[:, feature_name])  # .reshape(-1,1)

    # In[] find NaNs

    NaN_vector = np.argwhere(np.isnan(feature) == True)[:, 0]

    # In[] Add mean to missing values

    feature = feature.reshape(-1, 1)
    feature = imputer_mean.fit_transform(feature)
    feature = feature.flatten()

    # In[] normalize the feature (to be able to compare coeffs of all features)

    max_feature = max(feature)
    feature = feature / max_feature

    # In[] devide to HC and PD

    feature_HC = feature[is_PD == 0]
    feature_PD = feature[is_PD == 1]

    age_HC = age[is_PD == 0]
    age_PD = age[is_PD == 1]

    sex_HC = sex[is_PD == 0]
    sex_PD = sex[is_PD == 1]

    is_female_HC = is_female[is_PD == 0]
    is_female_PD = is_female[is_PD == 1]

    # In[] Lienar regression using sklearn

    # Lin Reg Model from HC

    indep_var = np.array([age_HC, is_female_HC]).T  # get matrix of input independent variables (age and sex)

    LR_model = LinearRegression().fit(indep_var, feature_HC)  # train the model
    LR_inter = LR_model.intercept_  # get the interception
    LR_coef = LR_model.coef_  # get coefficients of LR

    y_mean = np.mean(feature_HC)  # get the mean of feature extracted from HC

    # adjust HC (just to plot to see what is happening)
    y_reg = LR_inter + LR_coef[0] * age_HC + LR_coef[1] * is_female_HC  # create a regression "line"
    residue = feature_HC - y_reg  # get the residue (distance between the real value nad regression line)
    y_out_HC = y_mean + residue  # adjust the feature

    # adjust all (HC + PD) from model trained by HC (our output vvalues of features)
    y_reg_all = LR_inter + LR_coef[0] * age + LR_coef[1] * is_female
    residue = feature - y_reg_all
    y_out = y_mean + residue

    # create new model HC (plot to see if the feature is no more dependent on age and sex)
    LR_model2 = LinearRegression().fit(indep_var, y_out_HC)
    LR_inter2 = LR_model2.intercept_
    LR_coef2 = LR_model2.coef_

    y_reg_new = LR_inter2 + LR_coef2[0] * age_HC + LR_coef2[1] * is_female_HC

    # In[] return NaNs and save vector

    y_out[NaN_vector] = np.NaN
    df_out[feature_name] = (y_out * max_feature).tolist()

    # In[] save regression coefficient

    df_age.loc[feature_name, 'age coef'] = LR_coef[0]
    df_sex.loc[feature_name, 'sex coef'] = LR_coef[1]

    # In[] sort vectors with coefficients

    df_age = df_age.sort_values(by='age coef', ascending=False,
                                key=pd.Series.abs)
    df_sex = df_sex.sort_values(by='sex coef', ascending=False,
                                key=pd.Series.abs)

# In[] export datasets to excel

if only_one_feature == 0 and export_table == 1:
    with pd.ExcelWriter('adjusted_features.xlsx') as writer:
        df_out.to_excel(writer, sheet_name='adjusted features')
        df_age.to_excel(writer, sheet_name='age coef')
        df_sex.to_excel(writer, sheet_name='sex coef')

# In[] Plot the linear regression

if only_one_feature == 1:

    if LR_coef[0] > 0:
        a_r = 'increasing'
    elif LR_coef[0] < 0:
        a_r = 'decreasing'
    else:
        a_r = 'not changing'

    if LR_coef[1] > 0:
        g_r = 'reach higher values than'
    elif LR_coef[1] < 0:
        g_r = 'reach lower values than'
    else:
        g_r = 'reach same values as'

    result = ' '.join(['HC: The feature is', a_r,
                       'with age and females', g_r,
                       'males.'])
    plt.figure(1).clf()
    plt.figure(1)
    plt.scatter(age_HC, feature_HC, color='lightsteelblue')
    plt.scatter(age_HC, y_out_HC, color='steelblue')
    plt.plot(age_HC, y_reg, 'k.',
             np.array([np.min(age_HC), np.max(age_HC)]),
             np.array([y_mean, y_mean]), 'k',
             age_HC, y_reg_new, 'r_')
    plt.legend(['original',
                'adjusted',
                'regression line',
                'mean',
                'new reg. line'])

    plt.title(result)
    plt.xlabel('age')
    plt.ylabel(feature_name)
    plt.savefig('results/' + feature_name + '_effect.pdf')
    plt.close()

# In[]

print('finished')
