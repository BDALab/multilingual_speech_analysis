import pandas as pd
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

"""
Script for computing machine learning model performance with leave-one-language-out validation technique:
"""

# In[] variables

features_list = ['TSK3-HRF', 'TSK3-NAQ', 'TSK3-QOQ', 'TSK3-relF0SD', 'TSK3-Jitter (PPQ)', 'TSK2-RFA1', 'TSK2-RFA2',
                 'TSK2-#loc_max', 'TSK2-relF2SD', 'TSK2-#lndmrk', 'TSK7-relSDSD', 'TSK7-COV', 'TSK7-RI', 'TSK2-relF0SD',
                 'TSK2-SPIR']  # robust features according to statistical analysis

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'
output_filename = 'results/leave_one_language_out.xlsx'  # cross-language validation

scenario_list = ['CZ', 'US', 'IL', 'CO', 'IT']
metric_list = ['mcc', 'acc', 'sen', 'spe']

seed = 42  # random search

# In[] set the script

all_features = True  # False = use features in features_list
export_table = True  # export four tables in total

# In[] Load data

if all_features:
    df_feat = pd.read_csv(features_file_name, sep=';', index_col=0)
else:
    df_feat = pd.read_csv(features_file_name, sep=';', index_col=0, usecols=['ID'] + features_list)

df_clin = pd.read_csv(clinical_file_name, sep=';', index_col=0, usecols=['ID', 'nationality', 'diagnosis'])

df_data = df_feat.copy().join(df_clin['diagnosis'])


# In[] Replace HC for 0 and PD for 1

def replace_diagnosis_with_numbers(df):
    df_out = df.replace(['HC', 'PD'], [0, 1])
    return df_out


# In[] Define the classification metrics

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


# In[] Prepare the table

if export_table:
    writer_cross = pd.ExcelWriter(output_filename)

# create empty dataframe
df_cross_language = pd.DataFrame(0.0, index=metric_list, columns=scenario_list)

# In[] Define the classifier settings

model_params = {
    "booster": "gbtree",
    "n_jobs": -1,
    "use_label_encoder": False,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": seed,
}

param_grid = {
    "n_estimators": [100, 500, 1000],
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "gamma": [0, 0.10, 0.15, 0.25, 0.5],
    "max_depth": [4, 6, 8, 10, 12, 15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'max_delta_step': [0, 1, 5, 10],
    "scale_pos_weight": [1, 5, 10, 20, 50, 100]
}

search_settings = {
    "param_distributions": param_grid,
    "scoring": 'balanced_accuracy',
    "n_jobs": -1,
    "n_iter": 500,
    "verbose": 10
}

# In[] Loop through scenarios

for scenario_test in scenario_list:

    # In[] create training and testing dataframes

    df_scenario_test = df_data.loc[df_clin['nationality'] == scenario_test]
    df_scenario_train = df_scenario_test.copy()

    scenario_list_train = scenario_list.copy()

    # Drop one scenario that will be the testing scenario
    scenario_list_train.remove(scenario_test)

    for scenario_train in scenario_list_train:

        # Load the feature matrix and merge
        df_scenario = df_data.loc[df_clin['nationality'] == scenario_train]
        df_scenario_train = pd.concat([df_scenario_train, df_scenario])

    df_scenario_train.drop(index=df_scenario_train.index[:df_scenario_test.shape[0]], axis=0, inplace=True)

    # In[] Divide into HC and PD

    df_scenario_train = replace_diagnosis_with_numbers(df_scenario_train)

    X_train = df_scenario_train.iloc[:, 0:-1].values
    y_train = df_scenario_train.iloc[:, -1].values

    # In[] Search for the best hyper-parameters

    print('Hyper-parameters tuning - scenario out:' + scenario_test)

    # Create the classifier
    model = xgb.XGBClassifier(**model_params)

    # Get the cross-validation indices
    kfolds = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # Employ the hyper-parameter tuning
    random_search = RandomizedSearchCV(
        model, cv=kfolds.split(X_train, y_train), random_state=seed,
        **search_settings)
    random_search.fit(X_train, y_train)

    best_tuning_score = random_search.best_score_
    best_tuning_hyperparams = random_search.best_estimator_

    # In[] Testing on the one scenario left

    model = xgb.XGBClassifier(**random_search.best_params_)
    model.fit(X_train, y_train)

    df_scenario_test = replace_diagnosis_with_numbers(df_scenario_test)

    X_test = df_scenario_test.iloc[:, 0:-1].values
    y_test = df_scenario_test.iloc[:, -1].values

    y_pred = model.predict(X_test)

    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    sen = recall_score(y_test, y_pred)
    spe = specificity_score(y_test, y_pred)

    df_cross_language.loc['mcc', scenario_test] = round(mcc, 2)
    df_cross_language.loc['acc', scenario_test] = round(acc, 2)
    df_cross_language.loc['sen', scenario_test] = round(sen, 2)
    df_cross_language.loc['spe', scenario_test] = round(spe, 2)

# In[] export tables

if export_table:
    df_cross_language.to_excel(writer_cross, sheet_name='leave-one-language-out')

# In[] save and close excel files

if export_table:
    writer_cross.save()

# In[]

print('Script finished.')
