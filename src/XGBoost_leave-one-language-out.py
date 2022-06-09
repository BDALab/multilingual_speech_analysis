import pandas as pd
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# TODO Daniel
# load data from `data/features_adjusted.csv` and `data/labels.csv` instead
# of `dataset_XGB.xlsx`. Change output file names and types to csv.
# Use `df = pd.read_csv(path, sep=';')` and `df.to_csv(path, sep=';')`.
# Btw, with csv, you do not need any writer

# In[] variables

# name of the excel file with features
file_name = 'dataset_XGB.xlsx'

# name of the folder where results (tables and graphs) will be stored
folder_save = 'results'

# name of specific sheets in excel file (file_name)
scenario_list = list(['CZ', 'US', 'IL', 'CO', 'IT'])

export_table = True  # export four tables in total

seed = 42  # random search


# In[] Define the classification metrics
def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)

# In[] Set the script


if export_table:
    writer_cross = pd.ExcelWriter(folder_save + '/leave-one-language-out.xlsx')

metric_list = ['MCC', 'ACC', 'SEN', 'SPE']

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

    df_feat_test = pd.read_excel(
        file_name, sheet_name=scenario_test, index_col=0)
    feature_list = list(df_feat_test.columns)
    df_feat_train = df_feat_test.copy()

    # In[] Load datasets and merge

    scenario_list_train = scenario_list.copy()

    # Drop one scenario that will be the testing scenario
    scenario_list_train.remove(scenario_test)

    for scenario_train in scenario_list_train:
        # Load the feature matrix
        df_feat = pd.read_excel(
            file_name, sheet_name=scenario_train, index_col=0)
        df_feat_train = pd.concat([df_feat_train, df_feat])

    df_feat_train.drop(index=df_feat_train.index[:df_feat_test.shape[0]],
                       axis=0, inplace=True)

    # In[] Divide into HC and PD

    df_feat_train = df_feat_train.replace(['PD'], 1)
    df_feat_train = df_feat_train.replace(['HC'], 0)

    X_train = df_feat_train.iloc[:, 0:-1].values
    y_train = df_feat_train.iloc[:, -1].values

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

    df_feat_test = df_feat_test.replace(['PD'], 1)
    df_feat_test = df_feat_test.replace(['HC'], 0)

    X_test = df_feat_test.iloc[:, 0:-1].values
    y_test = df_feat_test.iloc[:, -1].values

    y_pred = model.predict(X_test)

    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    sen = sensitivity_score(y_test, y_pred)
    spe = specificity_score(y_test, y_pred)

    df_cross_language.loc['MCC', scenario_test] = round(mcc, 2)
    df_cross_language.loc['ACC', scenario_test] = round(acc, 2)
    df_cross_language.loc['SEN', scenario_test] = round(sen, 2)
    df_cross_language.loc['SPE', scenario_test] = round(spe, 2)

# In[] export tables

if export_table:

    df_cross_language.to_excel(
        writer_cross, sheet_name='leave-one-language-out')

# In[] save and close excel files
if export_table:
    writer_cross.save()

# In[]

print('Script finished.')
