
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import xgboost as xgb

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, precision_score, recall_score, \
    f1_score, roc_curve, auc

import shap

# sns.set_theme()

# In[] variables

file_name = 'dataset.xlsx'
scenario_list = list(['CZ', 'US', 'IL', 'CO', 'IT', 'all'])  # name of the excel sheet


only_one_scenario = 0  # 1 =  process just one scenario
scenario = 'IL'  # in the case of only_one_scenario = 1

export_table = 1

seed = 42

# In[] Set the script

if export_table == 1:
    writer_imp = pd.ExcelWriter('results/feature_importances.xlsx')
    writer_per = pd.ExcelWriter('results/model_performance.xlsx')

    if not only_one_scenario == 1:
        writer_cross = pd.ExcelWriter('results/cross_language.xlsx')
        writer_cross_mcc = pd.ExcelWriter('results/cross_language_mcc.xlsx')

if only_one_scenario == 1:
    scenario_list = list([scenario])

metric_list = ['mcc', 'F1', 'AUC', 'acc', 'sen', 'spe']  # list of metrics in final table
df_performance = pd.DataFrame(0.00, index=scenario_list, columns=metric_list)  # create empty dataframe
df_cross_language = pd.DataFrame(0.00, index=scenario_list, columns=metric_list)  # create empty dataframe
df_cross_language_MCC = pd.DataFrame(0.00, index=scenario_list, columns=scenario_list)  # create empty dataframe

# In[] Define the classifier settings

model_params = {
    "booster": "gbtree",
    "n_jobs": -1,
    "use_label_encoder": False,  # wtf

    "objective": "binary:logistic",  # https://xgboost.readthedocs.io/en/latest/parameter.html
    "eval_metric": "logloss",
    "seed": seed,
}

param_grid = {
    "n_estimators": [100, 500, 1000],  # original 500
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "gamma": [0, 0.10, 0.15, 0.25, 0.5],
    "max_depth": [4, 6, 8, 10, 12, 15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'max_delta_step': [0, 1, 5, 10],  # for imbalanced data
    "scale_pos_weight": [1, 5, 10, 20, 50, 100]
    # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
}

search_settings = {
    "param_distributions": param_grid,
    "scoring": 'balanced_accuracy',
    # f1_micro https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    "n_jobs": -1,
    "n_iter": 500,  # 500 original
    "verbose": 10
}


# In[] Define the classification metrics

def roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr)


def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


# In[] Loop through scenarios

count_X = 0

for scenario in scenario_list:

    count_X += 1

    # In[] Load the feature matrix
    df_feat = pd.read_excel(file_name, sheet_name=scenario, index_col=0)

    df_feat = df_feat.replace(['PD'], 1)
    df_feat = df_feat.replace(['HC'], 0)

    X = df_feat.iloc[:, 0:-1].values
    y = df_feat.iloc[:, -1].values

    # In[] Search for the best hyper-parameters

    print('Hyper-parameters tuning - scenario:' + scenario)

    # Create the classifier
    model = xgb.XGBClassifier(**model_params)

    # Get the cross-validation indices
    kfolds = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # Employ the hyper-parameter tuning
    random_search = RandomizedSearchCV(model, cv=kfolds.split(X, y), random_state=seed, **search_settings)
    random_search.fit(X, y)

    best_tuning_score = random_search.best_score_
    best_tuning_hyperparams = random_search.best_estimator_

    # In[] Evaluate the classifier using cross-validation

    params = random_search.best_params_

    # Get the classifier
    model = xgb.XGBClassifier(**random_search.best_params_)

    # Prepare the cross-validation scheme
    kfolds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=seed)

    # Prepare the scoring
    scoring = {
        "mcc": make_scorer(matthews_corrcoef, greater_is_better=True),
        'F1': make_scorer(f1_score, greater_is_better=True),
        "roc_auc": make_scorer(roc_auc, greater_is_better=True),
        "acc": make_scorer(accuracy_score, greater_is_better=True),
        "sen": make_scorer(sensitivity_score, greater_is_better=True),
        "spe": make_scorer(specificity_score, greater_is_better=True)
    }

    # Cross-validate the classifier
    cv_results = cross_validate(model, X, y, scoring=scoring, cv=kfolds)

    # Compute the mean and std of the metrics
    cls_report = {
        "mcc_avg": round(float(np.mean(cv_results["test_mcc"])), 4),
        "mcc_std": round(float(np.std(cv_results["test_mcc"])), 4),
        "F1_avg": round(float(np.mean(cv_results["test_F1"])), 4),
        "F1_std": round(float(np.std(cv_results["test_F1"])), 4),
        "AUC_avg": round(float(np.mean(cv_results["test_roc_auc"])), 4),
        "AUC_std": round(float(np.std(cv_results["test_roc_auc"])), 4),
        "acc_avg": round(float(np.mean(cv_results["test_acc"])), 4),
        "acc_std": round(float(np.std(cv_results["test_acc"])), 4),
        "sen_avg": round(float(np.mean(cv_results["test_sen"])), 4),
        "sen_std": round(float(np.std(cv_results["test_sen"])), 4),
        "spe_avg": round(float(np.mean(cv_results["test_spe"])), 4),
        "spe_std": round(float(np.std(cv_results["test_spe"])), 4)

    }

    mcc = f"{cls_report['mcc_avg']:.2f} +- {cls_report['mcc_std']:.2f}"
    F1 = f"{cls_report['F1_avg']:.2f} +- {cls_report['F1_std']:.2f}"
    AUC = f"{cls_report['AUC_avg']:.2f} +- {cls_report['AUC_std']:.2f}"
    acc = f"{cls_report['acc_avg']:.2f} +- {cls_report['acc_std']:.2f}"
    sen = f"{cls_report['sen_avg']:.2f} +- {cls_report['sen_std']:.2f}"
    spe = f"{cls_report['spe_avg']:.2f} +- {cls_report['spe_std']:.2f}"

    # print(f" mcc = {mcc}, F1 = {F1}, AUC = {AUC}, ACC = {acc}, SEN = {sen}, SPE = {spe} \n")

    df_performance.loc[scenario, :] = [mcc, F1, AUC, acc, sen, spe]

    # In[] Feature importances

    model.fit(X, y)

    feature_list = list(df_feat.columns)
    feature_list.pop()
    df_importances = pd.DataFrame(model.feature_importances_, index=feature_list, columns=['importance'])
    df_importances = df_importances.sort_values(by=['importance'], ascending=False, key=pd.Series.abs)

    # Export table

    if export_table == 1:
        df_importances.to_excel(writer_imp, sheet_name=scenario)

    # In[] Shap values

    X = df_feat.iloc[:, 0:-1]
    shap_values = shap.TreeExplainer(model).shap_values(X)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = shap.summary_plot(shap_values, X, max_display=10)

    fig.savefig('results/SHAP_' + scenario + '_summary.pdf')
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = shap.summary_plot(shap_values, X, max_display=10, plot_type="bar")
    fig.savefig('results/SHAP_' + scenario + '_mean.pdf')
    plt.close()

    # In[] Cross-language (transfer learning)

    if not only_one_scenario == 1:

        for scenario_test in scenario_list:

            df_feat_test = pd.read_excel(file_name, sheet_name=scenario_test, index_col=0)

            df_feat_test = df_feat_test.replace(['PD'], 1)
            df_feat_test = df_feat_test.replace(['HC'], 0)

            X_test = df_feat_test.iloc[:, 0:-1].values
            y_test = df_feat_test.iloc[:, -1].values

            y_pred = model.predict(X_test)

            mcc = metrics.matthews_corrcoef(y_test, y_pred)
            F1 = metrics.f1_score(y_test, y_pred)
            AUC = roc_auc(y_test, y_pred)
            acc = metrics.accuracy_score(y_test, y_pred)
            sen = sensitivity_score(y_test, y_pred)
            spe = specificity_score(y_test, y_pred)

            df_cross_language.loc[scenario_test, :] = [mcc, F1, AUC, acc, sen, spe]
            df_cross_language_MCC.loc[scenario_test, scenario] = round(mcc, 2)

        # Export table

        if export_table == 1:
            df_cross_language.to_excel(writer_cross, sheet_name=scenario)

# In[] export tables

if export_table == 1:
    df_performance.to_excel(writer_per, sheet_name='performance')

    if not only_one_scenario == 1:
        df_cross_language_MCC.to_excel(writer_cross_mcc, sheet_name='MCC')

# In[] save and close excel files
if export_table == 1:
    writer_imp.save()
    writer_per.save()

    if not only_one_scenario == 1:
        writer_cross.save()
        writer_cross_mcc.save()

# In[]