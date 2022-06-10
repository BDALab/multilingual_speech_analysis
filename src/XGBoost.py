import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, auc
from sklearn.metrics import matthews_corrcoef, recall_score, f1_score, roc_curve

"""
Script for computing machine learning model performance:
 - stratified k fold cross-validation
 - cross-language validation
 - feature importances of model trained by all labels of particular language
 - SHAP values of model trained by all languages
"""

# In[] variables

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'

output_filename_imp = 'results/feature_importances.xlsx'  # feature importances
output_filename_per = 'results/model_performance.pdf'  # model performances (cross-validation)
output_filename_cross = 'results/cross_language.xlsx'  # cross-language validation

scenario_list = list(['CZ', 'US', 'IL', 'CO', 'IT', 'all'])  # nationality (all = all nationality together)

only_one_scenario = False  # True = process just one scenario (otherwise loop)
scenario = 'IL'  # choose language in the case of only_one_scenario = True
export_table = True  # export four tables in total
seed = 42  # for random search and cross-validation

# In[] Set the script

if export_table:
    writer_imp = pd.ExcelWriter(output_filename_imp)
    writer_per = pd.ExcelWriter(output_filename_per)

    if not only_one_scenario:
        writer_cross = pd.ExcelWriter(output_filename_cross)

if only_one_scenario:
    scenario_list = list([scenario])

# In[] Load data

df_feat = pd.read_csv(features_file_name, sep=';', index_col=0)
df_clin = pd.read_csv(clinical_file_name, sep=';', index_col=0, usecols=['ID', 'nationality', 'diagnosis'])

df_data = df_feat.copy().join(df_clin['diagnosis'])

metric_list = ['mcc', 'F1', 'AUC', 'acc', 'sen', 'spe']  # list of metrics in final table
df_performance = pd.DataFrame(0.00, index=scenario_list, columns=metric_list)  # create empty dataframe
df_cross_language = df_performance.copy()


# In[] Replace HC for 0 and PD for 1

def replace_diagnosis_with_numbers(df):
    df_out = df.replace(['HC', 'PD'], [0, 1])
    return df_out


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


# In[] Define the classification metrics

def roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


# In[] Loop through scenarios

count_X = 0

for scenario in scenario_list:

    count_X += 1

    # In[] Load the feature matrix

    if scenario == 'all':
        df_scenario = df_data.copy()
        df_scenario["class"] = df_clin["nationality"] + '-' + df_data["diagnosis"]
        df_scenario = replace_diagnosis_with_numbers(df_scenario)

        X = df_scenario.iloc[:, 0:-2].values
        y = df_scenario.iloc[:, -2].values
        # Aux series for stratification according to language-diagnosis
        r = df_scenario.iloc[:, -1].values
    else:
        df_scenario = df_data.loc[df_clin['nationality'] == scenario]
        df_scenario = replace_diagnosis_with_numbers(df_scenario)

        X = df_scenario.iloc[:, 0:-1].values
        y = df_scenario.iloc[:, -1].values
        r = y  # Only basic stratification

    # In[] Search for the best hyper-parameters

    print('Hyper-parameters tuning - scenario:' + scenario)

    # Create the classifier
    model = xgb.XGBClassifier(**model_params)

    # 10-fold cross-validation
    kfolds = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # Employ the hyper-parameter tuning
    random_search = RandomizedSearchCV(
        model, cv=kfolds.split(X, r), random_state=seed, **search_settings)

    random_search.fit(X, y)

    best_tuning_score = random_search.best_score_
    best_tuning_hyperparams = random_search.best_estimator_

    # In[] Evaluate the classifier using cross-validation

    params = random_search.best_params_

    # Get the classifier
    model = xgb.XGBClassifier(**random_search.best_params_)

    # Prepare the cross-validation scheme
    kfolds = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=20, random_state=seed)

    # Prepare the scoring
    scoring = {
        "mcc": make_scorer(matthews_corrcoef, greater_is_better=True),
        'F1': make_scorer(f1_score, greater_is_better=True),
        "roc_auc": make_scorer(roc_auc, greater_is_better=True),
        "acc": make_scorer(accuracy_score, greater_is_better=True),
        "sen": make_scorer(recall_score, greater_is_better=True),
        "spe": make_scorer(specificity_score, greater_is_better=True)
    }

    # Cross-validate the classifier
    cv_results = cross_validate(
        model, X, y, scoring=scoring, cv=kfolds.split(X, r))

    # Get the mean and std of the metrics
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

    df_performance.loc[scenario, :] = [mcc, F1, AUC, acc, sen, spe]

    # In[] Feature importances

    model.fit(X, y)

    feature_list = list(df_feat.columns)
    df_importances = pd.DataFrame(
        model.feature_importances_, index=feature_list, columns=['importance'])
    df_importances = df_importances.sort_values(
        by=['importance'], ascending=False, key=pd.Series.abs)

    # Export table

    if export_table:
        df_importances.to_excel(writer_imp, sheet_name=scenario)

    # In[] Shap values

    shap_values = shap.TreeExplainer(model).shap_values(X)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = shap.summary_plot(shap_values, X, max_display=17)

    fig.savefig('results/SHAP_' + scenario + '_summary.pdf')
    plt.close()

    # In[] Cross-language (transfer learning)

    if not only_one_scenario:
        if not scenario == 'all':
            for scenario_test in scenario_list:

                df_scenario_test = df_data.loc[df_clin['nationality'] == scenario_test]
                df_scenario_test = replace_diagnosis_with_numbers(df_scenario_test)

                X_test = df_scenario_test.iloc[:, 0:-1].values
                y_test = df_scenario_test.iloc[:, -1].values

                y_pred = model.predict(X_test)

                mcc = metrics.matthews_corrcoef(y_test, y_pred)
                F1 = metrics.f1_score(y_test, y_pred)
                AUC = roc_auc(y_test, y_pred)
                acc = metrics.accuracy_score(y_test, y_pred)
                sen = recall_score(y_test, y_pred)
                spe = specificity_score(y_test, y_pred)

                df_cross_language.loc[scenario_test, :] = [mcc, F1, AUC, acc, sen, spe]

            # Export table

            if export_table:
                df_cross_language.to_excel(writer_cross, sheet_name='Training scenario: ' + scenario)

# In[] export tables

if export_table:
    df_performance.to_excel(writer_per, sheet_name='performance')

# In[] save and close excel files
if export_table:
    writer_imp.save()
    writer_per.save()

    if not only_one_scenario:
        writer_cross.save()

# In[]

print('Script finished.')
