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

# In[] Set the script

all_features = True  # False = use features in features_list
only_one_scenario = False  # True = process just one scenario (otherwise loop)
export_table = True  # export four tables in total

scenario = 'IL'  # choose language in the case of only_one_scenario = True

# In[] Variables

features_list = ['TSK3-HRF', 'TSK3-NAQ', 'TSK3-QOQ', 'TSK3-relF0SD', 'TSK3-Jitter (PPQ)', 'TSK2-RFA1', 'TSK2-RFA2',
                 'TSK2-#loc_max', 'TSK2-relF2SD', 'TSK2-#lndmrk', 'TSK7-relSDSD', 'TSK7-COV', 'TSK7-RI', 'TSK2-relF0SD',
                 'TSK2-SPIR']  # robust features according to stat. analysis (int the case of all_features = False)

features_file_name = 'data/features_adjusted.csv'
clinical_file_name = 'data/labels.csv'

output_filename_imp = 'results/feature_importances.xlsx'  # feature importances
output_filename_per = 'results/model_performance.xlsx'  # model performances (cross-validation)
output_filename_cross = 'results/cross_language.xlsx'  # cross-language validation
output_filename_shap = 'results/SHAP_all.pdf'  # shap values of the model trained by all languages

scenario_list = ['CZ', 'US', 'IL', 'CO', 'IT', 'all']  # languages (all = all languages together)
metric_list = ['mcc', 'F1', 'AUC', 'acc', 'sen', 'spe']  # list of metrics in final table

seed = 42  # for random search and cross-validation

# In[] Load data

if all_features:
    df_feat = pd.read_csv(features_file_name, sep=';', index_col=0)
    features_list = list(df_feat.columns)
else:
    df_feat = pd.read_csv(features_file_name, sep=';', index_col=0, usecols=['ID'] + features_list)
df_clin = pd.read_csv(clinical_file_name, sep=';', index_col=0, usecols=['ID', 'nationality', 'diagnosis'])

df_data = df_feat.copy().join(df_clin['diagnosis'])

# In[] Prepare tables

if export_table:
    writer_imp = pd.ExcelWriter(output_filename_imp)
    writer_per = pd.ExcelWriter(output_filename_per)
    if not only_one_scenario:
        writer_cross = pd.ExcelWriter(output_filename_cross)

if only_one_scenario:
    scenario_list = list([scenario])
else:
    imp_array = np.ones((len(features_list), 1))  # prepare array for global feature importnaces

scenarios = scenario_list.copy()
if 'all' in scenario_list:
    scenarios.remove('all')

df_performance = pd.DataFrame(0.00, index=scenario_list, columns=metric_list)  # create empty dataframe
df_cross_language = pd.DataFrame(0.00, index=scenarios, columns=metric_list)  # create empty dataframe


# In[] Define the function to replace HC for 0 and PD for 1

def replace_diagnosis_with_numbers(df):
    df_out = df.replace(['HC', 'PD'], [0, 1])
    return df_out


# In[] Define the classification metrics

def roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


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
        # aux series for stratification according to language-diagnosis
        r = df_scenario.iloc[:, -1].values
    else:
        df_scenario = df_data.loc[df_clin['nationality'] == scenario]
        df_scenario = replace_diagnosis_with_numbers(df_scenario)

        X = df_scenario.iloc[:, 0:-1].values
        y = df_scenario.iloc[:, -1].values
        r = y  # only basic stratification

    # In[] Search for the best hyper-parameters

    print('Hyper-parameters tuning - scenario:' + scenario)

    # create the classifier
    model = xgb.XGBClassifier(**model_params)

    # 10-fold cross-validation
    kfolds = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # employ the hyper-parameter tuning
    random_search = RandomizedSearchCV(
        model, cv=kfolds.split(X, r), random_state=seed, **search_settings)

    random_search.fit(X, y)

    best_tuning_score = random_search.best_score_
    best_tuning_hyperparams = random_search.best_estimator_

    # In[] Evaluate the classifier using cross-validation

    params = random_search.best_params_

    # get the classifier
    model = xgb.XGBClassifier(**random_search.best_params_)

    # prepare the cross-validation scheme
    kfolds = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=20, random_state=seed)

    # prepare the scoring
    scoring = {
        "mcc": make_scorer(matthews_corrcoef, greater_is_better=True),
        'F1': make_scorer(f1_score, greater_is_better=True),
        "roc_auc": make_scorer(roc_auc, greater_is_better=True),
        "acc": make_scorer(accuracy_score, greater_is_better=True),
        "sen": make_scorer(recall_score, greater_is_better=True),
        "spe": make_scorer(specificity_score, greater_is_better=True)
    }

    # cross-validate the classifier
    cv_results = cross_validate(
        model, X, y, scoring=scoring, cv=kfolds.split(X, r))

    # get the mean and std of the metrics
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

    df_importances = pd.DataFrame(
        model.feature_importances_, index=features_list, columns=['coefficient'])
    df_importances = df_importances.sort_values(
        by=['coefficient'], ascending=False, key=pd.Series.abs)

    # export table

    if export_table:
        df_importances.to_excel(writer_imp, sheet_name=scenario)

    # In[] Shap values

    if scenario == 'all':
        X = df_scenario.iloc[:, 0:-2]
        shap_values = shap.TreeExplainer(model).shap_values(X)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax = shap.summary_plot(shap_values, X, max_display=17)

        fig.savefig(output_filename_shap)
        plt.close()

    # In[] Store feature importances for calculation of global feature importances

    elif not only_one_scenario:
        imp_array = np.append(imp_array, model.feature_importances_.reshape(-1, 1), axis=1)

    # In[] Cross-language (transfer learning)

    if not only_one_scenario:
        if not scenario == 'all':
            for scenario_test in scenarios:

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

            # export table

            if export_table:
                df_cross_language.to_excel(writer_cross, sheet_name='Train-' + scenario)

# In[] Global feature importances

if not only_one_scenario:
    multi = np.prod(imp_array, 1)  # multiply feature importances of languages
    multi_norm = multi/max(multi)  # normalisation (the most important feature has value equal to 1)

    df_imp_glob = pd.DataFrame(multi_norm, index=features_list, columns=['coefficient'])
    df_imp_glob = df_imp_glob.sort_values(by=['coefficient'], ascending=False, key=pd.Series.abs)

    if export_table:
        df_imp_glob.to_excel(writer_imp, sheet_name='global')

# In[] Export tables

if export_table:
    df_performance.to_excel(writer_per, sheet_name='performance')

# In[] Save and close excel files

if export_table:
    writer_imp.save()
    writer_per.save()

    if not only_one_scenario:
        writer_cross.save()

# In[]

print('Script finished.')
