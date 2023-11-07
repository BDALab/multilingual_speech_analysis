# Multilingual speech analysis

In this project, we analyze robustness of various acoustic features to changes in language.
We extracted the features from speech recordings of people with Parkinson's disease and healthy controls.
The original recordings come from five data sets, each recorded in a different language.
Statistical tests and descriptive statistics are used to analyze language robustness.
The downstream task is a binary classification of the recordings (healthy or parkinson's) based on the features.
We used XGBoost classifier to solve the task and extract the feature importances.
Moreover, we analyze the feature importance using SHAP values.

## Research article

In case you build upon this work, please, cite our work.

```
Biomedical Signal Processing and Control: [doi.org/10.1016/j.bspc.2023.105667](https://doi.org/10.1016/j.bspc.2023.105667)
MedRxiv: [doi.org/10.1101/2022.10.24.22281459](https:/doi.org/10.1101/2022.10.24.22281459)
```

## Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the
recordings, the features, nor the labels. However, we provide the source
code (in folder `src`), so you can run the same experiments on your own data. 

### Install the dependencies

```
# Clone the repo
git clone git@github.com:BDALab/multilingual_speech_analysis.git
cd multilingual_speech_analysis

# Create a virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate

# Install the deps from the last known working environment with exact versions
pip install -r requirements.txt
```

If you want to experiment with different versions of packages here is a
list of those that need to be installed: `shap`, `numpy`, `scipy`, `pandas`,
`sklearn`, `seaborn`, `xgboost`, `matplotlib`, `statsmodels`.

### Prepare the data

Create two csv files, `data/labels.csv` and `data/features.csv` delimited with `;`.
Each of the files must contain one subject per row. The first column
in each file must be a unique subject ID. Both files must be sorted the same way.

- Structure of `data/labels.csv`:

| column name     | dtype        | description
| -------------   | ----------   | -----------
| ID              | str or int   | Unique identifier of each subject
| nationality     | str          | Used for stratification
| diagnosis       | 'HC' or 'PD' | Used for stratification and as a target of classification
| sex             | 'M' or 'F'   | Used for data adjustment
| age             | numeric      | Used for data adjustment
| duration_of_PD  | numeric      | Time since the first symptoms
| LED             | numeric      | Daily dose of medication (L-dopa)
| UPDRSIII        | numeric      | Unified Parkinson Disease Rating Scale (part 3)
| UPDRSIII-speech | numeric      | Same as previous but only for speech
| H&Y             | numeric      | Hoehn & Yahr rating scale

Clinical data starting with the row "duration_of_PD" can be changed or extended.

- Structure of `data/features.csv`:

| column name     | dtype        | description
| -------------   | ----------   | -----------
| ID              | str or int   | Unique identifier of each subject
| feature_1_name  | numeric      | Feature values computed from raw audio
| ...             | numeric      | ...
| feature_N_name  | numeric      | Use as many features as you need

### Repeat the experiments

1. run `src/data_info.py` to get `results/violin_graph.pdf`, `results/demograf.xlsx` and `results/demograf.xlsx`
2. run `src/adjust_features.py` to obtain `data/features_adjusted.csv`
3. run `src/get_statistics.py` to get `results/stats_results.xlsx`
4. run `src/correlations.py` to get `results/spearman.xlsx`
5. run `src/XGBoost.py` to get `results/feature_importances.xlsx`, `results/model_performance.xlsx`,
 `results/cross_language.xlsx` and `results/SHAP_all.pdf`
6. run `src/XGBoost_leave-one-language-out.py` to get `results/leave_one_language_out.xlsx`

## License

This project is licensed under the terms of the MIT license.

## Acknowledgement

This work was supported by the Czech Ministry of Health under grant no. NU20-04-00294,
by EU -- Next Generation EU (project no. LX22NPO5107 (MEYS)), and by the European Union's Horizon 2020
research and innovation program under the Marie Skłodowska-Curie grant agreement no. 734718 (CoBeN).
