# Multilingual speech analysis

In this project, we analyze robustness of various acoustic features to changes in language.
We extracted the features from speech recordings of people with Parkinson's disease and healthy controls.
The original recordings come from five data sets, each recorded in a different language.
The downstream task is a binary classification of the recordings (healthy or parkinson's) based on the features.
We used XGBoost classifier to solve the task and extract the feature importances.
Moreover, we analyze the feature importance using SHAP values.

## Research article

The findings of this study are published in [update this ArXiv link](https://arxiv.org).
In case you build upon this work, please, cite our work.

```bibtex
@article{kovac2022,
  title={Title,
  author={Kovac et al.},
  journal={Journal name},
  volume={XX},
  number={0},
  pages={0--10},
  year={2022},
  publisher={Publisher},
  eprint={arXiv}
}
```

## Reproducibility

Due to the licensing of the used data sets, we are not allowed to publish the
recordings, the features, nor the labels. However, we provide all the source
code (in folder `src`), so you can run the same experiments on your own data.
Additionally, in the article, we list the software packages used to compute
the features from the raw audio recordings.  

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
Each of the files must contain one recording per row. The first column
in each file must be a unique recording ID. Both files must be sorted the same way.

Structure of `data/labels.csv`:

| column name     | dtype        | description
| -------------   | ----------   | -----------
| ID              | str or int   | Unique identifier of each recording
| nationality     | str          | Used for stratification
| diagnosis       | 'HC' or 'PD' | Used for stratification and as a target of classification
| sex             | 'M' or 'F'   | Used for data adjustment
| age             | numeric      | Used for data adjustment
| duration_of_PD  | numeric      | Time since the first symptoms
| LED             | numeric      | Daily dose of medication (L-dopa)
| UPDRSIII        | numeric      | Unified Parkinson Disease Rating Scale (part 3)
| UPDRSIII-speech | numeric      | Same as previous but only for speech
| H&Y             | numeric      | Hoehn & Yahr scale

Structure of `data/features.csv`:

| column name     | dtype        | description
| -------------   | ----------   | -----------
| ID              | str or int   | Unique identifier of each recording
| feature_1_name  | numeric      | Feature values computed from raw audio
| ...             | numeric      | ...
| feature_N_name  | numeric      | Use as many features as you need

### Repeat the experiments

1. run `src/adjust_features.py` to get `data/features_adjusted.csv`
2. run `src/plot_age.py` to get `results/age.pdf`
3. run `src/correlations.py` to obtain `results/spearman.csv`
3. configure and run `src/correlations.py` to obtain `results/pearson.csv`
4. run `src/get_statistics.py` to obtain `results/stats_results.csv`
5. run `src/XGBoost.py` to obtain feature importances and classification results
6. run `src/XGBoost_leave-one-language-out.py` to obtain `results/leave-one-language-out.csv`


## License
This project is licensed under the terms of the MIT license.

## Acknowledgement

This work was supported by ...
