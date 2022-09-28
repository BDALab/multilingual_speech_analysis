# Multilingual speech analysis

In this project, we analyze robustness of various acoustic features to changes in language.
We extracted the features from speech recordings of people with Parkinson's disease and healthy controls.
The original recordings come from five data sets, each recorded in a different language.
The downstream task is a binary classification of the recordings (healthy or parkinson's) based on the features.
We used XGBoost classifier to solve the task and extract the feature importances.
Moreover, we analyze the feature importance using SHAP values.

## Research article

The findings of this study are published in [update this ArXiv link](https://arxiv.org).
Please, in case you build upon this work, cite the article above.

```bibtex
@article{kovac2022,
  title={Title,
  author={Kovac et al.},
  journal={Journal name},
  volume={XX},
  number={0},
  pages={0--10},
  year={2022},
  publisher={Publisher}
}
```

## Reproducibility

The data sets of the original audio recordings are private, but we provide the
tables (in folder `data`) with the computed features for reproducibility of
all the results presented in the article. Along with the data used in this study,
we provide the source code to compute the features on your own recordings
([update this git repo](https://github.com)), as well as the code (in folder
`src`) to repeat our experiments.

### Code

```
# Clone the repo
git clone git@github.com:BDALab/multilingual_speech_analysis.git
cd multilingual_speech_analysis

# Create a virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### Data

* `extracted_features.xlsx`
  \- Contains results of parametrization based on raw audio files.
* `labels_data.xlsx`
  \- Contains a list of all recordings (one per row) along with their labels.

### Steps to repeat the experiments

1. run `adjust_features.py` to get `adjusted_features.xlsx`
2. files `dataset_*.xlsx` are just manually restructured `adjusted_features.xlsx`
3. run `correlations.py` to obtain `results/spearman.xlsx`
4. run `get_statistics.py` to obtain `results/stats_results.xlsx`
5. run `XGBoost.py` to obtain feature importances and classification results
6. run `XGBoost_leave-one-language-out.py` to obtain `leave-one-language-out.xlsx`


## License
This project is licensed under the terms of the MIT license.

## Acknowledgement

This work was supported by ...
