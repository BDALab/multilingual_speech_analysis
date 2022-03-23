
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

sns.set_theme()

# In[] Variables

clinical_file_name = "labels_data.xlsx"

# In[] Load data

df_clin = pd.read_excel(clinical_file_name, index_col=0)

# In[] Plot data
colors = ['lightsteelblue', "peachpuff"]
fig, ax = plt.subplots( nrows=1, ncols=1 )
ax = sns.violinplot(x="nationality", y="age", hue="diagnosis",
                    data=df_clin, palette=colors, split=True,
                    scale="count", inner="quartile")
plt.legend(loc='lower left')
fig.savefig("results/age.pdf", bbox_inches='tight')
plt.close()

