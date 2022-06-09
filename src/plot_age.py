import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

# In[] Variables

clinical_file_name = 'data/labels.csv'
output_file_name = 'results/age.pdf'

# In[] Load data

df_clin = pd.read_csv(clinical_file_name, index_col=0, sep=';')

# In[] Plot data
colors = ['lightsteelblue', "peachpuff"]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.violinplot(x="nationality", y="age", hue="diagnosis",
                    data=df_clin, palette=colors, split=True,
                    scale="count", inner="quartile")
plt.legend(loc='lower left')
os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
fig.savefig(output_file_name, bbox_inches='tight')
plt.close()
