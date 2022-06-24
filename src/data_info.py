import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Script for computing clinical data and numbers of men, women, HCs and PDs in each dataset.
"""

sns.set_theme()

# In[] Variables

clinical_file_name = 'data/labels.csv'
output_file_name_plot = 'results/violin_graph.pdf'
output_file_name_demo = 'results/demograf.xlsx'
output_file_name_clin = 'results/clinical.xlsx'

scenario_list = ['CZ', 'US', 'IL', 'CO', 'IT']
data_plot = 'age'  # to plot as a violin graph for each scenario, all in one picture

export_table = True

# In[] Load data

df = pd.read_csv(clinical_file_name, index_col=0, sep=';')

# In[] Prepare table

df_demograf = pd.DataFrame(index=['HC', 'PD', 'total'], columns=pd.MultiIndex.from_product(
    [scenario_list, ['F', 'M', 'total']], names=['Scenarios', 'Scores']))

df_clin = pd.DataFrame(index=df.columns[4:], columns=scenario_list)

# In[] Count men, women, HCs and PDs

HC_total_al = 0
PD_total_al = 0

for scenario in scenario_list:
    diag = df.loc[df["nationality"] == scenario].iloc[:]['diagnosis'].values
    sex = df.loc[df["nationality"] == scenario].iloc[:]['sex']

    df_demograf.loc['HC', (scenario, 'F')] = ((diag == 'HC') * (sex == 'F')).sum()
    df_demograf.loc['HC', (scenario, 'M')] = ((diag == 'HC') * (sex == 'M')).sum()
    df_demograf.loc['PD', (scenario, 'F')] = ((diag == 'PD') * (sex == 'F')).sum()
    df_demograf.loc['PD', (scenario, 'M')] = ((diag == 'PD') * (sex == 'M')).sum()

    df_demograf.loc['total', (scenario, 'F')] = df_demograf.loc['HC', (scenario, 'F')] + \
                                                df_demograf.loc['PD', (scenario, 'F')]
    df_demograf.loc['total', (scenario, 'M')] = df_demograf.loc['HC', (scenario, 'M')] + \
                                                df_demograf.loc['PD', (scenario, 'M')]

    HC_total = df_demograf.loc['HC', (scenario, 'F')] + \
               df_demograf.loc['HC', (scenario, 'M')]
    PD_total = df_demograf.loc['PD', (scenario, 'F')] + \
               df_demograf.loc['PD', (scenario, 'M')]

    df_demograf.loc['HC', (scenario, 'total')] = HC_total
    df_demograf.loc['PD', (scenario, 'total')] = PD_total

    df_demograf.loc['total', (scenario, 'total')] = HC_total + PD_total

    HC_total_al = HC_total_al + HC_total
    PD_total_al = PD_total_al + PD_total

df_demograf.loc['HC', 'altogether'] = HC_total_al
df_demograf.loc['PD', 'altogether'] = PD_total_al
df_demograf.loc['total', 'altogether'] = HC_total_al + PD_total_al

df_demograf = df_demograf.astype('int')

# In[] Get clinical data

for scenario in scenario_list:
    for clin in df_clin.index:
        data = df.loc[df["nationality"] == scenario].iloc[:][clin].values
        nan_mask = np.isnan(data)
        data_without_nan = data[~nan_mask]
        if np.any(data_without_nan):
            df_clin.loc[clin, scenario] = str(np.round(np.mean(data_without_nan), 3)) + u"\u00B1" + \
                                      str(np.round(np.std(data_without_nan), 3))
        else:
            df_clin.loc[clin, scenario] = 'U'

# In[] Export tables

if export_table:
    df_demograf.to_excel(output_file_name_demo)
    df_clin.to_excel(output_file_name_clin)

# In[] Plot data
colors = ['lightsteelblue', "peachpuff"]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.violinplot(x="nationality", y=data_plot, hue="diagnosis",
                    data=df, palette=colors, split=True,
                    scale="count", inner="quartile")
plt.legend(loc='lower left')
os.makedirs(os.path.dirname(output_file_name_plot), exist_ok=True)
fig.savefig(output_file_name_plot, bbox_inches='tight')
plt.close()

# In[]

print('Script finished.')
