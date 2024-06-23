import pandas as pd
import numpy as np
from scipy import stats

df_both = pd.read_csv('data/results/classifier/svm_results_both.csv')
df_predictability = pd.read_csv('data/results/classifier/svm_results_predictability.csv')
df_metric_values = pd.read_csv('data/results/classifier/svm_results_metric_values.csv')

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def div0( a, b ):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def bowker(table):

    num_cols, num_rows = table.shape
    tril = np.tril(table, k=-1)
    triu = np.triu(table, k=1)
    numer = (tril - triu.T)**2
    denom = (tril + triu.T)
    quot = div0(numer, denom)
    statistic = np.sum(quot)
    df = int(num_rows * (num_rows -1) / 2)
    pvalue = stats.chi2.sf(statistic, df)
    return DotDict({'pvalue' : pvalue, 'statistic' : statistic, 'df' : df})

y_pred_model1 = df_metric_values['predictions']
y_pred_model2 = df_both['predictions']

# Number of classes
num_classes = 3

# Initialize the contingency table
contingency_table = pd.DataFrame(0, index=range(num_classes), columns=range(num_classes))

# Fill the contingency table based on the predictions of both models
for pred1, pred2 in zip(y_pred_model1, y_pred_model2):
    contingency_table.iloc[pred1, pred2] += 1

print("Contingency Table:")
print(contingency_table)

# Convert pandas DataFrame to numpy array for the bowker function
contingency_table_np = contingency_table.to_numpy()

# Perform the Bowker-McNemar test
result = bowker(contingency_table_np)

print(f"Chi-square statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
print(f"Degrees of freedom: {result.df}")

