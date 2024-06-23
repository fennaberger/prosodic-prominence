# Import library
from scipy.stats import ttest_rel, pearsonr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pingouin as pg
evaluate_svm = False
evaluate_svm_train = False
evaluate_random_forest = False
evaluate_random_forest_train = False
evaluate_ridge = False
evaluate_ridge_train = True

if evaluate_svm:
    df_both = pd.read_csv('data/results/svm_results_both.csv')
    df_predictability = pd.read_csv('data/results/svm_results_predictability.csv')
    df_metric_values = pd.read_csv('data/results/svm_results_metric_values.csv')

if evaluate_svm_train:
    df_both = pd.read_csv('data/results/svm_results_both_train.csv')
    df_predictability = pd.read_csv('data/results/svm_results_predictability_train.csv')
    df_metric_values = pd.read_csv('data/results/svm_results_metric_values_train.csv')

if evaluate_random_forest:
    df_both = pd.read_csv('data/results/random_forest_results_both.csv')
    df_predictability = pd.read_csv('data/results/random_forest_results_predictability.csv')
    df_metric_values = pd.read_csv('data/results/random_forest_results_metric_values.csv')

if evaluate_random_forest_train:
    df_both = pd.read_csv('data/results/random_forest_results_both_train.csv')
    df_predictability = pd.read_csv('data/results/random_forest_results_predictability_train.csv')
    df_metric_values = pd.read_csv('data/results/random_forest_results_metric_values_train.csv')

if evaluate_ridge:
    df_both = pd.read_csv('data/results/linear_regression_results_both.csv')
    df_predictability = pd.read_csv('data/results/linear_regression_results_predictability.csv')
    df_metric_values = pd.read_csv('data/results/linear_regression_results_metric_values.csv')

if evaluate_ridge_train:
    df_both = pd.read_csv('data/results/linear_regression_results_both_train.csv')
    df_predictability = pd.read_csv('data/results/linear_regression_results_predictability_train.csv')
    df_metric_values = pd.read_csv('data/results/linear_regression_results_metric_values_train.csv')

df_both['difference'] = abs(df_both['target'] - df_both['prediction'])
df_predictability['difference'] = abs(df_predictability['target'] - df_predictability['prediction'])
df_metric_values['difference'] = abs(df_metric_values['target'] - df_metric_values['prediction'])


print(f"mean metric values {np.mean(df_metric_values['difference'])}")
print(f"mean predictability and metric values: {np.mean(df_both['difference'])}")
print(f"mean predictability: {np.mean(df_predictability['difference'])}")

print('-------')
print(f"standard deviation metric values: {np.std(df_metric_values['difference'])}")
print(f"standard deviation predictability and metric values: {np.std(df_both['difference'])}")
print(f"standard deviation predictability: {np.std(df_predictability['difference'])}")

rms_pred = mean_squared_error(df_predictability['target'], df_predictability['prediction'], squared=True)
rms_metric = mean_squared_error(df_metric_values['target'], df_metric_values['prediction'], squared=True)
rms_both = mean_squared_error(df_both['target'], df_both['prediction'], squared=True)

print('-------')
print(f"RMSE metric values: {rms_metric}")
print(f"RMSE predictability and metric values: {rms_both}")
print(f"RMSE predictabiility: {rms_pred}")

test_predictability_both = ttest_rel(df_predictability['difference'], df_both['difference'])
test_metric_values_both = ttest_rel(df_metric_values['difference'], df_both['difference'])
print('--------')
print(f"t-test predictability, both: {test_predictability_both}")


t = pg.ttest(df_predictability['difference'], df_both['difference'], paired=True).round(4)
pg.print_table(t)

print(f"t-test metric values, both: {test_metric_values_both}")

t = pg.ttest(df_metric_values['difference'], df_both['difference'], paired=True).round(4)
pg.print_table(t)