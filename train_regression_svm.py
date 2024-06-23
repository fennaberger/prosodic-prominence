import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV

def main():
    use_predictability = True
    use_metric_values = True

    df = pd.read_csv('features.csv', index_col=0)

    y = df['prominence_real']

    if use_predictability and use_metric_values:
        x = df[['predictability', 'head_complement_normalized', 'left_right_normalized', 'lexical_functional', 'word', 'sentence', 'ID']]
    elif use_predictability:
        x = df[['predictability', 'word', 'sentence', 'ID']]
    elif use_metric_values:
        x = df[['head_complement_normalized', 'left_right_normalized', 'lexical_functional', 'word', 'sentence', 'ID']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)

    df_info = x_test[['word', 'sentence', 'ID']]
    df_info_train = x_train[['word', 'sentence', 'ID']]
    x_test = x_test.drop(['word', 'sentence', 'ID'], axis=1)
    x_train = x_train.drop(['word', 'sentence', 'ID'], axis=1)

    model = SVR()  # als alles verder klopt wil ik verschillende classifiers gaan vergelijken, weet nog niet welke
    parameter_grid = {'C': [1, 10, 100],
                      'gamma': [0.1, 0.01, 0.001],
                      }
    model_cv = GridSearchCV(model, parameter_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
    model_cv.fit(x_train, y_train)
    final_model = model_cv.best_estimator_

    final_model.fit(x_train, y_train)

    y_pred = final_model.predict(x_test)
    y_pred_train = final_model.predict(x_train)

    r = permutation_importance(final_model, x_test, y_test, n_repeats=30, random_state=6)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{x_train.columns[i]:<8}"
                  f" {r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

    x_test['prediction'] = y_pred
    x_test['target'] = y_test
    df_test = x_test.join(df_info)

    x_train['prediction'] = y_pred_train
    x_train['target'] = y_train
    df_train = x_train.join(df_info_train)

    if use_predictability and use_metric_values:
        df_test.to_csv('data/results/svm_results_both.csv',index=False)
        df_train.to_csv('data/results/svm_results_both_train.csv', index=False)
    elif use_predictability:
        df_test.to_csv('data/results/svm_results_predictability.csv',index=False)
        df_train.to_csv('data/results/svm_results_predictability_train.csv', index=False)
    elif use_metric_values:
        df_test.to_csv('data/results/svm_results_metric_values.csv',index=False)
        df_train.to_csv('data/results/svm_results_metric_values_train.csv', index=False)

if __name__ == '__main__':

    main()