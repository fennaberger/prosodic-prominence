import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def main():
    use_predictability = True
    use_metric_values = False

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

    model = RandomForestRegressor()  # als alles verder klopt wil ik verschillende classifiers gaan vergelijken, weet nog niet welke
    parameter_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
    }
    model_cv = GridSearchCV(model, parameter_grid, cv=5, scoring='neg_mean_absolute_error', verbose=True, n_jobs=-1)
    model_cv.fit(x_train, y_train)
    final_model = model_cv.best_estimator_

    final_model.fit(x_train, y_train)

    y_pred = final_model.predict(x_test)
    y_pred_train = final_model.predict(x_train)

    x_test['prediction'] = y_pred
    x_test['target'] = y_test
    df_test = x_test.join(df_info)

    x_train['prediction'] = y_pred_train
    x_train['target'] = y_train
    df_train = x_train.join(df_info_train)

    if use_predictability and use_metric_values:
        df_test.to_csv('data/results/random_forest_results_both.csv',index=False)
        df_train.to_csv('data/results/random_forest_results_both_train.csv', index=False)
    elif use_predictability:
        df_test.to_csv('data/results/random_forest_results_predictability.csv',index=False)
        df_train.to_csv('data/results/random_forest_results_predictability_train.csv', index=False)
    elif use_metric_values:
        df_test.to_csv('data/results/random_forest_results_metric_values.csv',index=False)
        df_train.to_csv('data/results/random_forest_results_metric_values_train.csv', index=False)

if __name__ == '__main__':

    main()