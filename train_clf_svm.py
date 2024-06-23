import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

def main():
    use_predictability = True
    use_metric_values = True

    df = pd.read_csv('features.csv', index_col=0)

    y = df['prominence_discrete']

    if use_predictability and use_metric_values:
        x = df[['predictability', 'head_complement_normalized', 'left_right_normalized', 'lexical_functional', 'word', 'sentence', 'ID']]
    elif use_predictability:
        x = df[['predictability', 'word', 'sentence', 'ID']]
    elif use_metric_values:
        x = df[['head_complement_normalized', 'left_right_normalized', 'lexical_functional', 'word', 'sentence', 'ID']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)
    x_train = x_train.drop(['word', 'sentence', 'ID'], axis=1)

    df_info = x_test[['word', 'sentence', 'ID']]
    x_test = x_test.drop(['word', 'sentence', 'ID'], axis=1)

    clf = SVC()
    parameter_grid = {'C': [1, 10, 100],
                  'gamma': [0.1, 0.01, 0.001],
                  }
    classifier_cv = GridSearchCV(clf, parameter_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1)
    classifier_cv.fit(x_train, y_train)
    classifier = classifier_cv.best_estimator_

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_pred_train = classifier.predict(x_train)
    print(classification_report(y_train, y_pred_train, digits=4))

    print(classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))

    coefficients = classifier.coef0.flatten()
    feature_importance_df = pd.DataFrame({
        'Feature': x_train.column.values,
        'Importance': np.abs(coefficients)
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    x_test['target'] = y_test
    x_test['predictions'] = y_pred
    df_test = x_test.join(df_info)

    if use_predictability and use_metric_values:
        df_test.to_csv('data/results/classifier/svm_results_both.csv',index=False)
    elif use_predictability:
        df_test.to_csv('data/results/classifier/svm_results_predictability.csv',index=False)
    elif use_metric_values:
        df_test.to_csv('data/results/classifier/svm_results_metric_values.csv',index=False)

if __name__ == '__main__':

    main()