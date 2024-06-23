import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def main():
    df = pd.read_csv('features_set3.csv', index_col=0)
    y = df['prominence_discrete']

    x_both = df[['predictability', 'head_complement_normalized', 'left_right_normalized', 'lexical_functional']]
    x_pred = df[['predictability']]
    x_metric = df[['head_complement_normalized', 'left_right_normalized', 'lexical_functional']]

    classifier_both = pickle.load(open('svm_both.sav', 'rb'))
    classifier_pred = pickle.load(open('svm_pred.sav', 'rb'))
    classifier_metric = pickle.load(open('svm_metric.sav', 'rb'))

    y_both = classifier_both.predict(x_both)
    y_pred = classifier_pred.predict(x_pred)
    y_metric = classifier_metric.predict(x_metric)

    print(confusion_matrix(y, y_both))
    print(confusion_matrix(y, y_metric))
    print(confusion_matrix(y, y_pred))

    print(f1_score(y, y_both))
    print(f1_score(y,y_metric))
    print(f1_score(y, y_pred))

    f1_scores_both = []
    f1_scores_pred = []
    f1_scores_metric = []
    for i in range(100):
        y_temp = y.copy()
        for j in range(len(y)):
            r = random.random()
            if y_temp.loc[j] == 1 and r > 0.86:
                y_temp.loc[j] = 0
            elif y_temp.loc[j] == 0 and r > 0.86:
                y_temp.loc[j] = 1
        f1_scores_pred.append(f1_score(y_temp, y_pred, average='weighted'))
        f1_scores_metric.append(f1_score(y_temp, y_metric, average='weighted'))
        f1_scores_both.append(f1_score(y_temp, y_both, average='weighted'))
    print(f1_scores_pred)
    print(f1_scores_metric)
    print(f1_scores_both)

    print(np.average(f1_scores_both))
    print(np.average(f1_scores_metric))
    print(np.average(f1_scores_pred))

    count_metric = 0
    count_pred = 0
    for i in range(len(f1_scores_metric)):
        if f1_scores_metric[i] > f1_scores_both[i]:
            count_metric += 1
        if f1_scores_pred[i] > f1_scores_both[i]:
            count_pred += 1

    print(count_metric)
    print(count_pred)

if __name__ == '__main__':

    main()