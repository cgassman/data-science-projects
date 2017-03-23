# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:20:33 2016

@author: claudine
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_data():
    # load data
    train_data = pd.read_csv('../data/train.csv', sep=',', header=0)
    test_data = pd.read_csv('../data/test.csv', sep=',', header=0)

    print('shape of train_data: ', train_data.shape)

#    test_ids = test_data.Id
    test_data = test_data.drop(['Id'], axis=1)

    # split features from target variable
    target_variable = train_data['SalePrice']
    features = train_data.drop(['Id', 'SalePrice'], axis=1)
    return(features, target_variable)


def split_data(features, target_variable):
    # split data
    X_train, X_valid, y_train, y_valid = train_test_split(features,
                                                          target_variable,
                                                          test_size=0.2,
                                                          random_state=0)

#    print('shape X_train: ', X_train.shape)
#    print('shape y_train: ', y_train.shape)
#    print('shape X_valid: ', X_valid.shape)
#    print('shape y_valid: ', y_valid.shape)
    return(X_train, X_valid, y_train, y_valid)


def prep_analysis_sheet(features):
    # prepare data analysis sheet
    attributes = list(features.columns.values)
    attr_types = list(features.dtypes)

    overview = pd.DataFrame({"AttributeNames": attributes,
                             "DataType": attr_types,
                             "VarType": "",
                             "Expectation": "",
                             "Conclusion": "",
                             "Comments": ""})
    overview.to_csv("../output/overview.csv", columns=["AttributeNames",
                                                       "DataType", "VarType",
                                                       "Expectation",
                                                       "Conclusion",
                                                       "Comments"],
                    header=True, index=False)


features, target_variable = load_data()
prep_analysis_sheet(features)
X_train, X_valid, y_train, y_valid = split_data(features, target_variable)



#
#
## CONTINUE TO WORK WITH TRAIN SPLIT ONLY !!!
#
## train models on X_train and Y_train
#rf = RandomForestClassifier()
#parameters = {'n_estimators': [10, 200, 500, 1000]}
#
#clf = GridSearchCV(rf, parameters)
#clf.fit(X_train, Y_train)
#
#print('best params: ', clf.best_params_)
#
#rf = RandomForestClassifier(n_estimators=500)
#rf.fit(X_train, Y_train)
#importances = rf.feature_importances_
#indices = np.argsort(importances)[::-1]
#
#print('Feature ranking: ')
#for f in range(X_train.shape[1]):
#    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))
#
#plt.figure()
#plt.bar(range(X_train.shape[1]), importances[indices],
#        color='r', align='center')
#plt.xticks(range(X_train.shape[1]), indices)
#plt.xlim([-1, X_train.shape[1]])
#plt.show()
#
## predict on X_valid
#pred_labels = clf.predict(X_valid)
#proba_labels = clf.predict_proba(X_valid)
#
## evaluate performance with confusion matrix
#cm = confusion_matrix(Y_valid, pred_labels)
#print(cm)
#print("pred accuracy score: ", accuracy_score(Y_valid, pred_labels))
#
## evaluate performance with log_loss metric
#print("log loss: ", log_loss(Y_valid, proba_labels))
#
## if model has been optimized, predict on test data
#predictions = clf.predict_proba(test_data)
#
## prepare submission
#classes = list(le.classes_)
#submission = pd.DataFrame(predictions, columns=classes)
#submission.insert(0, 'id', test_ids)
#submission.reset_index()
#submission.to_csv("submission.csv", index=False)
