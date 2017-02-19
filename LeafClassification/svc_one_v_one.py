# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:20:33 2016

@author: claudine
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

# load data
train_data = pd.read_csv('train.csv', sep=',', header=0)
test_data = pd.read_csv('test.csv', sep=',', header=0)

test_ids = test_data.id
test_data = test_data.drop(['id'], axis=1)

# check label balances
labels = train_data['species']
features = train_data.drop(['id', 'species'], axis=1)

le = LabelEncoder().fit(labels)
labels = le.transform(labels)

# print(np.unique(labels, return_counts=True))  # all labels occur 8 times each

# splitdata
sfs = StratifiedShuffleSplit(labels, test_size=0.2)
for train_index, test_index in sfs:
    # needs iloc to get indexes straight
    X_train, X_valid = features.iloc[train_index], features.iloc[test_index]
    Y_train, Y_valid = labels[train_index], labels[test_index]


# CONTINUE TO WORK WITH TRAIN SPLIT ONLY !!!
# train models on X_train and Y_train
clf = SVC(probability=True)
model_to_set = OneVsRestClassifier(clf)
parameters = {'estimator__C': [1, 2, 4, 8],
              'estimator__kernel': ['poly', 'rbf'],
              'estimator__degree': [1, 2, 3, 4]}

model_tuning = GridSearchCV(model_to_set, param_grid=parameters)

model_tuning.fit(X_train, Y_train)

print('One vs Rest SVC best score: ', model_tuning.best_score_)
print('One vs Rest SVC best params: ', model_tuning.best_params_)

# predict on X_valid
pred_labels = model_tuning.predict(X_valid)
proba_labels = model_tuning.predict_proba(X_valid)

# evaluate performance with confusion matrix
cm = confusion_matrix(Y_valid, pred_labels)

# evaluate performance with log_loss metric
print("log loss: ", log_loss(Y_valid, proba_labels))

# if model has been optimized, predict on test data
predictions = model_tuning.predict_proba(test_data)

# prepare submission
classes = list(le.classes_)
submission = pd.DataFrame(predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()
submission.to_csv("submission.csv", index=False)
