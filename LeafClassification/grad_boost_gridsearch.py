# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:20:33 2016

@author: claudine
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

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
sfs = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
sfs.get_n_splits(features, labels)
for train_index, test_index in sfs.split(features, labels):
    # needs iloc to get indexes straight
    X_train, X_valid = features.iloc[train_index], features.iloc[test_index]
    Y_train, Y_valid = labels[train_index], labels[test_index]


# CONTINUE TO WORK WITH TRAIN SPLIT ONLY !!!

# train models on X_train and Y_train
gbc = GradientBoostingClassifier()
parameters = {'learning_rate': [0.1, 0.01],
              'max_depth': [3, 5],
              'n_estimators': [500, 700]}

clf = GridSearchCV(gbc, parameters, scoring='neg_log_loss', verbose=5)
clf.fit(X_train, Y_train)

print('best params: ', clf.best_params_)

#rf = RandomForestClassifier(n_estimators=500)
#rf.fit(X_train, Y_train)
#importances = rf.feature_importances_
#indices = np.argsort(importances)[::-1]

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

# predict on X_valid
pred_labels = clf.predict(X_valid)
proba_labels = clf.predict_proba(X_valid)

# evaluate performance with confusion matrix
cm = confusion_matrix(Y_valid, pred_labels)
print(cm)
print("pred accuracy score: ", accuracy_score(Y_valid, pred_labels))

# evaluate performance with log_loss metric
print("log loss: ", log_loss(Y_valid, proba_labels))

# if model has been optimized, predict on test data
predictions = clf.predict_proba(test_data)

# prepare submission
classes = list(le.classes_)
submission = pd.DataFrame(predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()
submission.to_csv("submission.csv", index=False)
