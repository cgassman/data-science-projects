# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:20:33 2016

@author: claudine
"""

import itertools
import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    '''source: from scikit learn'''
    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

print(X_train.shape)
print(X_valid.shape)

# CONTINUE TO WORK WITH TRAIN SPLIT ONLY !!!
# train models on X_train and Y_train
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, Y_train)

# predict on X_valid
pred_labels_lda = lda.predict(X_valid)
proba_labels_lda = lda.predict_proba(X_valid)

pred_labels_qda = qda.predict(X_valid)
proba_labels_qda = qda.predict_proba(X_valid)

# evaluate performance with confusion matrix
cm = confusion_matrix(Y_valid, pred_labels_lda)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=labels)
plt.show()

print("pred accuracy score lda: ", accuracy_score(Y_valid, pred_labels_lda))
print("pred accuracy score qda: ", accuracy_score(Y_valid, pred_labels_qda))

# evaluate performance with log_loss metric
print("log loss lda: ", log_loss(Y_valid, proba_labels_lda))
print("log loss qda: ", log_loss(Y_valid, proba_labels_qda))


# if model has been optimized, predict on test data
predictions = lda.predict_proba(test_data)

# prepare submission
#classes = list(le.classes_)
#submission = pd.DataFrame(predictions, columns=classes)
#submission.insert(0, 'id', test_ids)
#submission.reset_index()
#submission.to_csv("submission.csv", index=False)
