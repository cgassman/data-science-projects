# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:20:33 2016

@author: claudine
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


def load_data():
    # load data
    train_data = pd.read_csv('../data/train.csv', sep=',', header=0)
    test_data = pd.read_csv('../data/test.csv', sep=',', header=0)

    return(train_data, test_data)


def split_data(train_data):

    # split features from target variable
    target_variable = train_data['SalePrice']
    features = train_data.drop(['Id', 'SalePrice'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(features,
                                                          target_variable,
                                                          test_size=0.2,
                                                          random_state=0)

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


def verify_feat_importance(train_data):

    target = train_data['SalePrice']
    features = train_data.drop(['Id', 'SalePrice'], axis=1)
    
    feature_names = features.columns

    # no text allowed anymore
    clf = RandomForestClassifier()
    clf.fit(features, target)
    

    importances = clf.feature_importances_
    d ={'names':feature_names, 'importances':importances}
    
    df = pd.DataFrame(d, columns=['names', 'importances'])
    df = df.sort_values(by='importances', ascending=False)
    
    print(df)

    plt.figure()
    plt.bar(range(features.shape[1]), df['importances'],
            color='r', align='center')
    plt.xticks(range(features.shape[1]),
               df['names'], rotation=90)
    plt.xlim([-1, features.shape[1]/3])
    plt.show()

    return


def analyze_missing_data(dataset):

    # sets NA == 1. counts how many NAs there are in each column.
    total = dataset.isnull().sum().sort_values(ascending=False)

    percent = (dataset.isnull().sum()/dataset.isnull().count()). \
        sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1,
                             keys=['TotalMissing', 'Percent'])

    print(missing_data.head(20))

    return


def deal_with_missing_data(train_data):
    # delete all features that have more than 15% missing data, as these
    # cannot meaningful be imputted.
    train_data = train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence',
                                  'FireplaceQu', 'LotFrontage'], axis=1)

    # TODO - delete observations which have NA values < 10 due to simplification.
    # in a next step find a solution for imputation
    train_data = train_data.drop(train_data.loc[train_data['MasVnrArea'].
                                                isnull()].index)
    train_data = train_data.drop(train_data.loc[train_data['MasVnrType'].
                                                isnull()].index)
    train_data = train_data.drop(train_data.loc[train_data['Electrical'].
                                                isnull()].index)

    # TODO - also delete all the other columns containing NA. But find an imputing
    # solution for them too
    train_data = train_data.drop(['GarageCond', 'GarageType', 'GarageYrBlt',
                                  'GarageFinish', 'GarageQual', 'BsmtExposure',
                                  'BsmtFinType2', 'BsmtFinType1', 'BsmtCond',
                                  'BsmtQual'], axis=1)

    return(train_data)


def descriptive_analysis(variable):

    print("variable {}".format(variable.name))
    print(variable.describe())
    print("*" * 30)

    plt.figure()
    sns.distplot(variable)

    return


def inference_analysis(variable):

    return


def plot_corr_matrix(train_data):
    plt.figure()
    k = 10  # number of variables for heatmap
    corrmat = train_data.corr()
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train_data[cols].values.T)
    sns.set(font_scale=0.75)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 5}, yticklabels=cols.values,
                xticklabels=cols.values)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    return


def handle_cat_data(train_data):
    """ convert all object and """
    return pd.get_dummies(train_data, drop_first=True)


def handle_num_data(train_data, var_name, func):

    return


def clf_initial_comparison(X_train, X_valid, y_train, y_valid):

    classifiers = [RandomForestClassifier(),
                   LinearRegression(),
                   Lasso(),
                   ElasticNet(),
                   Ridge(),
                   SVR(),
                   NuSVR(),
                   LinearSVR()]

    log_cols = ["Classifier", "Accuracy", "Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)

        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('**** Results ****')
        train_predictions = clf.predict(X_valid)
        acc = accuracy_score(y_valid, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_valid)
        loss = mean_squared_error(y_valid, train_predictions)
        print("Loss: {}".format(loss))

        log_entry = pd.DataFrame([[name, acc*100, loss]], columns=log_cols)
        log = log.append(log_entry)

    print("="*30)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color='b')

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Mean Square', y='Classifier', data=log, color='g')

    plt.xlabel('Mean Square Loss')
    plt.title('Classifier Mean Square Loss')
    plt.show()


def format_submission(predictions):
    ##    test_ids = test_data.Id
#    test_data = test_data.drop(['Id'], axis=1)
#classes = list(le.classes_)
#submission = pd.DataFrame(predictions, columns=classes)
#submission.insert(0, 'id', test_ids)
#submission.reset_index()
#submission.to_csv("submission.csv", index=False)
    return


""" start """
train_data, test_data = load_data()
print('original shape of train_data: ', train_data.shape)

prep_analysis_sheet(train_data)

# MISSING VALUES
analyze_missing_data(train_data)
train_data = deal_with_missing_data(train_data)
#print('missing values: ', train_data.isnull().sum().max())
#print("shape of train data, after missing values deletion:", train_data.shape)



# HANDLE CATEGORICAL VARIABLES
train_data = handle_cat_data(train_data)
#print("shape after convertion of cat features to dummies:",
#      train_data.shape)

verify_feat_importance(train_data)

plot_corr_matrix(train_data)

# ANALYSE NUMERICAL VARIABLES
descriptive_analysis(train_data["LotArea"])  # log transformation
descriptive_analysis(train_data["MasVnrArea"])  # ebenfalls log-traf versuchen
descriptive_analysis(train_data["TotalBsmtSF"])  # log-trafo
descriptive_analysis(train_data["GrLivArea"])  # log-trafo? komische figur

# SPLIT INTO TRAIN AND VALID
X_train, X_valid, y_train, y_valid = split_data(train_data)
#print('shape X_train: ', X_train.shape)
#print('shape y_train: ', y_train.shape)
#print('shape X_valid: ', X_valid.shape)
#print('shape y_valid: ', y_valid.shape)




# BUILD MODEL
#clf_init_comparison(X_train, X_valid, y_train, y_valid)


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

