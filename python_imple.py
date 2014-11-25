# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 23:21:41 2014

@author: Phani
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

data = pd.read_csv("D:/Analytics/CA/Exacerbation/CAX_ExacerbationModeling_TRAIN_data.csv")
test = pd.read_csv("D:/Analytics/CA/Exacerbation/CAX_ExacerbationModeling_Public_TEST_data.csv")
data_total = data.append(test)

imp = Imputer(missing_values='NaN', strategy='median', axis=0, copy = False)
data_total_imputed = imp.fit_transform(data_total)

train = pd.DataFrame(data_total_imputed[1:4099,:])
evaluation = pd.DataFrame(data_total_imputed[4099:,:])

train.columns = data.columns
evaluation.columns = data.columns

train_0 = train[train['Exacebator'] == 0]
train_1 = train[train['Exacebator'] == 1]

samp = np.random.choice(train_0.index.values, 350)
train_0_samp = train_0.ix[samp]

train = train_1.append(train_0_samp)
print train['Exacebator'].value_counts()

colsToDrop = ['Exacebator','sid']

y_train = train['Exacebator']
X_train = train.drop(colsToDrop, axis = 1)
skfold_train = StratifiedKFold(y_train, n_folds = 10)

y_eval = evaluation['Exacebator']
X_eval = evaluation.drop(colsToDrop, axis = 1)
skfold_eval = StratifiedKFold(y_eval, n_folds = 10)

## SGD Classifer
sgd = SGDClassifier(n_jobs = -1, penalty = 'l1', loss = 'log', n_iter = 10, l1_ratio = 0.85)
scores_sgd = cross_val_score(sgd, X_train, y_train, scoring = "accuracy", cv = skfold_train)
print("SGD CV Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd.mean(), scores_sgd.std() * 2))

sgd.fit(X_train,y_train)

X_train_preds_sgd = sgd.predict_proba(X_train)
X_train_preds_sgd_np = np.asarray(X_train_preds_sgd)
X_train_preds_sgd_1 = X_train_preds_sgd_np[:,1]
#plt.plot(X_preds_sgd_1)
#plt.show()

X_eval_preds_sgd = sgd.predict_proba(X_eval)
X_eval_preds_sgd_np = np.asarray(X_eval_preds_sgd)
X_eval_preds_sgd_1 = X_eval_preds_sgd_np[:,1]

## RF classifier
rf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = True)
scores_rf = cross_val_score(rf, X_train, y_train, scoring = "f1", cv = skfold_train)
print("RF CV Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

rf.fit(X_train, y_train)

X_train_preds_rf = rf.predict_proba(X_train)
X_train_preds_rf_np = np.asarray(X_train_preds_rf)
X_train_preds_rf_1 = X_train_preds_rf[:,1]
#plt.plot(X_train_preds_rf_1)
#plt.show()

X_eval_preds_rf = rf.predict_proba(X_eval)
X_eval_preds_rf_np = np.asarray(X_eval_preds_rf)
X_eval_preds_rf_1 = X_eval_preds_rf_np[:,1]

X_train_preds_rf_2round = np.around(X_train_preds_rf_1, decimals = 2)
X_train_preds_sgd_2round = np.around(X_train_preds_sgd_1, decimals = 2)
X_eval_preds_rf_2round = np.around(X_eval_preds_rf_1, decimals = 2)
X_eval_preds_rf_2round = np.around(X_eval_preds_sgd_1, decimals = 2)

train['rfoutcome'] = X_train_preds_rf_2round
train['sgdoutcome'] = X_train_preds_sgd_2round
evaluation['rfoutcome'] = X_eval_preds_rf_2round
evaluation['sgdoutcome'] = X_eval_preds_rf_2round

train.to_csv("D:/Analytics/CA/Exacerbation/CA-Exacerbator/CA-Exacerbator/data/train_new.csv", index = False)
evaluation.to_csv("D:/Analytics/CA/Exacerbation/CA-Exacerbator/CA-Exacerbator/data/evaluation_new.csv", index = False)
