# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 23:21:41 2014

@author: Phani
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

data = pd.read_csv("D:/CA/Exacerbation/CAX_ExacerbationModeling_TRAIN_data.csv")
test = pd.read_csv("D:/CA/Exacerbation/CAX_ExacerbationModeling_Public_TEST_data.csv")

## Imputation
data_total = data.append(test)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, copy = False)
data_total_imputed = imp.fit_transform(data_total)

## Train and Evaluation Split
train = pd.DataFrame(data_total_imputed[1:4099,:])
evaluation = pd.DataFrame(data_total_imputed[4099:,:])
train.columns = data.columns
evaluation.columns = data.columns

## Sampling
train_0 = train[train['Exacebator'] == 0]
train_1 = train[train['Exacebator'] == 1]

samp = np.random.choice(train_0.index.values, 343)
train_0_samp = train_0.ix[samp]
train = train_1.append(train_0_samp)

# Shuffling
train = train.iloc[np.random.permutation(np.arange(len(train)))]

# Ignoring sid variable
colsToDrop = ['Exacebator','sid']
y_train = train['Exacebator']
X_train = train.drop(colsToDrop, axis = 1)
y_eval = evaluation['Exacebator']
X_eval = evaluation.drop(colsToDrop, axis = 1)

## Stratified K fold for cross validation
skfold_train = StratifiedKFold(y_train, n_folds = 10)

## RF classifier
rf = RandomForestClassifier(n_estimators = 3000, n_jobs = -1, verbose = True)
scores_rf = cross_val_score(rf, X_train, y_train, scoring = "f1", cv = skfold_train)
print("RF CV Accuracy: %0.3f (+/- %0.3f)" % (scores_rf.mean(), scores_rf.std() * 2))

rf.fit(X_train, y_train)

X_eval_preds_rf = rf.predict_proba(X_eval)
X_eval_preds_rf_np = np.asarray(X_eval_preds_rf)
X_eval_preds_rf_1 = X_eval_preds_rf_np[:,1]

## GBM classifier
param_grid = { 'learning_rate' : [0.01, 0.02, 0.05, 0.1],
               'max_depth'     : [4,6],
               'min_samples_leaf' : [3,5,9,17],
                'max_features' : 35
                }

gbm = GradientBoostingClassifier(n_estimators = 2500, max_features = 45, subsample = 0.85, verbose = True)
#gs_cv = GridSearchCV(gbm, param_grid, scoring = 'f1', n_jobs = -1).fit(X_train, y_train)
#gs_cv.best_params_
scores_gbm = cross_val_score(gbm, X_train, y_train, scoring="f1", cv=skfold_train, n_jobs=-1, verbose=True)
print("GBM CV Accuracy: %0.3f (+/- %0.3f)" % (scores_gbm.mean(), scores_gbm.std() * 2))

gbm.fit(X_train, y_train)

indices_gbm = np.argsort(gbm.feature_importances_)
X_train.columns[indices_gbm]

X_eval_preds_gbm = gbm.predict_proba(X_eval)
X_eval_preds_gbm_np = np.asarray(X_eval_preds_gbm)
X_eval_preds_gbm_1 = X_eval_preds_gbm_np[:,1]

# Extra Trees Classifer
ext = ExtraTreesClassifier(n_jobs = -1, n_estimators = 3500, verbose = True, bootstrap = True, oob_score = True)
scores_ext = cross_val_score(ext, X_train, y_train, scoring = "f1", cv = skfold_train)
print("Extra Trees CV Accuracy: %0.3f (+/- %0.3f)" % (scores_ext.mean(), scores_ext.std() * 2))

ext.fit(X_train,y_train)

X_eval_preds_ext = ext.predict_proba(X_eval)
X_eval_preds_ext_np = np.asarray(X_eval_preds_ext)
X_eval_preds_ext_1 = X_eval_preds_ext_np[:,1]

evaluation['rfoutcome'] = X_eval_preds_rf_1
evaluation['gbmoutcome'] = X_eval_preds_gbm_1
evaluation['extoutcome'] = X_eval_preds_ext_1

evaluation.sid = evaluation.sid.astype(int)
evaluation.sid.dtypes
submission = pd.DataFrame({'sid':evaluation.sid, 'Exacebator':evaluation.gbmoutcome})
submission = submission.sort_index(axis=1, ascending = False)
submission.head()
## Change submission name
submission.to_csv("D:/CA/Exacerbation/ExacerbatorProject/CA-Exacerbator/results/GBM_1500_P.csv", index = False)

#train.to_csv("D:/CA/Exacerbation/ExacerbatorProject/CA-Exacerbator/data/train_new.csv", index = False)
#evaluation.to_csv("D:/CA/Exacerbation/ExacerbatorProject/CA-Exacerbator/data/evaluation_new.csv", index = False)
