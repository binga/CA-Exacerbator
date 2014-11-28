# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:18:39 2014

@author: phanisrikanth
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, space_eval, rand
from time import time

data = pd.read_csv("D:/CA/Exacerbation/CAX_ExacerbationModeling_TRAIN_data.csv")
test = pd.read_csv("D:/CA/Exacerbation/CAX_ExacerbationModeling_Public_TEST_data.csv")

## Imputation
data_total = data.append(test)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, copy = False)
data_total_imputed = imp.fit_transform(data_total)

## Train and Evaluation Split
train = pd.DataFrame(data_total_imputed[0:4099,:])
evaluation = pd.DataFrame(data_total_imputed[4099:,:])
train.columns = data.columns
evaluation.columns = data.columns

# Sampling
train_0 = train[train['Exacebator'] == 0]
train_1 = train[train['Exacebator'] == 1]

samp = np.random.choice(train_0.index.values, 343)
train_0_samp = train_0.ix[samp]
train = train_1.append(train_0_samp)

#Shuffling
train = train.iloc[np.random.permutation(np.arange(len(train)))]

# Ignoring sid variable
colsToDrop = ['Exacebator','sid']
y_train = train['Exacebator']
X_train = train.drop(colsToDrop, axis = 1)
y_eval = evaluation['Exacebator']
X_eval = evaluation.drop(colsToDrop, axis = 1)

def run_test(params, model):
    if model == "rf":
        n_tree, mtry, criterion, max_depth = params
        rf = RandomForestClassifier(n_estimators= int(n_tree), max_features= int(mtry),
                                    criterion = criterion, max_depth = max_depth, n_jobs = -1, oob_score = True)
        rf.fit(X_train, y_train)
        scores = rf.oob_score_
    elif model == "gbm":
        n_tree, max_features, subsample = params
        gbm = GradientBoostingClassifier(n_estimators = n_tree, max_features = max_features,
                                           subsample = subsample, n_jobs = -1)
        cv = StratifiedKFold(y_train, 10)
        scores = cross_val_score(rf, X_train, y_train, cv = cv, n_jobs = 4, scoring= 'f1')
    elif model == "ext":
        n_tree, criterion, max_features = params
        ext = ExtraTreesClassifier(n_estimators = n_tree, criterion = criterion, 
                                   max_features = max_features, n_jobs = -1)
        cv = StratifiedKFold(y_train, 10)
        scores = cross_val_score(rf, X_train, y_train, cv = cv, n_jobs = 4, scoring= 'f1')
    cvError = 1 - np.array(scores).mean()
    return cvError

def main(space, model = "rf"):
    global run_counter
    run_counter += 1
    model = "rf"
    #start_time = time()
    error = run_test(space, model)
    print "Run:", run_counter, "CV loss:", error, "Space:", list(space)
    #print "Time taken: %0.2f sec" % (time()-start_time)
    return error

run_counter = 0

# Hyperopt search
#rf
space = (hp.quniform('n_tree', 500, 3000, 1), hp.quniform('mtry', 40, 100, 1),
         hp.choice('criterion', ['entropy', 'gini']), hp.choice('max_depth', [4,5,6]))
model = "rf"
start_time = time()
best = fmin(main, space, algo= tpe.suggest, max_evals = 10)
print "Total time elapsed: {}s".format(time() - start_time)
print space_eval(space, best)
print "Best parameters: ", best

from sklearn import metrics
def custom_precision_score(y_true,y_pred):
    precision_tuple, recall_tuple, fscore_tuple, support_tuple = metrics.precision_recall_fscore_support(y_true, y_pred)  
    recallMinClass = recall_tuple[1]
    print recall_tuple
    return recallMinClass

custom_scorer = metrics.make_scorer(custom_precision_score)
rf = RandomForestClassifier(max_features= 55, n_estimators= 1500, criterion= 'entropy', max_depth= 6, n_jobs = -1, verbose = True, oob_score = True)
rf.fit(X_train, y_train)
rf.oob_score_
scores = cross_val_score(rf, X_train, y_train, cv=3, scoring=custom_scorer)

