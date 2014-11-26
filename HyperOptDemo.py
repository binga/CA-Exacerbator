import os
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

from sklearn.metrics import accuracy_score as accuracy
import pandas as pd

from hyperopt import hp, fmin, tpe


run_counter = 0

def main(params):
    print "started.."
    
    global run_counter
    
    run_counter += 1
    print "run", run_counter
    model = "rf"
    
    start_time = time()
    accu = run_test(params, model)
    print "elapsed: {}s".format(time() - start_time)
    #print "error", accu, "params", list(params)
    print "error", accu, "params", list(params),"\n"
    #o_f.flush()
    return accu

def AccuracyErrorCalc( y, p ):
    return 1 - accuracy(y, p)

def run_test(params, model):
    
    if model == "rf":
        n_tree, mtry = params
        print "# Trees: ", n_tree
        print "mtry: ", mtry
        rf = RandomForestClassifier(n_estimators= int(n_tree), verbose = True, 
                                n_jobs = -1, max_features= int(mtry))
        rf.fit(X, y)
        modelPred = rf.predict(X)
    elif model == "svm":
        C, kernel = params
        print "# Cost: ", C
        print "kernel: ", kernel
        svmod = SVC(int(C), kernel)
        svmod.fit(X, y)
        modelPred = svmod.predict(X)
    elif model == "knn":
        k = params
        print "# k: ", k
        knnmod = KNeighborsClassifier(int(k))
        knnmod.fit(X, y)
        modelPred =knnmod.predict(X)
    elif model == "NeuralNetwork":
        n_components, learning_rate, batch_size, n_iter = params
        print "# n_components: ", n_components
        print "# learning_rate: ", learning_rate
        print "# batch_size: ", batch_size
        print "# n_iter: ", n_iter 
        nnmod = BernoulliRBM(int(n_components), learning_rate, int(batch_size), int(n_iter))
        nnmod.fit(X, y)
        modelPred =nnmod.score_samples(X)
    
    accuError = AccuracyErrorCalc(y, modelPred)
    return accuError

if __name__ == "__main__":
    data = pd.read_csv("D:/Datasets/sampath/Datasets/original/UniversalBank.csv")
    #data = pd.read_csv("D:/Datasets/sampath/Datasets/original/WineQuality.csv")
    print "dataset dimensions: ", data.shape
    print "dataset columns: ", data.dtypes
    print "Current working directory: ", os.getcwd()
    
    X = data.ix[:, 0:12]
    y = data['CreditCard']
    
    
    # rf
    space = (hp.quniform('n_tree', 100, 500, 1), hp.quniform('mtry', 2, 10, 1))
    
    # svm
    #space = (hp.quniform('C', 100, 150, 1), hp.choice('kernel', ['linear', 'rbf']))
    
    # knn
    #space = (hp.choice('k', [1, 3, 5, 7, 9, 11]))
    
    #nn
    #space = (hp.quniform('n_components', 4, 7, 1), hp.quniform('learning_rate', 0.1, 0.5, 1),
    #         hp.quniform('batch_size', 1000, 2000, 1),hp.quniform('n_iter', 10, 20, 1))
    
    
    start_time = time()
    best = fmin(main, space, algo= tpe.suggest, max_evals = 10)
    print "Total time elapsed: {}s".format(time() - start_time)
    
    print "Best parameters: ", best