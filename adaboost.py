import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

def decision_stump(X, dim, threshold, strategy):
    n,d = X.shape
    pred_label = np.ones([n,1])
    if strategy == 'lt':
        pred_label[X[:,dim<=threshold]] = -1.0
    else:
        pred_label[X[:,dim>threshold]] = -1.0
            
    return pred_label

def build_classifier(X,Y,weight):
    n,d = X.shape
    interval = 10.0
    classifier = {}
    best_classifier_predict = np.zeros([n,1])
    min_err = float('inf')
    
    for i in range(d):
        fea_min = X[:,i].min()
        fea_max = X[:,i].max()
        setpsize = (fea_max - fea_min) / interval
        for j in range(-1,int(interval)+1):
            for strategy in ['lt','gt']:
                threshold = (fea_min + float(j)*setpsize)
                pred_label = decision_stump(X,i,threshold, strategy)
                wrong_idx = np.ones([n,1])
                wrong_idx[pred_label == Y] = 0
                predict_err = weight.T @ wrong_idx
                
                if predict_err < min_err:
                    min_err = predict_err
                    classifier['dim'] = i
                    classifier['threshold'] = threshold
                    classifier['strategy'] = strategy
                    best_classifier_predict = pred_label
    return classifier, min_err, best_classifier_predict

def adaboost_train(X,Y,iteration = 30):
    n,d = X.shape
    weak_classifier = []
    weights = np.ones([n,1])/n
    for i in range(iteration):
        best_classifier, error, pred_label = build_classifier(X,Y,weights)
        alpha = 0.5*np.log((1-error)/error)
        best_classifier['alpha'] = alpha
        weak_classifier.append(best_classifier)
        weights = weights*np.exp(-alpha*Y*pred_label)
        weights = weights/np.sum(weights,axis = 1, keepdims = True)
    
    return weak_classifier

