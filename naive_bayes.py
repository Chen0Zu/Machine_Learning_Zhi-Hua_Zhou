import numpy as np
import pandas as pd

def data_process(data):
    n,d = data.shape
    X = np.zeros([n,d])
    for i in range(d):
        tmp = data.iloc[:,i].astype('category')
        X[:,i] = np.array(tmp.cat.rename_categories(range(len(tmp.unique()))))
    return X

def naive_bayes(X,Y):
    n,d = X.shape
    classes = np.unique(Y)
    n_classes = len(classes)
    pri = np.zeros([n_classes,1])
    
    for i in range(n_classes):
        pri[i] = sum(Y==classes[i])/n
        
    con_p = {}
    for i in range(d):
        fea_value = np.unique(X[:,i])
        n_value = len(fea_value)
        con_p.setdefault(i,np.zeros([n_classes,n_value]))
        for j in range(n_value):
            for k in range(n_classes):
                c_idx = Y == classes[k]
                x_idx = X[:,i] == fea_value[j]
                con_p[i][k,j] = sum(c_idx & x_idx)/sum(c_idx)
    return (con_p,pri)

data = pd.read_csv('./西瓜数据集3.0.csv')
X_raw = data_process(data)
X = X_raw[:,0:6]
Y = X_raw[:,-1]
con_p,pri = naive_bayes(X,Y)