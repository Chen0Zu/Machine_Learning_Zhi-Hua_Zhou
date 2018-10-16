import numpy as np
import pandas as pd

def data_process(data):
    n,d = data.shape
    X = np.zeros([n,d])
    for i in range(d):
        tmp = data.iloc[:,i].astype('category')
        X[:,i] = np.array(tmp.cat.rename_categories(range(len(tmp.unique()))))
    return X

def entropy(y):
    n = y.shape[0]
    classes = np.unique(y)
    classes_n = classes.shape[0]
    p = np.zeros([classes_n,1])
    information_entopy = 0;
    for i in range(classes_n):
        p[i] = sum(y == i)/n
        information_entopy = information_entopy - (p[i]*np.log2(p[i]))
    return information_entopy

def gain(x,y):
    n = y.shape[0]
    values = np.unique(x)
    information_gain = entropy(y)
    for i in values:
        information_gain = information_gain - sum(x == i)/n*entropy(y[x==i])
    return information_gain

def choose_attr(X,y):
    n,d = X.shape
    gains = np.zeros([d,1])
    for i in range(d):
        gains[i] = gain(X[:,i],y)
    return gains.argmax()

def majority_vote(y):
    classes = np.unique(y)
    votes = np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        votes[i] = sum(y == classes[i])
    
    return classes[votes.argmax()]

def tree_generate(X,y,fea):
    
    if sum(y == y[0]) == len(y):
        return y[0] # 样本全属于同一类别
    
    if len(fea) == 0:
        return majority_vote(y) # 无特征可以划分,将类别标记为样本中数最多的类
    
    fea_idx = choose_attr(X,y)
    best_fea = fea[fea_idx]
    tree = {best_fea:{}}
    x_values = X[:,fea_idx]
    fea_values = np.unique(x_values)
    
    for value in fea_values:
        split_X = np.delete(X[x_values == value,:],fea_idx,axis=1)
        split_y = y[x_values == value]
        split_fea = np.delete(fea, fea_idx)
        tree[best_fea][value] = tree_generate(split_X,split_y,split_fea)
        
    return tree

# =============================================================================
watermelon = pd.read_csv('./西瓜数据集2.0.csv')
data = data_process(watermelon)
X = data[:,range(6)]
y = data[:,-1].reshape(-1,1)
fea = np.array(range(6))

# learn model
tree = tree_generate(X,y,fea)

# predict



