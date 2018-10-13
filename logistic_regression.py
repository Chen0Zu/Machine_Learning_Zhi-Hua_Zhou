import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_bfgs


np.random.seed(0)

def sigmoid(x):
    h = 1/(1+np.exp(-x))
    return h

def compute_cost(w,X,y):
    w = w.reshape(-1,1)
    J = -sum(y*(X@w) - np.log(1+np.exp(X@w)))
    return J.flatten()

def compute_grad(w,X,y):
    w = w.reshape(-1,1)
    grad = np.dot(X.T,(sigmoid(np.dot(X,w))-y))
    return grad.flatten()

def logistic_regression(X,y):
    learning_rate = 0.1
    n,d = X.shape
    iteration = 200
    
    w = np.zeros([d+1,1])
    w_h = np.zeros([d+1,1])
    X = np.c_[X,np.ones([n,1])]
    J = np.zeros([iteration,1])
    J_h = np.zeros([iteration,1])
    
    for i in range(iteration):
        # 梯度下降
        dw = X.T@(sigmoid(X@w)-y)       
        w = w - learning_rate*dw
        J[i] = -sum(y*(X@w) - np.log(1+np.exp(X@w)))
        
        # 牛顿法
        dw_h = X.T@(sigmoid(X@w_h) - y)
        D = sigmoid(X@w_h)*(1-sigmoid(X@w_h))
        D = D.reshape(-1)
        Hessian = X.T @ np.diag(D) @ X
        w_h = w_h - np.linalg.inv(Hessian)@dw_h
        J_h[i] = -sum(y*(X@w_h) - np.log(1+np.exp(X@w_h)))
        
    return (w,w_h,J,J_h)

def predict(X,y,w):
    n = X.shape[0]
    w = w.reshape(-1,1)
    pre_label = np.sign(np.c_[X,np.ones([n,1])] @ w)
    pre_label[pre_label == -1] = 0
    accuracy = np.mean(pre_label == y)
    return pre_label,accuracy

#==============================================================================
# show sigmoid function
z = np.linspace(-10, 10, 100)
zy = 1/(1+np.exp(-z))
plt.figure()
plt.plot(z,zy)

# generate data
mean1 = [-1, -1]
mean2 = [1, 1]
sigma = np.eye(2)
n = 100
x1 = np.random.multivariate_normal(mean1, sigma, int(n/2))
x2 = np.random.multivariate_normal(mean2, sigma, int(n/2))
X = np.r_[x1,x2]
y = np.r_[np.zeros(int(n/2)), np.ones(int(n/2))].reshape(-1,1)
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y.flatten())

# learn model
w_g,w_h,J,J_h = logistic_regression(X,y)
plt.figure()
plt.plot(J)
plt.figure()
plt.plot(J_h)
# show decision boundary
x_pos = np.linspace(-3,3,10)
y_pos = (-w_h[0]*x_pos-w_h[2])/w_h[1]
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y.flatten())
plt.plot(x_pos,y_pos)

# predct
pre_label,accuracy = predict(X,y,w_h)
print("Accuracy is", accuracy)

#==============================================================================
watermelon = pd.read_csv('西瓜数据集3.0α.csv')
X = watermelon.iloc[:,0:2].values
n = X.shape[0]
label = watermelon.iloc[:,-1].values.reshape(-1,1)
w,w_h,J,J_h = logistic_regression(X,label)
plt.figure()
plt.plot(J)
plt.figure()
plt.plot(J_h)

# predct
pre_label,accuracy = predict(X,label,w_h)
print("Accuracy of watermelon is", accuracy)

#==============================================================================
# 10 folds
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import load_iris
nkfolds = 10;
data = load_iris()
class_idx1 = data.target == 0
class_idx2 = data.target == 1
idx = class_idx1 | class_idx2
y = data.target[idx].reshape(-1,1)
X = data.data[idx,:]
d = X.shape[1]

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
accuracy = []
for train_idx, test_idx in rskf.split(y,y):
    print("%s %s" % (train_idx,test_idx))
    n_train_idx = len(train_idx)
    n_test_idx = len(test_idx)
    w = np.zeros([d+1,1])
    X_train = np.c_[X[train_idx],np.ones([n_train_idx,1])]
    y_train = y[train_idx]
    opt_result = fmin_bfgs(compute_cost, w, compute_grad, args = (X_train,y_train), disp=True, maxiter=400, full_output = True, retall=True)     
    X_test = np.c_[X[test_idx], np.ones([n_test_idx,1])]
    y_test = y[test_idx]
    pre_label,acc = predict(X[test_idx],y_test,opt_result[0])
    accuracy.append(acc)
fold_10_acc = np.mean(accuracy)
#==============================================================================
# leave one out
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = []
for train_idx, test_idx in loo.split(y,y):
    print("%s %s" % (train_idx,test_idx))
    n_train_idx = len(train_idx)
    n_test_idx = len(test_idx)
    w = np.zeros([d+1,1])
    X_train = np.c_[X[train_idx],np.ones([n_train_idx,1])]
    y_train = y[train_idx]
    opt_result = fmin_bfgs(compute_cost, w, compute_grad, args = (X_train,y_train), disp=True, maxiter=400, full_output = True, retall=True)     
    X_test = np.c_[X[test_idx], np.ones([n_test_idx,1])]
    y_test = y[test_idx]
    pre_label,acc = predict(X[test_idx],y_test,opt_result[0])
    accuracy.append(acc)
loo_acc = np.mean(accuracy)
