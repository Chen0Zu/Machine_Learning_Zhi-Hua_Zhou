import numpy as np
#import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def kmeans(X):
    n,d = X.shape
    c = 3
    means = X[np.random.choice(n,c)]
    iteration = 100
    
    for i in range(iteration):
        dist_mat = cdist(X,means)**2
        idx = np.argsort(dist_mat, axis = 1)
        label = idx[:,0]
        for j in range(c):
            means[j,:] = np.mean(X[label == j,:], axis = 0, keepdims = True)
    Y = label
    
    return Y

# generate data
mu1 = np.array([2,2])
mu2 = np.array([1,1])
mu3 = np.array([0,0])
sigma = 0.1*np.eye(2)
n = 90
X1 = np.random.multivariate_normal(mu1,sigma, int(n/3))
X2 = np.random.multivariate_normal(mu2,sigma, int(n/3))
X3 = np.random.multivariate_normal(mu3, sigma, int(n/3))
X = np.r_[X1,X2,X3]
Y = np.r_[np.zeros([int(n/3),1]), np.ones([int(n/3),1]), 2*np.ones([int(n/3),1])].reshape(-1,1)
plt.figure()
plt.scatter(X[:,0], X[:,1], c = Y.flatten())

# =============================================================================
cluster = kmeans(X)
plt.figure()
plt.scatter(X[:,0], X[:,1], c = cluster.flatten())