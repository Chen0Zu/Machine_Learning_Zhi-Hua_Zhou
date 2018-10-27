import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def knn(X,Y,x):
    dist_mat = cdist(x,X)
    idx = np.argmax(dist_mat, axis = 1)
    pred_label = Y[idx]
    return pred_label

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
x = np.random.multivariate_normal(mu1,sigma, 10)
pred_label = knn(X,Y,x)