import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

mu1 = np.array([-2,-2])
mu2 = np.array([2,2])
sigma = np.eye(2)
n = 400
X1 = np.random.multivariate_normal(mu1,sigma,int(n/2))
X2 = np.random.multivariate_normal(mu2,sigma,int(n/2))
X = np.r_[X1,X2]
Y = np.r_[np.ones(int(n/2)),-np.ones(int(n/2))].reshape(-1,1)

clf = svm.SVC(kernel='linear')
clf.fit(X,Y.reshape(-1))

plt.figure()
plt.scatter(X[:,0],X[:,1],c=Y.flatten())
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100)