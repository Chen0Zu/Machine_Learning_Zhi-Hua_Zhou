import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)

def LDA(X,y):
    y = y.flatten()
    X1 = X[y==0,:]
    X2 = X[y==1,:]
    
    mu1 = np.mean(X1,0,keepdims=True)
    mu2 = np.mean(X2,0,keepdims=True)
    sigma1 = (X1-mu1).T @ (X1-mu1)
    sigma2 = (X2-mu2).T @ (X2-mu2)
    
    Sw = sigma1 + sigma2
    
    w = np.linalg.inv(Sw) @ (mu1-mu2).T
    
    return w

# generate data
mean1 = [-2, -2]
mean2 = [2, 2]
sigma = np.eye(2)
n = 100
x1 = np.random.multivariate_normal(mean1, sigma, int(n/2))
x2 = np.random.multivariate_normal(mean2, sigma, int(n/2))
X = np.r_[x1,x2]
y = np.r_[np.zeros(int(n/2)), np.ones(int(n/2))]
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y.reshape(-1))

# learn model
w = LDA(X,y)

# show projection line
x_pos = np.linspace(-4,4)
y_pos = -1*x_pos/(-w[0]/w[1])
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y.reshape(-1))
plt.plot(x_pos,y_pos)

# =============================================================================
watermelon = pd.read_csv('西瓜数据集3.0α.csv')
X = watermelon.iloc[:,0:2].values
n = X.shape[0]
label = watermelon.iloc[:,-1].values.reshape(-1,1)
# learn model
w = LDA(X,label)

x_pos = np.linspace(0,0.8)
y_pos = -1*x_pos/(-w[0]/w[1])
plt.figure()
plt.scatter(X[:,0],X[:,1],c = label.flatten())
plt.plot(x_pos,y_pos)