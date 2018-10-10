import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def sigmoid(x):
    h = 1/(1+np.exp(-x))
    return h

def logistic_regression(X,y):
    learning_rate = 0.001
    n,d = X.shape
    y = y.reshape(n,1)
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
        
    return (w,J,J_h)
        

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
y = np.r_[np.zeros(int(n/2)), np.ones(int(n/2))]
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y)

# learn model
w,J,J_h = logistic_regression(X,y)
plt.figure()
plt.plot(J)
plt.figure()
plt.plot(J_h)

# show decision boundary
x_pos = np.linspace(-3,3,10)
y_pos = (-w[0]*x_pos-w[2])/w[1]
plt.figure()
plt.scatter(X[:,0], X[:,1],c=y)
plt.plot(x_pos,y_pos)

# predct
pre_label = np.sign(np.c_[X,np.ones([n,1])]@w)
pre_label[pre_label == -1] = 0
accuracy = np.mean(pre_label == y.reshape(-1,1))
print("Accuracy is", accuracy)