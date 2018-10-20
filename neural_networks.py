import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a

def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros([n_y,1])
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1,X.shape[1]))
    
    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    
    return A2, cache

def compute_cost(A2, Y):
    
    m = Y.shape[1]
    logprobs = Y*np.log(A2) + (1-Y)*np.log(1-A2)
    cost = -1/m*np.sum(logprobs)
    
    cost = np.squeeze(cost)
    
    assert(isinstance(cost, float))
    
    return cost

def back_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = 1/m*(dZ2 @ A1.T)
    db2 = 1/m*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T@dZ2*(A2*(1-A2))
    dW1 = 1/m*dZ1@X.T
    db1 = 1/m*np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {'dW1':dW1,
             'db1':db1,
             'dW2':dW2,
             'db2':db2
            }
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    
mu1 = np.array([-2,-2])
mu2 = np.array([2,2])
sigma = 2*np.eye(2)
n = 400
X1 = np.random.multivariate_normal(mu1, sigma, int(n/2))
X2 = np.random.multivariate_normal(mu2, sigma, int(n/2))
X = np.r_[X1,X2]
y = np.r_[np.ones([int(n/2),1]),np.zeros([int(n/2),1])]
plt.figure()
plt.scatter(X[:,0], X[:,1], c = y.flatten())

parameters = initialize_parameters(2,4,1)

A2, cache = forward_propagation(X.T, parameters)
cost = compute_cost(A2, y.T)
grads = back_propagation(parameters, cache, X.T, y.T)