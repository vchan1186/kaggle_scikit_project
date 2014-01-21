# function to calculate the cost function and grad
# input and output data structure function is np.array.

# import numpy library for exponent function
import numpy as np
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def cost_function(theta, X, y):
    
    h = sigmoid(np.dot(X,theta))
    
    # Calculate cost function
    # y.T takes transpose of numpy array y.
    m = X.shape[0] # number of training sets.
    J=1.0/m*(-np.dot(y.T,np.log(h)) - np.dot((1.0-y).T,np.log(1.0-h)))
    
    grad = np.dot(X.T,1.0/m*(h-y))
    
    return J, grad
