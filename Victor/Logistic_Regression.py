import pandas as pd
import numpy as np
import scipy.optimize as sp

# File containing Logistic Regression
# A few things to keep in mind are:
# 1. n by 1 2D numpy array behaves differently from 1D array
def optimize(theta, X, y):
	""" Performs optimization using scipy fmin_cg for conjugate gradient """
	""" Outputs 1D theta_min numpy array of size number of features + 1 """
	theta_min = sp.optimize.fmin_cg(cost, theta, grad, (X,y))
	return theta_min

def sigmoid(z):
	""" Calculates sigmoid function from input. """ 
 	return 1.0/(1.0+np.exp(-z))

def cost(theta, X, y):
	""" Calculates the cost function for Logistic Regression. """
	""" Returns a float J. """
	m = X.shape[0]
	theta = np.reshape(theta,(theta.size,1))
	h = sigmoid(np.dot(X,theta))
	J=1.0/m*(-np.dot(y.T,np.log(h)) - np.dot((1.0-y).T,np.log(1.0-h)))
	return J[0,0]

def grad(theta, X, y): 
	""" Calculates gradient for Logistic Regression. """
	""" Outputs 1D grad numpy array of size feature size + 1 """
	m = X.shape[0]
	theta = np.reshape(theta,(theta.size,1))
	h = sigmoid(np.dot(X,theta)) 
	grad = np.dot(X.T,1.0/m*(h-y))
	grad = np.reshape(grad,grad.shape[0])
	return grad

