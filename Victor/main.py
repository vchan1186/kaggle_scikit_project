# import useful libraries
# ------------------------
# 1. pandas: contains two main data structures. the Series and DataFrame.
import pandas as pd 
# 2. numpy: 
import numpy as np
# 3. scipy:
from scipy import optimize

import f2
# def sigmoid(z):
#     return 1.0/(1.0+np.exp(-z))

# def cost(X, theta, y):
#     print("X1 size is = ",X.shape)
#     print("theta1 size is = ", theta.shape)
#     h = sigmoid(np.dot(X,theta))
    
#     # Calculate cost function
#     # y.T takes transpose of numpy array y.
#     J=1/m*(-np.dot(y.T,np.log(h)) - np.dot((1.0-y).T,np.log(1.0-h)))
#     return J[0,0]
# def grad(X, theta, y): 
#     print("X size is = ",X.shape)
#     print("theta size is = ", theta.shape)
#     h = sigmoid(np.dot(X,theta))  
#     grad = np.dot(X.T,1.0/m*(h-y))
#     return grad

# ---------------------------------------------------
# Read In Data
# ---------------------------------------------------
# Read in data using pandas' read_csv function. The read_csv function reads data in as 
# a DataFrame structure.

# tr_data[i] will select data in the ith column.
# tr_data.ix[i] will select data in the ith row.
# The extracted data from tr_data[i] and tr_data.ix[i] are in Series data structure.
# use type(a) function to inquire about type of variable a.
# header is set to None because there is no header in the dataset.
trn_data = pd.read_csv('train.csv',header=None)
trn_label = pd.read_csv('trainLabels.csv',header=None)

# Convert data frame into numpy arrays
X = trn_data.values
y = trn_label.values

# trn_data.index and tr_data.columns list the indices for rows and columns of a data frame, 
# respectively. 
# trn_data.shape gives the dimension of the tr_data: # of training sets (m) by # of features (n).
m = X.shape[0]
n = X.shape[1]

print "the number of features is ",n
print "the number of training sets is ", m

# bias
bias = np.zeros((m,1))
bias[:] = 1

# Initialize learning parameter array with zeros.
# initial theta is n+1 x 1 array.
initial_theta = np.zeros((n+1,1))

# Pad X with ones: hstack = "horizontal stack" (column wise)
#                  vstack = "vertical stack" (row wise)
#                  dstack = "depth stack" (depth wise)
# X is now m x n+1 array
X=np.hstack((bias,X))

cost,grad = f2.cost_function(initial_theta, X, y)

# res1 = optimize.fmin_cg(cost(X,initial_theta,y), initial_theta, fprime=grad(X,initial_theta,y))
# ---------------------------------------------------
# Calculate Cost Functions and Gradients
# ---------------------------------------------------
# Calculate sigmoid of the first training set. 
# The .values of a Series data structure produces a numpy array.
# h is m x 1 array
# h = sgmd.sigmoid(np.dot(X,initial_theta));
# 
# # y.T takes transpose of numpy array y.
# J=1/m*(-np.dot(y.T,np.log(h)) - np.dot((1-y).T,np.log(1-h)))
# 
# print(J)