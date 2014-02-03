import pandas as pd
import numpy as np
import Logistic_Regression as LR

# Read in Data
trn_data = pd.read_csv('train.csv',header=None)
trn_label = pd.read_csv('trainLabels.csv',header=None)

# Break Data into three sets
# training (60% = 600): Xtr, ytr
# cross-validation (20% = 200): Xcv, ycv
# test (20% = 200): Xtst, ytst
X = trn_data.values
y = trn_label.values

# Perform Feature Scaling X = X-averag(X)/range of X.
X = (X-np.mean(X))/(np.max(X)-np.min(X))

Xtr = X[0:600][:]
ytr = y[0:600][:]

Xcv = X[600:800][:]
ycv = y[600:800][:]

Xtst = X[800:1000][:]
ytst = y[800:1000][:]

# Train parameters using training set
# m = number of examples in training set.
# n = number of features in training set.
m = Xtr.shape[0]
n = Xtr.shape[1]

# Generate bias
bias = np.zeros((m,1))
bias[:] = 1

# Initialize learning parameters, theta
theta = np.zeros(n+1)

# Append bias to training set
Xtr = np.hstack((bias,Xtr))

# Perform Logistic Regression using Logistic Regression Module.
theta_min = LR.optimize(theta,Xtr,ytr)

# Calculate errors
theta_min = np.reshape(theta_min,(theta_min.size,1))

# Training
htr = LR.sigmoid(np.dot(Xtr,theta_min))
Jtr = 1.0/(2.0*m)*(np.dot((htr-ytr).T,htr-ytr))

# Cross-validation
Xcv = np.hstack((bias[0:Xcv.shape[0]],Xcv))
hcv = LR.sigmoid(np.dot(Xcv,theta_min))
Jcv = 1.0/(2.0*Xcv.shape[0])*(np.dot((hcv-ycv).T,hcv-ycv))

# Test
Xtst = np.hstack((bias[0:Xtst.shape[0]],Xtst))
htst = LR.sigmoid(np.dot(Xtst,theta_min))
Jtst = 1.0/(2.0*Xtst.shape[0])*(np.dot((htst-ytst).T,htst-ytst))

print("The training error is ", Jtr[0][0])
print("The cross-validation error is ", Jcv[0][0])
print("The test error is ", Jtst[0][0])
