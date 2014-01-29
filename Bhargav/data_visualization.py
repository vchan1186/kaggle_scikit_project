import numpy as np
import dataproc as dp
from sklearn.decomposition import PCA,KernelPCA
import matplotlib.pyplot as plt

# Prep the data

# Files
trainData = '/home/avasbr/Desktop/kaggle_scikit_project/Bhargav/train.csv'
trainTargets = '/home/avasbr/Desktop/kaggle_scikit_project/Bhargav/trainLabels.csv'
testData = '/home/avasbr/Desktop/kaggle_scikit_project/Bhargav/test.csv'

print "Reading training and testing data..."
X = dp.read_csv_file(trainData)
Y = dp.read_csv_file(trainTargets)
X = dp.normalize_range(X) # do a little feature scaling

# PCA
kpca = KernelPCA(n_components=2,kernel='rbf')
Xr = kpca.fit_transform(X)
plt.scatter(Xr[Y==0,0],Xr[Y==0,1],color='red')
plt.scatter(Xr[Y==1,0],Xr[Y==1,1],color='blue')
plt.show()

