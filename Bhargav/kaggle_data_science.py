import numpy as np
import dataproc as dp
import neuralnet as nn

# Parameters
param = {
'decay':0.0,
'nIter':1000,
'alpha':0.9,
'lrate':0.7,
'nHid': 30,
'batchSize':700,
'earlyStop':True,
'update':'improved momentum'
}

# Files
trainData = '/home/avasbr/Desktop/nnet/train.csv'
trainTargets = '/home/avasbr/Desktop/nnet/trainLabels.csv'
testData = '/home/avasbr/Desktop/nnet/test.csv'

# Read in the data
print "Reading training and testing data..."
X = dp.read_csv_file(trainData).T
X = dp.normalize_range(X)
tY = dp.read_csv_file(trainTargets)
Y = np.zeros([2,tY.size])
for idx,y in enumerate(Y.T):
	y[tY[idx]] = 1

# Split data into training and validation sets
trIdx, valIdx = dp.split_train_validation(X,0.7)
Xtr = X[:,trIdx]
Ytr = Y[:,trIdx]
Xval = X[:,valIdx]
Yval = Y[:,valIdx]

# Testing 
Xte = dp.read_csv_file(testData).T

print "Training and Validation Phase:"
print "------------------------------"
print "Number of training examples: ",np.shape(Xtr)[1]
print "Number of validation examples: ",np.shape(Xval)[1]
print "Input dimension: ",np.shape(Xtr)[0]
print "Output dimension: ",np.shape(Ytr)[0]

# Train a neural network
print "Training neural network..."
d = np.shape(Xtr)[0]
k = np.shape(Ytr)[0]

nnet = nn.nnet(d,k,param) 
nnet.initialize_weights()
nnet.train(Xtr,Ytr,Xval,Yval)

mce_tr = nnet.predict(Xtr,Ytr)
mce_val = nnet.predict(Xval,Yval)
print "Training error: ",mce_tr
print "Testing error: ",mce_val