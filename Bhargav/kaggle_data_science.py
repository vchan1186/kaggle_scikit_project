import numpy as np
import dataproc as dp
import neuralnet as nn

# Parameters
param = {
'decay':0.0,
'nIter':2000,
'alpha':0.9,
'lrate':0.7,
'adaptive':True,
'nHid': 20,
'batchSize':600,
'earlyStop':True,
'update':'improved_momentum'
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
idx = dp.split_train_validation_test(X,[0.6,0.2,0.2])
Xtr = X[:,idx[0]]
Ytr = Y[:,idx[0]]
Xval = X[:,idx[1]]
Yval = Y[:,idx[1]]
Xte = X[:,idx[2]]
Yte = Y[:,idx[2]]

# Train a neural network
print "Training neural network..."
d = np.shape(Xtr)[0]
k = np.shape(Ytr)[0]

nnet = nn.nnet(d,k,param) 
nnet.initialize_weights()
nnet.train(Xtr,Ytr,Xval,Yval)

predTr, mceTr = nnet.predict(Xtr,Ytr)
predVal, mceVal = nnet.predict(Xval,Yval)
predTe, mceTe = nnet.predict(Xte,Yte)

print "Training error: ",mceTr
print "Validation error: ",mceVal
print "Testing error: ",mceTe

# Prediction 
# Xpr = dp.read_csv_file(testData).T
# Ypr = nnet.predict(Xpr)