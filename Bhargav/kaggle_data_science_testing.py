import numpy as np
import dataproc as dp
import neuralnet as nn

# Parameters
param = {
'decay':0.01,
'nIter':1000,
'alpha':0.9,
'lrate':0.35,
'nHid': 20,
'batchSize':100,
'earlyStop':True,
'update':'momentum'
}

def plot_validation_curves(key,keyRange):
	print "Varying parameter: ",key
	bValLoss = float('inf')
	valLoss = []
	trLoss = []
	bidx = 0
	for idx,v in enumerate(keyRange):
		print "Testing parameter = ",v
		param[key] = v
		nnet = nn.nnet(d,k,param)
		tl,vl = nnet.train(Xtr,ytr,Xval,yval)
		valLoss.append(vl)
		trLoss.append(tl)
		if vl < bValLoss:
			bValLoss = vl
			bidx = idx
	print "Best Validation Loss: ",bValLoss
	print "Best Value for parameter ",key,": ",keyRange[bidx]

	plt.plot(keyRange,valLoss,label='Validation Loss')
	plt.plot(keyRange,trLoss,label='Training Loss')
	plt.xlabel(key)
	plt.ylabel("Cross-entropy Loss")
	plt.title("Effects of NN Parameters on Cross-Entropy Loss")
	plt.legend(loc='upper right')
	plt.show()

	return bidx

# Files
trainData = '/home/avasbr/Desktop/nnet/train.csv'
trainTargets = '/home/avasbr/Desktop/nnet/trainLabels.csv'
testData = '/home/avasbr/Desktop/nnet/test.csv'

# Read in the data
print "Reading training and testing data..."
X = dp.read_csv_file(trainData).T
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

# Train a neural network
print "Training neural network..."
d = np.shape(Xtr)[0]
k = np.shape(Ytr)[0]
print 'Output size: ',k

nnet = nn.nnet(d,k,param) 
nnet.train(Xtr,Ytr,Xval,Yval)