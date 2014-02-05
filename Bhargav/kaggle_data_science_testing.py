import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import neuralnet as nn
import dataproc as dp

# Parameters
param = {
'decay':0.0,
'nIter':1000,
'alpha':0.85,
'lrate':0.6,
'nHid': 25,
'adaptive': True,
'batchSize':600,
'earlyStop':True,
'update':'imp1roved_momentum'
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
		tl,vl = nnet.train(Xtr,Ytr,Xval,Yval)
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

# Load the data
trainData = '/home/avasbr/Desktop/nnet/train.csv'
trainTargets = '/home/avasbr/Desktop/nnet/trainLabels.csv'
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

d = Xtr.shape[0]
k = Yte.shape[0]

# Vary number of hidden units
key = 'nHid'
keyRange = range(5,35,5)
bidx = plot_validation_curves(key,keyRange)

# Vary the learning rate
# key = 'lrate'
# keyRange = np.arange(0.1,1.5,0.1)
# bidx = plot_validation_curves(key,keyRange)

# Vary the momentum term
# key = 'alpha'
# keyRange = np.arange(0.1,1.0,0.1)
# bidx = plot_validation_curves(key,keyRange)

# Report error
param[key] = keyRange[bidx]
nnet = nn.nnet(d,k,param)
nnet.train(Xtr,Ytr,Xval,Yval)
pred,mce = nnet.predict(Xte,Yte)
print "Misclassification error: ",mce