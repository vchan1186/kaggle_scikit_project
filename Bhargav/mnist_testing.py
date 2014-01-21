import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import neuralnet as nn

# Parameters
param = {
'decay':0.01,
'nIter':1000,
'alpha':0.9,
'lrate':0.35,
'nHid': 50,
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

# MNIST data
data = scipy.io.loadmat('training_data.mat')

# Training, Validation and Testing data

Xtr = data['training_inputs']
ytr = data['training_targets']

Xval = data['validation_inputs']
yval = data['validation_targets']

Xte = data['test_inputs']
yte = data['test_targets']

d = Xtr.shape[0]
k = yte.shape[0]

# Vary number of hidden units
# key = 'nHid'
# keyRange = range(10,150,10)
# bidx = plot_validation_curves(key,keyRange)

# Vary the learning rate
key = 'lrate'
keyRange = np.arange(0.1,1.5,0.1)
bidx = plot_validation_curves(key,keyRange)

# Report error
param[key] = keyRange[bidx]
nnet = nn.nnet(d,k,param)
nnet.train(Xtr,ytr,Xval,yval)
mce = nnet.predict(Xte,yte)
print "Misclassification error: ",mce