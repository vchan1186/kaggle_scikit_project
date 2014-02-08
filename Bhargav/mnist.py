import numpy as np
import scipy as sp
import scipy.io
import neuralnet as nn

# Parameters
param = {
'decay':0.0,
'nIter':1000,
'alpha':0.9,
'lrate':0.7,
'nHid': 100,
'adaptive':True,
'batchSize':1000,
'earlyStop':False,
'update':'improved_momentum'
}

# MNIST data
data = scipy.io.loadmat('training_data.mat')

# Training...
Xtr = data['training_inputs']
ytr = data['training_targets']

# Validation..
Xval = data['validation_inputs']
yval = data['validation_targets']

#...and Test sets
Xte = data['test_inputs']
yte = data['test_targets']

print "Initializing Neural Net..."
d = Xtr.shape[0]
k = yte.shape[0]
nnet = nn.nnet(d,k,param)
nnet.initialize_weights()

print "Training..."
nnet.train(Xtr,ytr,Xval,yval)

print "Testing..."
predTe,mceTe = nnet.predict(Xte,yte)
predTr,mceTr = nnet.predict(Xtr,ytr)
print "Summary of Results:"
print "Misclassification error on test set: ",mceTe
print "Misclassification error on training set: ",mceTr