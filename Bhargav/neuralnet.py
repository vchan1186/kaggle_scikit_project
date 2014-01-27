import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

# Default parameters for the neural net
dParam = {
'decay':0.0,
'nIter':1000,
'alpha':0.9,
'lrate':0.35,
'adaptive':False,
'nHid': 50,
'batchSize':100,
'earlyStop':False,
'update':'momentum'
}

class nnet:

	def __init__(self,d,k,param=dParam):
		self.param = param
		self.d = d # input (layer) size
		self.k = k # output (layer) size
		self.W_in2hid = 0.1*np.random.rand(d+1,param['nHid'])	# weights from input to hidden layer (+bias), transposed
		self.W_hid2out = 0.1*np.random.rand(param['nHid']+1,k) # weights from hidden layer to output (+bias), transposed
		
		# needed for early stopping
		self.bW_in2hid = np.random.rand(d+1,param['nHid']) # stores the best W_in2hid
		self.bW_hid2out = np.random.rand(param['nHid']+1,k) # stores the best W_hid2out
		
		# needed for momentum
		self.mW_in2hid = np.zeros([d+1,param['nHid']])
		self.mW_hid2out = np.zeros([param['nHid']+1,k])

		# needed for adaptive learning
		# gain values
		self.gW_in2hid = np.ones([d+1,param['nHid']])
		self.gW_hid2out = np.ones([param['nHid']+1,k])
		# last computed derivative
		self.ldE_dW_in2hid = np.ones([d+1,param['nHid']])
		self.ldE_dW_hid2out = np.ones([param['nHid']+1,k])

		# uncomment for gradient checking
		#self.gradient = np.empty((d+1)*param['nHid'] + (param['nHid']+1)*k) # needed only for doing gradient checking

	def initialize_weights(self):
		"""Useful for customizing weights, and for testing purposes"""
		self.W_in2hid  = 0.1*np.cos(np.arange(self.W_in2hid.size).reshape(self.W_in2hid.shape))
		self.W_hid2out = 0.1*np.cos((np.arange(self.W_hid2out.size)+self.W_in2hid.size).reshape(self.W_hid2out.shape))

	def fprop(self,X,W_in2hid=None,W_hid2out=None):
		"""Perform forward propagation"""

		if W_in2hid is None and W_hid2out is None:
			W_in2hid = self.W_in2hid
			W_hid2out = self.W_hid2out

		N = np.shape(X)[1] # number of training cases in this batch of data
		hidAct = self.logit(np.dot(W_in2hid.T,X)) # activation of the hidden layer
		hidAct = np.append(np.ones([1,N]),hidAct,axis=0) # needed for bias
		outAct = self.softmax(np.dot(W_hid2out.T,hidAct)) # output activation		
		
		return hidAct, outAct

	def bprop(self,X,y,hidAct,outAct,W_in2hid=None,W_hid2out=None):
		"""Performs backpropagation"""

		if W_in2hid is None and W_hid2out is None:
			W_in2hid = self.W_in2hid
			W_hid2out = self.W_hid2out

		N = np.shape(X)[1]
		dE_dzo = outAct-y # derivative of the cost-function with respect to the logit into the output layer
		dE_dW_hid2out = 1.0/N*np.dot(hidAct,dE_dzo.T) + self.param['decay']*W_hid2out
		dE_dyh = np.dot(W_hid2out,dE_dzo)
		dE_dzh = (dE_dyh*hidAct*(1-hidAct))[1:,:]
		dE_dW_in2hid = 1.0/N*np.dot(X,dE_dzh.T)+self.param['decay']*W_in2hid
		
		return dE_dW_in2hid, dE_dW_hid2out

	def train(self,Xtr,ytr,Xval=None,yval=None):
		""" Performs repeated fprop+bprop with an update method to train a 2-layer feed-forward neural network """
		
		nTr = np.shape(Xtr)[1]
		Xtr = np.append(np.ones([1,nTr]),Xtr,axis=0)
		
		if Xval is not None and yval is not None:		
			nVal = np.shape(Xval)[1]
			Xval = np.append(np.ones([1,nVal]),Xval,axis=0)
			valLoss = []
			if self.param['earlyStop']:
				bvalLoss = float("inf")

		bidx = -1
		trLoss = []
		for i in range(self.param['nIter']):

			idx = np.random.permutation(nTr)[:self.param['batchSize']] # mini-batch indices	
			
			if self.param['update']=='improved momentum':
				# take a step first in the direction of the accumulated gradient
				self.W_in2hid += self.mW_in2hid
				self.W_hid2out += self.mW_hid2out

			# fprop,bprop
			hidAct, outAct = self.fprop(Xtr[:,idx])
			dE_dW_in2hid, dE_dW_hid2out = self.bprop(Xtr[:,idx],ytr[:,idx],hidAct,outAct)

			# uncomment for gradient checking
			#self.gradient = self.unroll(dE_dW_in2hid, dE_dW_hid2out)
			#self.check_gradients(Xtr[:,idx],ytr[:,idx])

			# update weights
			self.update_weights(dE_dW_in2hid,dE_dW_hid2out)

			# keep track of the last-used gradient if we are doing adaptive learning
			self.ldE_dW_in2hid = dE_dW_in2hid
			self.ldE_dW_hid2out = dE_dW_hid2out

			# uncomment to compute training and (if val-set provided) validation loss
			hidAct, outAct = self.fprop(Xtr)
			trLoss.append(self.compute_loss(outAct,ytr))

			if Xval is not None and yval is not None:
				hidAct, outAct = self.fprop(Xval)
				thisLoss = self.compute_loss(outAct, yval)
				valLoss.append(thisLoss)
				
				# keep track of the best weights so far 
				if self.param['earlyStop'] and thisLoss < bvalLoss:
					bvalLoss = thisLoss
					bidx = i
					self.bW_in2hid = np.copy(self.W_in2hid)
					self.bW_hid2out = np.copy(self.W_hid2out)

		# set the weights to the best weights
		if self.param['earlyStop']:
			self.W_in2hid = self.bW_in2hid
			self.W_hid2out = self.bW_hid2out

		print "Final (or best) Training Cross-Entropy Error: ",trLoss[bidx]
		
		if Xval is not None and yval is not None:
			print "Final (or best) Validation Cross-Entropy Error: ",valLoss[bidx]
			self.plot_curves(trLoss,valLoss)
			return trLoss[bidx], valLoss[bidx]
		else:
			self.plot_curves(trLoss)
			return trLoss[bidx]

	def plot_curves(self,trLoss,valLoss=None):
		""" Plot training and (optional) validation loss """
	
		plt.plot(range(self.param['nIter']),trLoss,label='Training Loss')
		if valLoss is not None:
			plt.plot(range(self.param['nIter']),valLoss,label='Validation loss')

		plt.xlabel("Iteration")
		plt.ylabel("Cross-entropy Loss")
		plt.title("Cross entropy loss per Iteration")
		plt.legend(loc='upper right')
		plt.show()

	def update_weights(self,dE_dW_in2hid,dE_dW_hid2out):
		
		# decide if we are in a fixed-rate or adaptive learning rate regime
		if self.param['adaptive']:
			# check for the agreement of signs
			sW_in2hid = self.ldE_dW_in2hid*dE_dW_in2hid
			sW_hid2out = self.ldE_dW_hid2out*dE_dW_hid2out
			self.gW_in2hid += (sW_in2hid>0)*0.05
			self.gW_in2hid *= (sW_in2hid<0)*0.95
			self.gW_hid2out += (sW_hid2out>0)*0.05
			self.gW_hid2out *= (sW_hid2out<0)*0.95
			# keep the learning rates clamped
			self.gW_in2hid = self.clamp(self.gW_in2hid,0.1,10)
			self.gW_hid2out = self.clamp(self.gW_hid2out,0.1,10)

		# simple gradient-descent
		if self.param['update']=='default':
			self.W_in2hid -= self.param['lrate']*self.gW_in2hid*dE_dW_in2hid
			self.W_hid2out -= self.param['lrate']*self.gW_hid2out*dE_dW_hid2out
		
		# momentum
		elif self.param['update']=='momentum':
			self.mW_in2hid = self.param['alpha']*self.mW_in2hid + dE_dW_in2hid
			self.mW_hid2out = self.param['alpha']*self.mW_hid2out + dE_dW_hid2out
			self.W_in2hid -= self.param['lrate']*self.gW_in2hid*self.mW_in2hid
			self.W_hid2out -= self.param['lrate']*self.gW_hid2out*self.mW_hid2out
		
		# improved momentum
		elif self.param['update']=='improved momentum':
			# same as 'default' 
			self.W_in2hid -= self.param['lrate']*self.gW_in2hid*dE_dW_in2hid
			self.W_hid2out -= self.param['lrate']*self.gW_hid2out*dE_dW_hid2out
			# update the momentum terms
			self.mW_in2hid = self.param['alpha']*(self.mW_in2hid - self.param['lrate']*self.gW_in2hid*dE_dW_in2hid)
			self.mW_hid2out = self.param['alpha']*(self.mW_hid2out - self.param['lrate']*self.gW_hid2out*dE_dW_hid2out)

		# rmsprop

	def predict(self,X,y=None):
		"""Uses fprop for predicting labels of data"""

		n = np.shape(X)[1]
		X = np.append(np.ones([1,n]),X,axis=0)
		hidAct, outAct = self.fprop(X,self.W_in2hid,self.W_hid2out)
		if y==None:
			return np.argmax(outAct,axis=0)
		return 1.0-np.mean(1.0*(np.argmax(outAct,axis=0)==np.argmax(y,axis=0)))

	def compute_mce(self,pr,te):
		" Computes the misclassification error"
		return 1.0-np.mean(1.0*(pr==te))

	def logit(self,z):
		"""Computes the element-wise logit of z"""
		
		return 1./(1. + np.exp(-1.*z))

	def softmax(self,z):
		""" Computes the softmax of the outputs in a numerically stable manner"""
		
		maxV = np.max(z,axis=0)
		logSum = np.log(np.sum(np.exp(z-maxV),axis=0))+maxV
		return np.exp(z-logSum)

	def compute_class_loss(self,outAct,y):
		"""Computes the cross-entropy classification loss of the model (without weight decay)"""
		
		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer		
		return np.mean(np.sum(-1.0*y*np.log(outAct),axis=0))

	def compute_loss(self,outAct,y,W_in2hid=None, W_hid2out=None):
		"""Computes the cross entropy classification (with weight decay)"""
		
		if W_in2hid is None and W_hid2out is None:
			W_in2hid = self.W_in2hid
			W_hid2out = self.W_hid2out
		
		return self.compute_class_loss(outAct,y) + 0.5*self.param['decay']*(np.sum(W_in2hid**2)+np.sum(W_hid2out**2))

	def unroll(self,_in2hid,_hid2out):
		"""Flattens matrices and concatenates to a vector """

		return np.append(_in2hid.flatten(),_hid2out.flatten())

	def reroll(self,v):
		"""Re-rolls a vector of weights into the in2hid- and hid2out-sized weight matrices """
		
		_in2hid = np.reshape(v[:self.W_in2hid.size],self.W_in2hid.shape)
		_hid2out = np.reshape(v[self.W_in2hid.size:],self.W_hid2out.shape)
		
		return _in2hid,_hid2out

	def clamp(self,a,minv,maxv):
		""" imposes a range on all values of a matrix """
		return np.fmax(minv,np.fmin(maxv,a))

	def check_gradients(self,X,y):
		"""Computes a finite difference approximation of the gradient to check the correction of 
		the backpropagation algorithm"""
	
		N = np.shape(X)[1] # number of training cases in this batch of data
		
		err_tol = 1e-8	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = self.W_in2hid.size + self.W_hid2out.size
 		idx = np.random.permutation(n)[:(n/5)] # choose a random 20% 
 		apprxDerv = [None]*len(idx)

		for i,x in enumerate(idx):
			
			W_plus = self.unroll(self.W_in2hid,self.W_hid2out)
			W_minus = self.unroll(self.W_in2hid,self.W_hid2out)
			
			# Perturb one of the weights by eps
			W_plus[x] += eps
			W_minus[x] -= eps
			W_in2hid_plus, W_hid2out_plus = self.reroll(W_plus)
			W_in2hid_minus, W_hid2out_minus = self.reroll(W_minus)

			# run fprop and compute the loss for both sides  
			hidAct,outAct = self.fprop(X,W_in2hid_plus,W_hid2out_plus)
			lossPlus = self.compute_loss(outAct, y, W_in2hid_plus, W_hid2out_plus)
			hidAct,outAct = self.fprop(X,W_in2hid_minus,W_hid2out_minus)
			lossMinus = self.compute_loss(outAct, y, W_in2hid_minus, W_hid2out_minus)
			
			apprxDerv[i] = 1.0*(lossPlus-lossMinus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(apprxDerv-self.gradient[idx]))
		if(cerr>=err_tol):
			print 'Mean computed error ',cerr,' is larger than the error tolerance -- there is probably an error in the computation'
		else:
			print 'Mean computed error ',cerr,' is smaller than the error tolerance -- the computation was probably correct'

	# The next three functions are for doing batch-optimization using routines from 
	# scipy

	def compute_gradient(self,w,X,y):
		""" Computation of the gradient """
		W_in2hid,W_hid2out = self.reroll(w)
		hidAct,outAct = self.fprop(X,W_in2hid,W_hid2out)
		dE_dW_in2hid, dE_dW_hid2out = self.bprop(X,y,hidAct,outAct,W_in2hid,W_hid2out)
		return self.unroll(dE_dW_in2hid,dE_dW_hid2out)

	def compute_cost(self,w,X,y):
		W_in2hid,W_hid2out = self.reroll(w)
		hidAct,outAct = self.fprop(X,W_in2hid,W_hid2out)
		return self.compute_loss(outAct,y,W_in2hid,W_hid2out)

	def optimize(self,Xtr,ytr):
		""" uses scipy optimization routines to minimize the cost function """
		nTr = np.shape(Xtr)[1]
		Xtr = np.append(np.ones([1,nTr]),Xtr,axis=0)
		w0 = self.unroll(self.W_in2hid, self.W_hid2out)
		wf = fmin_cg(self.compute_cost,w0,self.compute_gradient,(Xtr,ytr))
		W_in2hid,W_hid2out = self.reroll(wf)
		self.W_in2hid = W_in2hid
		self.W_hid2out = W_hid2out


