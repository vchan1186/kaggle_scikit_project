import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

class MultiLayerNet:

	def __init__(self,n_hid=[50,25],decay=0.0,alpha=0.9,learn_rate=0.35,adaptive='False',
		batch_size=100,n_layers=2,update='improved_momentum'):
		
		self.decay = decay
		self.alpha = alpha
		self.learn_rate = learn_rate
		self.adaptive = adaptive
		self.batch_size = batch_size
		self.early_stop = early_stop
		self.update = update

		return self

	def fit(self,X,y,n_iter=1000):

		# the matrix multiplications just look prettier this way...
		X = X.T
		y = y.T 

		d = X.shape[0] # input (layer) size
		k = y.shape[0] # output (layer) size
		m = y.shape[1] # number of instances

		# initialize weights randomly
		n_nodes = d+self.n_hid+k
		self.weights = []:
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:])
			self.weights.append(0.01*np.random.rand(n1+1,n2))
		
		accum_grad = []
		# needed for momentum, improved_momentum
		if self.update=='momentum' or self.update=='improved_momentum':
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:])
				accum_grad.append(np.zeros([n1+1,n2]))

		# needed for adaptive learning
		gain = []
		last_grad = []
		if self.adaptive:
			# local gain terms
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:])
			gain.append(np.ones([n1+1,n2]))
			# gradient values from previous iteration
			last_grad,append(np.ones([n1+1,n2]))

		# uncomment for gradient checking
		# gradient = np.empty((d+1)*self.n_hid + (self.n_hid+1)*k) # needed only for doing gradient checking

		# uses the scipy routine for conjugate gradient
		if self.update == 'conjugate_gradient':
			w0 = self.unroll(self.W_in2hid_,self.W_hid2out_)
			wf = fmin_cg(self.compute_cost,w0,self.compute_gradient,(X,y))
			W_in2hid,W_hid2out = self.reroll(wf)
			self.W_in2hid_ = W_in2hid
			self.W_hid2out_ = W_hid2out
		else:
			for i in range(n_iter):

				idx = np.random.permutation(n)[:self.batch_size] # mini-batch indices	
				
				if self.update=='improved_momentum':
					# take a step first in the direction of the accumulated gradient
					self.weights = map(sum,zip(self.weights,accum_grad))

				# propagate the data 
				hidAct, outAct = self.fprop(X[:,idx])
				dE_dW_in2hid, dE_dW_hid2out = self.bprop(X[:,idx],y[:,idx],hidAct,outAct)

				# uncomment for gradient checking
				# self.gradient = self.unroll(dE_dW_in2hid, dE_dW_hid2out)
				# self.check_gradients(X[:,idx],y[:,idx])

				if self.adaptive:
					# check for the agreement of signs
					sW_in2hid = ldE_dW_in2hid*dE_dW_in2hid
					sW_hid2out = ldE_dW_hid2out*dE_dW_hid2out
					
					# same sign --> increase learning rate, opposite --> decrease 
					np.putmask(gW_in2hid, sW_in2hid<0, gW_in2hid*0.95)
					np.putmask(gW_in2hid, sW_in2hid>0, gW_in2hid+0.05)
					np.putmask(gW_hid2out, sW_hid2out<0, gW_in2hid*0.95)
					np.putmask(gW_hid2out, sW_hid2out>0, gW_hid2out+0.05)

					# keep the learning rates clamped
					gW_in2hid = clamp(gW_in2hid,0.1,10)
					gW_hid2out = clamp(gW_hid2out,0.1,10)

				# simple gradient-descent
				if self.update=='default':
					self.W_in2hid_ -= self.learn_rate*gW_in2hid*dE_dW_in2hid
					self.W_hid2out_ -= self.learn_rate*gW_hid2out*dE_dW_hid2out
				
				# momentum
				elif self.update=='momentum':
					mW_in2hid = self.alpha*mW_in2hid + dE_dW_in2hid
					mW_hid2out = self.alpha*mW_hid2out + dE_dW_hid2out
					self.W_in2hid_ -= self.learn_rate*gW_in2hid*mW_in2hid
					self.W_hid2out_ -= self.learn_rate*gW_hid2out*mW_hid2out
				
				# improved momentum
				elif self.update=='improved_momentum':
					# same as 'default' 
					self.W_in2hid_ -= self.learn_rate*gW_in2hid*dE_dW_in2hid
					self.W_hid2out_ -= self.learn_rate*gW_hid2out*dE_dW_hid2out
					# update the momentum terms
					mW_in2hid = self.alpha*(mW_in2hid - self.learn_rate*gW_in2hid*dE_dW_in2hid)
					mW_hid2out = self.alpha*(mW_hid2out - self.learn_rate*gW_hid2out*dE_dW_hid2out)

		return self

	def fprop(self,X,weights=None):
		"""Perform forward propagation"""

		if weights==None:
			weights = self.weights

		N = X.shape[1] # number of training cases in this batch of data
		act = [np.append(np.ones([1,N]),self.logit(np.dot(weights[0].T,X),axis=0)]
		for i,W in enumerate(weights[1:-1]):
			act.append(np.append(np.ones([1,N]),self.logit(np.dot(W.T,act[i])),axis=0)) # sigmoid activations
		act.append(self.softmax(np.dot(weights[-1].T,act[-1]))) # output of the last layer is a softmax
		
		return act

	def bprop(self,X,y,act,weights=None):
		"""Performs backpropagation"""

		if weights==None:
			weights = self.weights

		N = np.shape(X)[1]
		weight_derivs = []
		
		dE_dzo = outAct-y # derivative of the cost-function with respect to the logit into the output layer
		dE_dW_hid2out = 1.0/N*np.dot(hidAct,dE_dzo.T) + self.param['decay']*W_hid2out
		dE_dyh = np.dot(W_hid2out,dE_dzo)
		dE_dzh = (dE_dyh*hidAct*(1-hidAct))[1:,:]
		dE_dW_in2hid = 1.0/N*np.dot(X,dE_dzh.T)+self.param['decay']*W_in2hid
		
	def predict(self,X,y=None):
		"""Uses fprop for predicting labels of data. If labels are also provided, also returns mce """

		n = np.shape(X)[1]
		X = np.append(np.ones([1,n]),X,axis=0)
		hidAct, outAct = self.fprop(X,self.W_in2hid,self.W_hid2out)
		pred = np.argmax(outAct,axis=0)
		if y==None:
			return pred
		mce = 1.0-np.mean(1.0*(np.argmax(outAct,axis=0)==np.argmax(y,axis=0)))
		return pred,mce

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

	# The following are convenience functions for doing batch-optimization using routines from 
	# scipy (e.g, fmin_cg)

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