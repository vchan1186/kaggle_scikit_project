import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
import operator as op

class MultiLayerNet:

	def __init__(self,n_hid=[50,25],decay=0.0,alpha=0.9,learn_rate=0.35,adaptive='False',
		batch_size=100,update='improved_momentum'):
		
		self.n_hid = n_hid
		self.decay = decay
		self.alpha = alpha
		self.learn_rate = learn_rate
		self.adaptive = adaptive
		self.batch_size = batch_size
		self.update = update

	def fit(self,X,y,n_iter=1000):
		
		# the matrix multiplications just look prettier this way...
		X = X.T
		y = y.T 

		d = X.shape[0] # input (layer) size
		k = y.shape[0] # output (layer) size
		m = y.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 

		# initialize weights randomly
		n_nodes = [d]+self.n_hid+[k] # concatenate the input and output layers
		self.weights = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			self.weights.append(0.01*np.random.rand(n1+1,n2))
		
		accum_grad = []
		# needed for momentum, improved_momentum
		if self.update=='momentum' or self.update=='improved_momentum':
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
				accum_grad.append(np.zeros([n1+1,n2]))

		# needed for adaptive learning
		gain = []
		last_grad = []
		if self.adaptive:
			# local gain terms
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
				gain.append(np.ones([n1+1,n2]))
				# gradient values from previous iteration
				last_grad.append(np.ones([n1+1,n2]))
		else:
			gain = len(self.weights)*[1.0]

		# uncomment for gradient checking
		grad_vector = np.empty(sum([w.size for w in self.weights]))

		# uses the scipy routine for conjugate gradient
		if self.update == 'conjugate_gradient':
			w0 = self.unroll(self.weights)
			wf = fmin_cg(self.compute_cost,w0,self.compute_gradient,(X,y))
			weights = self.reroll(wf)
			self.weights = weights

		else:
			for i in range(n_iter):

				idx = np.random.permutation(m)[:self.batch_size] # mini-batch indices	
				
				if self.update=='improved_momentum':
					# take a step first in the direction of the accumulated gradient
					self.weights = map(sum,zip(self.weights,accum_grad))

				# propagate the data 
				act = self.fprop(X[:,idx]) # get the activations from forward propagation
				grad = self.bprop(X[:,idx],y[:,idx],act)

				# uncomment for gradient checking
				gradient = self.unroll(grad)
				self.check_gradients(X[:,idx],y[:,idx],gradient)

				if self.adaptive:
					# same sign --> increase learning rate, opposite --> decrease 
					for i,(d,l,g) in enumerate(zip(grad,last_grad,gain)):
						sign_grad = d*l
						np.putmask(g,sign_grad<0,g*0.95)
						np.putmask(g,sign_grad>0,g+0.05)
						gain[i] = self.clamp(g,0.1,10)

				# simple gradient-descent
				if self.update=='default':
					self.weights = [self.weights[i]-self.learn_rate*g*d for i,(d,g) in enumerate(zip(grad,gain))]
				
				# momentum
				elif self.update=='momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						accum_grad[i] = self.alpha*accum_grad[i] + d
						self.weights[i] -= self.learn_rate*g*accum_grad[i]
				
				# improved momentum
				elif self.update=='improved_momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						self.weights[i] -= self.learn_rate*g*d
						accum_grad[i] = self.alpha*(accum_grad[i] - self.learn_rate*g*d)
			
		return self

	def fprop(self,X,weights=None):
		"""Perform forward propagation"""

		if weights==None:
			weights = self.weights

		m = X.shape[1] # number of training cases in this batch of data
		act = [np.append(np.ones([1,m]),self.logit(np.dot(weights[0].T,X)),axis=0)] # use the first data matrix to compute the first activation
		for i,w in enumerate(weights[1:-1]):
			act.append(np.append(np.ones([1,m]),self.logit(np.dot(w.T,act[i])),axis=0)) # sigmoid activations
		act.append(self.softmax(np.dot(weights[-1].T,act[-1]))) # output of the last layer is a softmax
		
		return act

	def bprop(self,X,y,act,weights=None):
		"""Performs backpropagation"""

		# reversing the lists makes it easier to work with 
		if weights==None:
			weights = self.weights[::-1]
		act = act[::-1]

		N = X.shape[1]
		grad = []
		
		# the final layer is a softmax, so calculate this first
		grad_z = act[0]-y
		
		for i,a in enumerate(act[1:]):
			grad.append(1.0/N*np.dot(a,grad_z.T) + self.decay*weights[i])
			grad_y = np.dot(weights[i],grad_z)
			grad_z = (grad_y*a*(1-a))[1:,:] # no connection to the bias node
		
		grad.append(1.0/N*np.dot(X,grad_z.T) + self.decay*weights[-1])

		# re-reverse and return
		return grad[::-1]
		
	def predict(self,X,y=None):
		"""Uses fprop for predicting labels of data. If labels are also provided, also returns mce """

		X = X.T
		y = y.T
		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		act = self.fprop(X)
		pred = np.argmax(act[-1],axis=0) # only the final activation contains the 
		if y==None:
			return pred
		mce = 1.0-np.mean(1.0*(pred==np.argmax(y,axis=0)))
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

	def compute_class_loss(self,act,y):
		"""Computes the cross-entropy classification loss of the model (without weight decay)"""
		
		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
		#import pdb; pdb.set_trace()		
		return np.mean(np.sum(-1.0*y*np.log(act),axis=0))

	def compute_loss(self,act,y,weights=None):
		"""Computes the cross entropy classification (with weight decay)"""
		
		if weights is None:
			weights = self.weights
		return self.compute_class_loss(act,y) + 0.5*self.decay*sum([np.sum(w**2) for w in weights])

	def unroll(self,weights):
		"""Flattens matrices and concatenates to a vector """
		v = np.array([])
		for w in weights:
			v = np.concatenate((v,np.ndarray.flatten(w)))
		#reduce(np.append,(map(np.ndarray.flatten,weights))) # un-pythonic, functional programming method
		return v

	def reroll(self,v):
		"""Re-rolls a vector of weights into the in2hid- and hid2out-sized weight matrices"""

		idx = 0
		r_weights = []
		for w in self.weights:
			r_weights.append(np.reshape(v[idx:idx+w.size],w.shape))
			idx+=w.size
		
		return r_weights
		
	def clamp(self,a,minv,maxv):
		""" imposes a range on all values of a matrix """
		return np.fmax(minv,np.fmin(maxv,a))

	def check_gradients(self,X,y,gradient):
		"""Computes a finite difference approximation of the gradient to check the correction of 
		the backpropagation algorithm"""

		m = X.shape[1] # number of training cases in this batch of data
		
		err_tol = 1e-8	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in self.weights)
 		idx = np.random.permutation(n)[:(n/10)] # choose a random 10% 
 		apprxDerv = [None]*len(idx)

		for i,x in enumerate(idx):
			W_plus = self.unroll(self.weights)
			W_minus = self.unroll(self.weights)
			
			# Perturb one of the weights by eps
			W_plus[x] += eps
			W_minus[x] -= eps
			#import pdb; pdb.set_trace()
			weights_plus = self.reroll(W_plus)
			weights_minus = self.reroll(W_minus)

			# run fprop and compute the loss for both sides  
			act = self.fprop(X,weights_plus)
			lossPlus = self.compute_loss(act[-1], y, weights_plus)
			act = self.fprop(X,weights_minus)
			lossMinus = self.compute_loss(act[-1], y, weights_minus)
			
			apprxDerv[i] = 1.0*(lossPlus-lossMinus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(apprxDerv-gradient[idx]))
		if(cerr>=err_tol):
			print 'Mean computed error ',cerr,' is larger than the error tolerance -- there is probably an error in the computation'
		else:
			print 'Mean computed error ',cerr,' is smaller than the error tolerance -- the computation was probably correct'

	# The following are convenience functions for doing batch-optimization using routines from 
	# scipy (e.g, fmin_cg)

	def compute_gradient(self,w,X,y):
		""" Computation of the gradient """
		weights = self.reroll(w)
		act = self.fprop(X,weights)
		grad = self.bprop(X,y,act,weights)
		return self.unroll(grad)

	def compute_cost(self,w,X,y):
		weights = self.reroll(w)
		act = self.fprop(X,weights)
		return self.compute_loss(act,y,weights)