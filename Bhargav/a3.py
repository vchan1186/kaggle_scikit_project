import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import copy
from nnetpy import *

# Main function
def main(param):
	""" Main segment of code which executes the initialization, training, and application of the 
	neural network to the ML problem"""
	
	#-------------------#
	# PARAMETER SETTING #
	#-------------------#
	print 'Initial Settings of the Neural Network:'
	print '---------------------------------------'
	print 'Number of hidden units: ',param.hid_size
	print 'Learning Rate: ',param.learning_rate
	print 'Regularization Coefficient (Weight Decay): ',param.weight_decay
	print 'alpha Coefficient (Viscosity): ',param.alpha
	print 'Minibatch Size: ',param.minibatch_size
	print 'Early Stopping: ',param.early_stopping_flag
	print 'Dropout: ',param.dropout_flag
	print 'Cost Function: Cross-entropy error (Softmax)'
	print 'Current Training Mode: Stochastic Gradient Descent with Momentum'

	#---------------------#
	# DATA INITIALIZATION #
	#---------------------#
	# Read the "training.mat" file which has the input/output values... 
	trdata = scipy.io.loadmat('training_data.mat')
	#...and store them into numpy arrays, concatenated with a row of 1s for the biases
	num_tr = trdata['training_inputs'].shape[1]
	num_val = trdata['validation_inputs'].shape[1]
	num_te = trdata['test_inputs'].shape[1]
	training = data(trdata['training_inputs'],trdata['training_targets'])	# training set
	validation = data(trdata['validation_inputs'],trdata['validation_targets'])	# validation set
	test = data(trdata['test_inputs'],trdata['test_targets'])	# test set

 	#-------------------------#
	# NEURAL NETWORK TRAINING #
	#-------------------------#
	# Intiialization
	print '\nInitializing the 2-layer Neural Network...'
	input_size = training.inputs.shape[0]
	output_size = training.targets.shape[0]
	nn = mlp(input_size, output_size,param)
	nn.initialize()
	# pre-append a row of ones to account for the bias-term - doing this once at the very
	# beginning will save time
	training.append_ones()
	validation.append_ones()
	test.append_ones()

	print 'Starting Training...'
	# vectors which store training, validation, and test errors
	training_loss_v = np.empty(param.num_iter)
	validation_loss_v = np.empty(param.num_iter)
	# stores the final loss values; these are deep copies
	training_loss = validation_loss = test_loss = 0
	# needed for weight-decay
	training_class_loss = validation_class_loss = test_class_loss = 0
	
	# early stopping
	best_validation_loss = np.inf
	if param.early_stopping_flag:
		nn.store_best_model()

	iters = np.arange(0,param.num_iter)
	for iter in iters:
		# consider only a small batch of the training set
		minidx = (iter*param.minibatch_size)%num_tr
		maxidx = ((iter+1)*param.minibatch_size)%num_tr
		if minidx > maxidx:
			batch_idx = range(minidx,num_tr)+range(0,maxidx)
		else:
			batch_idx = range(minidx,maxidx)
		training.batch(batch_idx)	# set pointers for current batch
		nn.propagate_data(training)	# perform fprop and brprop to get the derivatives...
		nn.update_weights() #...and update the weights

		training.batch(range(0,training.inputs.shape[1]))	# when we compute training loss, we compute it on the entire training set
		training_loss_v[iter] = nn.compute_loss(training)
		validation_loss_v[iter] = nn.compute_loss(validation)
		if param.early_stopping_flag and validation_loss_v[iter] < best_validation_loss:
			best_validation_loss = validation_loss_v[iter]
			nn.store_best_model()
		
		if (iter+1)%10==0:
			print 'After ',iter+1,' iterations, training loss = ',training_loss_v[iter],', validation loss = ',validation_loss_v[iter]

	#-------------------------------------#
	# ACCURACY ASSESSMENT AND PERFORMANCE #
	#-------------------------------------#
	training.batch(range(0,training.inputs.shape[1]))	# reset the training data to include all data

	# early stopping
	if param.early_stopping_flag:
		nn.use_best_model()	# this chooses the model that had achieved the best validation loss
		validation_loss = best_validation_loss # should theoretically be equal to nn.compute_loss(validation) - good for sanity checks
		training_loss = nn.compute_loss(training)	# the model has been reset to when the best valdation loss was achieved, so recompute
	else:
		# we just want the last value of the loss vectors
		validation_loss = validation_loss_v[-1]	
		training_loss = training_loss_v[-1]

	# Measure how well the classifier did on the test data
	test_loss = nn.compute_loss(test)	

	# weight-decay
	if not param.weight_decay == 0:
		validation_class_loss = nn.compute_class_loss(validation)
		training_class_loss = nn.compute_class_loss(training)
		test_class_loss = nn.compute_class_loss(test)

	# Misclassification error
	training_mce = nn.compute_mce(training)
	validation_mce = nn.compute_mce(validation)
	test_mce = nn.compute_mce(test)

	print '\nStatistics & Performance:'
	print '-----------------------------'
	print 'Final loss on training data: ', training_loss
	print 'Final loss on validation data: ', validation_loss
	print 'Final loss on testing data: ', test_loss
	if not param.weight_decay == 0:
		print 'Classification loss (no weight decay term) on training data: ', nn.compute_class_loss(training)
		print 'Classification loss (no weight decay term) on validation data: ',nn.compute_class_loss(validation)
		print 'Classification loss (no weight decay term) no test data: ',nn.compute_class_loss(test)
	print 'Training Error: ',training_mce
	print 'Validation Error: ',validation_mce
	print 'Test Error: ',test_mce

	# Plot Learning curves:
	plt.plot(iters+1,training_loss_v,'r',iters+1,validation_loss_v,'b')
	plt.xlabel('Iterations')
	plt.ylabel('Cross-entropy Error')
	plt.show() 

if __name__ == '__main__':

	# Parameters of the neural network
	num_iter=1000
	weight_decay=0.001
	hid_size=37
	learning_rate=0.35
	alpha=0.9
	minibatch_size=100
	gradient_flag=False
	early_stopping_flag=True
	dropout_flag=False
	training_mode = 'momentum'

	param = parameters(num_iter,weight_decay,hid_size,learning_rate,alpha,minibatch_size,gradient_flag,
		early_stopping_flag,dropout_flag,training_mode)
	main(param)