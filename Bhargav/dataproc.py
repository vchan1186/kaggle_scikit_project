import numpy as np

# note: data is assumed to be of the form d x n, where the rows correspond to individual features,
# and the columns correspond to instances

def read_csv_file(csvFile):
	""" reads a csv file """
	return np.genfromtxt(csvFile,delimiter=",")

def normalize_range(X):
	""" Given a data matrix of continuous values, normalizes the range to -1 to 1 """
	

def get_subset_idx(X,n):
	""" Returns a random percentage (or number) of the provided data (just indices) """
	N = np.shape(X)[1]
	rIdx = np.random.permutation(N)
	
	if n > N:
		raise ValueError("%d exceeds the number of instances, %d" %(n,N))

	# accounts for both a percentage or actual number
	if n <= 1.0:
		return rIdx[:np.floor(n*N)]
	else:
		return rIdx[:n]

def split_train_validation(X,n):
	""" Reserves n (either a percent or number of instances) instances for training,
	and assigns the remaining for validation """

	N = np.shape(X)[1]
	allIdx = range(N)
	trIdx = get_subset_idx(X,n)
	valIdx = np.setdiff1d(allIdx,trIdx)
	return trIdx, valIdx
	
def shuffle_data(X):
	""" Shuffles data """
	return X[:,get_subset_idx(X,1.0)]