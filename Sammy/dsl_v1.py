from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import csv_io
import pylab as pl

def plotData(train, tr_label):
    # Obtain the indices with a label 0 and 1
    tr_label1_idx = [idx for idx, val in enumerate(tr_label) if val == 1]
    tr_label0_idx = [idx for idx, val in enumerate(tr_label) if val == 0]
    # cv_label1_idx = [idx for idx, val in enumerate(cv_label) if val == 1]
    # cv_label0_idx = [idx for idx, val in enumerate(cv_label) if val == 0]

    # Plot training data based off of labels
    plt.plot(train[tr_label1_idx,0], train[tr_label1_idx,1], 'bo')
    plt.plot(train[tr_label0_idx,0], train[tr_label0_idx,1], 'ro')
    plt.show()

    plt.plot(train[tr_label1_idx,10], train[tr_label1_idx,11], 'bo')
    plt.plot(train[tr_label0_idx,10], train[tr_label0_idx,11], 'ro')
    plt.show()

    plt.plot(train[tr_label1_idx,30], train[tr_label1_idx,31], 'bo')
    plt.plot(train[tr_label0_idx,30], train[tr_label0_idx,31], 'ro')
    plt.show()


def costFunction(X, y):
    cost = 0
    return cost

def getOptCandGamma(cv_train, cv_label):
    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=cv_label, n_folds=40)

    # Use the svm.SVC() as the cost function to evaluate parameter choices
    # NOTE: Perhaps we should run computations in parallel if needed.
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(cv_train, cv_label)

    score_dict = grid.grid_scores_
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    pl.figure(figsize=(8,6))
    pl.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.xlabel('gamma')
    pl.ylabel('C')
    pl.colorbar()
    pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    pl.yticks(np.arange(len(C_range)), C_range)
    pl.show()

    print "The best classifier is: ", grid.best_estimator_

def main():
    # How much of the training set will be used to train
    # The remainder will be used as a cross-validation set
    train_ratio = 0.7

    # Import training data and normalize
    train = csv_io.read_csv_np("./train.csv", False)
    train_mean = np.mean(train,0)
    train_stddev = np.std(train,0)
    train = (train-train_mean)/train_stddev

    num_samples, num_features = train.shape
    print "Size of total training data", train.shape

    # Split training data 
    cv_train = train[num_samples*train_ratio::, :]
    train = train[0:num_samples*train_ratio, :]
    cv_num_samples = cv_train.__len__()
    print "Size of cross-validation set", cv_num_samples

    tr_label = (csv_io.read_csv_np("./trainLabels.csv", False)).ravel()
    cv_label = (tr_label[num_samples*train_ratio:]).astype(int)
    tr_label = (tr_label[0:num_samples*train_ratio]).astype(int)
    # plotData(train, tr_label)  

    # Split into training and cross-validation sets (70/30). 
    # Do we need to randomize our data?

    # Train SVM with polynomial kernel of degree 3
    # Fit takes a 1d array so we use ravel to transform from a column vector
    # into a 1D array
    # svc = svm.SVC(kernel='poly', degree=3)
    # svc.fit(train, tr_label.ravel())

    # Use the cross-validation set to select the proper C and gamma parameters
    # Once optimal C and gamma are obtained, hard code them so you don't have to
    # retrain your classifier everytime. 
    # getOptCandGamma(cv_train, cv_label)

    # Train SVM with RBF/Gaussian kernel
    svc = svm.SVC(C=100.0, kernel='rbf', degree=3, gamma=0.01, coef0=0.0, \
                  shrinking=True, probability=False, tol=0.001, cache_size=200, \
                  class_weight=None, verbose=False, max_iter=-1, \
                  random_state=None)
    svc.fit(train, tr_label.ravel())

    cv_pred = svc.predict(cv_train[:,:]).astype(int)
    cv_num_corr = sum(cv_pred == cv_label)
    cv_num_error = cv_num_samples - cv_num_corr
    cv_per_corr = cv_num_corr / float(cv_num_samples) * 100
    cv_per_error = 100 - cv_per_corr

    print ("# of errors: %d, Percent Error: %.2f%%"
           % (cv_num_error, cv_per_error))
    print ("# of correct: %d, Percent Corr: %.2f%%"
           % (cv_num_corr, cv_per_corr))

    # Test Set (Don't forget to scale the features!)
    test = csv_io.read_csv_np("./test.csv", False)
    test_pred = svc.predict(test[:,:]).astype(int)
    # test_num_corr = sum(test_pred == test_label)
    # test_num_error = test_num_samples - test_num_corr
    # test_per_corr = test_num_corr / float(test_num_samples) * 100
    # test_per_error = 100 - test_per_corr

    # print ("# of errors: %d, Percent Error: %.2f%%"
    #        % (test_num_error, test_per_error))
    # print ("# of correct: %d, Percent Corr: %.2f%%"
    #        % (test_num_corr, test_per_corr))
    # print test_pred
    
if __name__=="__main__":
    main()
