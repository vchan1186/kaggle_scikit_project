from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA

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
    print "Finding optimal C and gamma for SVM with RBF Kernel"
    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=cv_label, n_folds=40)

    # Use the svm.SVC() as the cost function to evaluate parameter choices
    # NOTE: Perhaps we should run computations in parallel if needed. Does it
    # do that already within the class?
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

def svmRBFPredict(train, tr_label, cv_train, cv_label, test, test_label):
    # Use the cross-validation set to select the proper C and gamma parameters
    # Once optimal C and gamma are obtained, hard code them so you don't have to
    # retrain your classifier everytime. 

    # Note: ran getOptCandGamma and got back: 
    # The best classifier is:  SVC(C=10.0, cache_size=200, class_weight=None,
    #   coef0=0.0, degree=3,
    #   gamma=0.01, kernel=rbf, max_iter=-1, probability=False,
    #   random_state=None, shrinking=True, tol=0.001, verbose=False)

    # getOptCandGamma(cv_train, cv_label)

    # Train SVM with RBF/Gaussian kernel
    svc = svm.SVC(C=10.0, kernel='rbf', degree=3, gamma=0.01, coef0=0.0, \
                  shrinking=True, probability=False, tol=0.001, cache_size=200, \
                  class_weight=None, verbose=False, max_iter=-1, \
                  random_state=None)
    svc.fit(train, tr_label.ravel())

    test_pred = svc.predict(test[:,:]).astype(int)
    test_num_corr = sum(test_pred == test_label)
    test_num_error = test.shape[0] - test_num_corr
    test_per_corr = test_num_corr / float(test.shape[0]) * 100
    test_per_error = 100 - test_per_corr

    print ("# of errors: %d, Percent Error: %.2f%%"
           % (test_num_error, test_per_error))
    print ("# of correct: %d, Percent Corr: %.2f%%"
           % (test_num_corr, test_per_corr))


def main():
    # How much of the training set will be used to train
    # The remainder will be used as a cross-validation set
    # train_ratio = 0.60
    # cv_ratio = 0.20
    # test_ratio = 0.20

    # First 35 components have 98% of the explained variance.
    numPrinComp = 35

    # Import training and test data and scale
    print "Importing training and test data"
    train = csv_io.read_csv_np("./train.csv", False)
    train_mean = np.mean(train, 0)
    train_stddev = np.std(train, 0)
    train = (train-train_mean)/train_stddev
    test = csv_io.read_csv_np("./test.csv", False)
    test_mean = np.mean(test, 0)
    test_stddev = np.std(test, 0)
    test = (test-test_mean)/test_stddev

    num_samples, num_features = train.shape
    print "size of training data:", train.shape

    # Split training and label data 
    test = train[800:, :]
    cv_train = train[600:800, :]
    train = train[0:600, :]
    # Split label data
    tr_label = (csv_io.read_csv_np("./trainLabels.csv", False)).ravel()
    cv_label = (tr_label[600:800]).astype(int)
    test_label = (tr_label[800:]).astype(int)
    tr_label = (tr_label[0:600]).astype(int)

    # Perform PCA on the training and cv set
    print "Applying PCA with %d principal components" % (numPrinComp)
    pca = PCA(n_components=numPrinComp)
    pca.fit(train)
    # print sum(pca.explained_variance_ratio_)
    train = pca.transform(train)
    cv_train = pca.transform(cv_train)
    test = pca.transform(test)

    # Train an SVM with RBF kernel and make predictions using the 
    # trained classifier.
    svmRBFPredict(train, tr_label, cv_train, cv_label, test, test_label)

if __name__=="__main__":
    main()
