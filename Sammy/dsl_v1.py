from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import csv_io

def costFunction(X, y):
    cost = 0
    return cost
   
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
    tr_label1_idx = [idx for idx, val in enumerate(tr_label) if val == 1]
    tr_label0_idx = [idx for idx, val in enumerate(tr_label) if val == 0]
    cv_label1_idx = [idx for idx, val in enumerate(cv_label) if val == 1]
    cv_label0_idx = [idx for idx, val in enumerate(cv_label) if val == 0]

    # Split into training and cross-validation sets (70/30). 
    # Do we need to randomize our data?

    # Plot training data based off of labels
#    plt.plot(train[tr_label1_idx,0], train[tr_label1_idx,1], 'bo')
#    plt.plot(train[tr_label0_idx,0], train[tr_label0_idx,1], 'ro')
#    plt.show()
#
#    plt.plot(train[tr_label1_idx,10], train[tr_label1_idx,11], 'bo')
#    plt.plot(train[tr_label0_idx,10], train[tr_label0_idx,11], 'ro')
#    plt.show()
#
#    plt.plot(train[tr_label1_idx,30], train[tr_label1_idx,31], 'bo')
#    plt.plot(train[tr_label0_idx,30], train[tr_label0_idx,31], 'ro')
#    plt.show()
#
    # Train SVM with polynomial kernel of degree 3
    # Fit takes a 1d array so we use ravel to transform from a column vector
    # into a 1D array
#    svc = svm.SVC(kernel='poly', degree=3)
#    svc.fit(train, tr_label.ravel())

    # Train SVM with RBF/Gaussian kernel
    svc = svm.SVC(kernel='rbf')
    svc.fit(train, tr_label.ravel())

    cv_pred = svc.predict(cv_train[:,:]).astype(int)
    cv_num_corr = sum(cv_pred == cv_label)
    cv_num_error = cv_num_samples - cv_num_corr
    cv_per_corr = cv_num_corr / float(cv_num_samples) * 100
    cv_per_error = 100 - cv_per_corr

    print "# of errors: %d, Percent Error: %.2f%%" % (cv_num_error, cv_per_error)
    print "# of correct: %d, Percent Corr: %.2f%%" % (cv_num_corr, cv_per_corr)

    
if __name__=="__main__":
    main()
