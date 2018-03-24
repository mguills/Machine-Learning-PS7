"""
Author      : Matt Guillory & Jackson Crewe
Class       : HMC CS 158
Date        : 2018 Mar 05
Description : Multiclass Classification on Soybean Dataset
              This code was adapted from course material by Tommi Jaakola (MIT)
"""

# utilities
from util import *

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn import metrics

import math

######################################################################
# output code functions
######################################################################

def generate_output_codes(num_classes, code_type) :
    """
    Generate output codes for multiclass classification.

    For one-versus-all
        num_classifiers = num_classes
        Each binary task sets one class to +1 and the rest to -1.
        R is ordered so that the positive class is along the diagonal.

    For one-versus-one
        num_classifiers = num_classes choose 2
        Each binary task sets one class to +1, another class to -1, and the rest to 0.
        R is ordered so that
          the first class is positive and each following class is successively negative
          the second class is positive and each following class is successively negatie
          etc

    Parameters
    --------------------
        num_classes     -- int, number of classes
        code_type       -- string, type of output code
                           allowable: 'ova', 'ovo'

    Returns
    --------------------
        R               -- numpy array of shape (num_classes, num_classifiers),
                           output code
    """

    # part a: generate output codes
    # hint : initialize with np.ones(...) and np.zeros(...)
    R = None
    if code_type == "ova":
        R = np.ones((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    R[i][j] = -1
    if code_type == "ovo":
        num_classifiers = num_classes*(num_classes-1)/2
        R = np.zeros((num_classes, num_classifiers))
        k = 0
        for i in range(num_classes):
            for j in range(num_classes-(i+1)):
                R[i][k+j] = 1
                R[i+j+1][k+j] = -1
            k += num_classes - (i+1)

    return R


def load_code(filename) :
    """
    Load code from file.

    Parameters
    --------------------
        filename -- string, filename
    """

    # determine filename
    import util
    dir = os.path.dirname(util.__file__)
    f = os.path.join(dir, '..', 'data', filename)

    # load data
    with open(f, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")

    return data


def test_output_codes():
    R_act = generate_output_codes(3, 'ova')
    R_exp = np.array([[  1, -1, -1],
                      [ -1,  1, -1],
                      [ -1, -1,  1]])
    assert (R_exp == R_act).all(), "'ova' incorrect"

    R_act = generate_output_codes(3, 'ovo')
    R_exp = np.array([[  1,  1,  0],
                      [ -1,  0,  1],
                      [  0, -1, -1]])
    assert (R_exp == R_act).all(), "'ovo' incorrect"


######################################################################
# loss functions
######################################################################

def compute_losses(loss_type, R, discrim_func, alpha=2) :
    """
    Given output code and distances (for one example), compute losses (for each class).

    hamming  : Loss  = (1 - sign(z)) / 2
    sigmoid  : Loss = 1 / (1 + exp(alpha * z))
    logistic : Loss = log(1 + exp(-alpha * z))

    Parameters
    --------------------
        loss_type    -- string, loss function
                        allowable: 'hamming', 'sigmoid', 'logistic'
        R            -- numpy array of shape (num_classes, num_classifiers)
                        output code
        discrim_func -- numpy array of shape (num_classifiers,)
                        distance of sample to hyperplanes, one per classifier
        alpha        -- float, parameter for sigmoid and logistic functions

    Returns
    --------------------
        losses       -- numpy array of shape (num_classes,), losses
    """

    # element-wise multiplication of matrices of shape (num_classes, num_classifiers)
    # tiled matrix created from (vertically) repeating discrim_func num_classes times
    z = R * np.tile(discrim_func, (R.shape[0],1))    # element-wise

    # compute losses in matrix form
    if loss_type == 'hamming' :
        losses = np.abs(1 - np.sign(z)) * 0.5

    elif loss_type == 'sigmoid' :
        losses = 1./(1 + np.exp(alpha * z))

    elif loss_type == 'logistic' :
        # compute in this way to avoid numerical issues
        # log(1 + exp(-alpha * z)) = -log(1 / (1 + exp(-alpha * z)))
        eps = np.spacing(1) # numpy spacing(1) = matlab eps
        val = 1./(1 + np.exp(-alpha * z))
        losses = -np.log(val + eps)

    else :
        raise Exception("Error! Unknown loss function!")

    # sum over losses of binary classifiers to determine loss for each class
    losses = np.sum(losses, 1) # sum over each row

    return losses


def hamming_losses(R, discrim_func) :
    """
    Wrapper around compute_losses for hamming loss function.
    """
    return compute_losses('hamming', R, discrim_func)


def sigmoid_losses(R, discrim_func, alpha=2) :
    """
    Wrapper around compute_losses for sigmoid loss function.
    """
    return compute_losses('sigmoid', R, discrim_func, alpha)


def logistic_losses(R, discrim_func, alpha=2) :
    """
    Wrapper around compute_losses for logistic loss function.
    """
    return compute_losses('logistic', R, discrim_func, alpha)

def calculate_errors(y_test, y_pred) :
    error = 0
    for i in range(len(y_test)):
        error += y_test[i] != y_pred[i]
    return error


######################################################################
# classes
######################################################################

class MulticlassSVM :

    def __init__(self, R, C=10.0, kernel='poly', **kwargs) :
        """
        Multiclass SVM.

        Attributes
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            svms    -- list of length num_classifiers
                       binary classifiers, one for each column of R
            classes -- numpy array of shape (num_classes,) classes

        Parameters
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            C       -- numpy array of shape (num_classifiers,1) or float
                       penalty parameter C of the error term
            kernel  -- string, kernel type
                       see SVC documentation
            kwargs  -- additional named arguments to SVC
        """

        num_classes, num_classifiers = R.shape

        # store output code
        self.R = R

        # use first value of C if dimension mismatch
        try :
            if len(C) != num_classifiers :
                raise Warning("dimension mismatch between R and C " +
                                "==> using first value in C")
                C = np.ones((num_classifiers,)) * C[0]
        except :
            C = np.ones((num_classifiers,)) * C

        # set up and store classifier corresponding to jth column of R
        self.svms = [None for _ in xrange(num_classifiers)]
        for j in xrange(num_classifiers) :
            svm = SVC(kernel=kernel, C=C[j], **kwargs)
            self.svms[j] = svm


    def fit(self, X, y) :
        """
        Learn the multiclass classifier (based on SVMs).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), features
            y    -- numpy array of shape (n,), targets

        Returns
        --------------------
            self -- an instance of self
        """

        classes = np.unique(y)
        num_classes, num_classifiers = self.R.shape
        if len(classes) != num_classes :
            raise Exception('num_classes mismatched between R and data')
        self.classes = classes    # keep track for prediction

        n,d = X.shape
        for c in range(num_classifiers):
            # set up positive and negative arrays
            pos_ndx = []
            neg_ndx = []

            for i in range(n):
                # update positive and negative index arrays
                curr_class = np.where(classes == y[i])[0][0]
                if self.R[curr_class][c] == -1: neg_ndx.append(i)
                if self.R[curr_class][c] == 1: pos_ndx.append(i)
            X_train = np.zeros((len(pos_ndx)+len(neg_ndx), d))
            y_train = np.zeros(len(pos_ndx)+len(neg_ndx))

            # fill X_train and y_train using arrays
            for i in range(len(pos_ndx)):
                X_train[i] = X[pos_ndx[i]]
                y_train[i] = 1
            for i in range(len(neg_ndx)):
                X_train[i+len(pos_ndx)] = X[neg_ndx[i]]
                y_train[i+len(pos_ndx)] = -1

            # train the binary classifier
            svm = self.svms[c]
            svm.fit(X_train, y_train)
            self.svms[c] = svm



        pass
        ### ========== TODO : END ========== ###


    def predict(self, X, loss_func=hamming_losses) :
        """
        Predict the optimal class.

        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            loss_func -- loss function
                         allowable: hamming_losses, logistic_losses, sigmoid_losses

        Returns
        --------------------
            y         -- numpy array of shape (n,), predictions
        """

        n,d = X.shape
        num_classes, num_classifiers = self.R.shape

        # setup predictions
        y = np.zeros(n)

        for i in range(n):
            scores = np.zeros(num_classifiers)
            for j in range(num_classifiers):
                # calculate scores using distances to hyperplanes
                scores[j] = self.svms[j].decision_function(X[i].reshape(1,d))
            losses = compute_losses(loss_func, self.R, scores)
            # choose index of first minimum score
            y[i] = self.classes[np.argmin(losses)]

        return y


######################################################################
# main
######################################################################

def main() :
    # load data
    converters = {35: ord} # label (column 35) is a character
    train_data = load_data("soybean_train.csv", converters)
    test_data = load_data("soybean_test.csv", converters)
    num_classes = 15
    X_train = train_data.X
    y_train = train_data.y
    X_test = test_data.X
    y_test = test_data.y

    # part b : generate output codes
    test_output_codes()

    # parts c-e : train component classifiers, make predictions,
    #             compare output codes and loss functions
    R = generate_output_codes(num_classes, "ova")
    multi_svm = MulticlassSVM(R, C=10.0, kernel='poly', degree = 4, coef0 = 1, gamma = 1)
    multi_svm.fit(X_train, y_train)
    y_pred_ham = multi_svm.predict(X_test, loss_func="hamming")
    y_pred_sig = multi_svm.predict(X_test, loss_func="sigmoid")
    y_pred_log = multi_svm.predict(X_test, loss_func="sigmoid")

    print calculate_errors(y_test, y_pred_ham)
    print calculate_errors(y_test, y_pred_sig)
    print calculate_errors(y_test, y_pred_log)

    R = generate_output_codes(num_classes, "ovo")
    multi_svm = MulticlassSVM(R, C=10.0, kernel='poly', degree = 4, coef0 = 1, gamma = 1)
    multi_svm.fit(X_train, y_train)
    y_pred_ham = multi_svm.predict(X_test, loss_func="hamming")
    y_pred_sig = multi_svm.predict(X_test, loss_func="sigmoid")
    y_pred_log = multi_svm.predict(X_test, loss_func="sigmoid")

    print calculate_errors(y_test, y_pred_ham)
    print calculate_errors(y_test, y_pred_sig)
    print calculate_errors(y_test, y_pred_log)

    R = load_code("R1.csv")
    multi_svm = MulticlassSVM(R, C=10.0, kernel='poly', degree = 4, coef0 = 1, gamma = 1)
    multi_svm.fit(X_train, y_train)
    y_pred_ham = multi_svm.predict(X_test, loss_func="hamming")
    y_pred_sig = multi_svm.predict(X_test, loss_func="sigmoid")
    y_pred_log = multi_svm.predict(X_test, loss_func="sigmoid")

    print calculate_errors(y_test, y_pred_ham)
    print calculate_errors(y_test, y_pred_sig)
    print calculate_errors(y_test, y_pred_log)

    R = load_code("R2.csv")
    multi_svm = MulticlassSVM(R, C=10.0, kernel='poly', degree = 4, coef0 = 1, gamma = 1)
    multi_svm.fit(X_train, y_train)
    y_pred_ham = multi_svm.predict(X_test, loss_func="hamming")
    y_pred_sig = multi_svm.predict(X_test, loss_func="sigmoid")
    y_pred_log = multi_svm.predict(X_test, loss_func="sigmoid")

    print calculate_errors(y_test, y_pred_ham)
    print calculate_errors(y_test, y_pred_sig)
    print calculate_errors(y_test, y_pred_log)

    # test support vectors
    print multi_svm.svms[0].support_
    print multi_svm.svms[2].support_

if __name__ == "__main__" :
    main()
