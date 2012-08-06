#-------------------------------------------------------------------------------
# Name:        HMM Classifier
# Purpose:
#
# Author:      Mustafa
#
# Created:     05/08/2012
# Copyright:   (c) Mustafa Fanaswala, 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def main():
    pmf = np.array([0.4, 0.1, 0.1, 0.2, 0.2])
    alphabet = np.array([1,2,3,4,5])
    length=10000
    X = sp.zeros([length,1])
    for i in sp.r_[0:length]:
        X[i]=GenerateSample(alphabet, pmf)
    plt.hist(X, bins = [1,2,3,4,5,6], normed = True)
    plt.show()



def GenerateSample(alphabet, pmf):
    """ Generate a single sample from a discrete alphabet with given pmf using the
    discrete version of the acceptance-rejection method.
    alphabet - is a n x 1 numpy vector
    pmf - is a n x 1 numpy vector such that sum(pmf) = 1"""

    numSymbols = len(alphabet)
    assert(numSymbols == len(pmf))
    reference_pmf = sp.ones(numSymbols)/numSymbols
    c=max(pmf)*numSymbols
    condition_satisfied = False
    while (not condition_satisfied):
        Y = DiscreteRand(alphabet)
        fY = pmf[Y==alphabet]
        gY = reference_pmf[Y==alphabet]
        U = sp.rand()
        if ( U <= fY/(c*gY)):
            X=Y
            condition_satisfied = True
        else:
            condition_satisfied = False
    return X

def DiscreteRand(alphabet):
    """ Returns a symbol from alphabet with a uniform probability distribution over alphabet
    alphabet - is a n x 1 numpy vector containing symbols of the alphabet [1:n] """
    n = len(alphabet)
    U = sp.rand()
    X = sp.int32(n*U) + 1
    return alphabet[X-1]


if __name__ == '__main__':
    main()