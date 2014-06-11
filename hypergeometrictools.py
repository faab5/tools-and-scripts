# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

def HypergeometricVariableMinimum(N, K, n):
    "Return the minimum value the variable can take"
    return long(n-N+K if n-N+K > 0 else 0)

# <codecell>

def HypergeometricVariableMaximum(N, K, n):
    "Return the maximum value the variable can take"
    return long(n if n < K else K)

# <codecell>

import numpy
import scipy.stats
def HypergeometricSumOfLargerProbabilities(N, K, n, k):
    """
    Determines how unlikely it is that given a total sample of N elements with K elements of a specific type,
    a random subsample of n elements contains k elements of the specific type.
    Returns a value between 0 and 1, where 1 indicates very unlikely and 0 very likely.
    If the subsample is truly drawn in a random way from the total sample, then this 
    HypergeometricSumOfLargerProbabilities has a uniform distribution.
    """
    if n == 0 or K == 0:
        return numpy.random.uniform()
    kmin = HypergeometricVariableMinimum(N, K, n)
    kmax = HypergeometricVariableMaximum(N, K, n)
    sampleprobability = scipy.stats.hypergeom.pmf(k, N, K, n)
    result = sampleprobability * numpy.random.uniform()
    prob   = sampleprobability
    if (2*K == N):
        #This part is in stead of the else part because for symmetric distributions (where N = 2K)
        #the '<=' and the '<' below do not sufficiently distinguish the symmetric probabilities
        #left and right of the central value
        for i in range(k, n-k-1, +1):
            prob   = prob * float((K-i)*(n-i))/float((i+1)*(N-K-n+i+1))
            result = result + prob
        prob = sampleprobability
        for i in range(k, n-k, -1):
            prob   = prob * float((i)*(N-K-n+i))/float((n-i+1)*(K-i+1))
            result = result + prob
    else:
        for i in range(k, kmax, +1):
            prob   = prob * float((K-i)*(n-i))/float((i+1)*(N-K-n+i+1))
            if prob <= sampleprobability: break
            result = result + prob
        prob = sampleprobability
        for i in range(k, kmin, -1):
            prob   = prob * float((i)*(N-K-n+i))/float((n-i+1)*(K-i+1))
            if prob < sampleprobability: break
            result = result + prob
    return result
        

# <codecell>

import numpy
import scipy.stats
def InverseHypergeometricSumOfLargerProbabilities(m, M, n, N):
    """
    Determines how unlikely it is that after a first sample of N elements with n elements of a specific type,
    a second sample of M elements contains m elements of the specific type.
    Returns a value between 0 and 1, where 1 indicates very unlikely and 0 very likely.
    If the second sample is truly drawn from the same collection as the first sample, then this 
    InverseHypergeometricSumOfLargerProbabilities has a uniform distribution.
    """
    sampleprobability = scipy.stats.hypergeom.pmf(m, N+M, n+m, M)
    result = sampleprobability * numpy.random.uniform()
    prob = sampleprobability
    if (2*n == N):
        #This part is in stead of the else part because for symmetric distributions (where N = 2n)
        #the '<=' and the '<' below do not sufficiently distinguish the symmetric probabilities
        #left and right of the central value
        for i in range(m, M-m-1, +1):
            prob = prob * float((n+i+1)*(M-i))/float((i+1)*(N+M-n-i))
            result = result + prob
        prob = sampleprobability
        for i in range(m, M-m, -1):
            prob = prob * float((i)*(N+M-n-i+1))/float((n+i)*(M-i+1))
            result = result + prob
    else:
        for i in range(m, M, 1):
            prob = prob * float((n+i+1)*(M-i))/float((i+1)*(N+M-n-i))
            if prob <= sampleprobability: break
            result = result + prob
        prob = sampleprobability
        for i in range(m, 0, -1):
            prob = prob * float((i)*(N+M-n-i+1))/float((n+i)*(M-i+1))
            if prob < sampleprobability: break
            result = result + prob
    return result * float(N+1)/float(N+M+1)

# <codecell>

import numpy, math
def Hypergeometric2DHistogramCorrelationQuantisation(H2D):
    """
    Takes a 2D histogram of entries (unnormalized and unweighted)
    and determines if the two variables/axes are uncorrelated,
    i.e. if the probability to get an x and y is the product of
    probabilities to get x and y separately p(x,y) = p(x)p(y).
    It does so by calculating for each bin a p-value, whether 
    entries in bin (x,y) can be expected from all entries in x 
    and all entries in y.
    Returns a 2D histogram of p-values for each bin
    """
    xaxis = H2D.sum(0)
    yaxis = H2D.sum(1)
    SOLP  = numpy.zeros(shape = H2D.shape)
    ni = H2D.shape[1]
    nj = H2D.shape[0]
    N  = int(H2D.sum())
    for i in range(0, ni):
        for j in range(0, nj):
            p = HypergeometricSumOfLargerProbabilities(N, int(xaxis[i]), int(yaxis[j]), int(H2D[j][i]))
            SOLP[j][i] = p
    return SOLP

