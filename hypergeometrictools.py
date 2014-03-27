# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy

def BinomialCoefficient(n, k):
    "Calculates (n k) = n!/k!(n-k)!"
    l = min(k, n-k)
    result = 1.0
    for i in range(1, l+1):
        result = result * float(n-l+i) / float(i)
    return long(result+0.5)

def HypergeometricVariableMinimum(N, K, n):
    "Return the minimum value the variable can take"
    return long(n-N+K if n-N+K > 0 else 0)

def HypergeometricVariableMaximum(N, K, n):
    "Return the maximum value the variable can take"
    return long(n if n < K else K)

def HypergeometricProbability(N, K, n, k):
    "Calculates (K k) * (N-K n-k) / (N n)"
    return float(BinomialCoefficient(N-K, n-k) * BinomialCoefficient(K, k)) / float(BinomialCoefficient(N, n))

def HypergeomtricCumulativeProbability(N, K, n, k):
    "Returns the sum of probabilities from the minimum value up to and including k"
    kmin = HypergeometricVariableMinimum(N, K, n)
    kmax = HypergeometricVariableMaximum(N, K, n)
    result = 0.0
    if (k - kmin + 1 < kmax - k):
        prob   = HypergeometricProbability(N, K, n, kmin)
        result = prob 
        for i in range(kmin, k):
            prob   = prob * float((K-i)*(n-i))/float((i+1)*(N-K-n+i+1))
            result = result + prob
    else:
        prob   = HypergeometricProbability(N, K, n, k)
        result = 1
        for i in range(k, kmax):
            prob   = prob * float((K-i)*(n-i))/float((i+1)*(N-K-n+i+1))
            result = result - prob
    return float(result)

def HypergeometricSumOfLargerProbabilities(N, K, n, k):
    "Returns the sum of probabilites between k and some value k' where the probabilites are larger or equal than that of k"
    kmin = HypergeometricVariableMinimum(N, K, n)
    kmax = HypergeometricVariableMaximum(N, K, n)
    sampleprobability = HypergeometricProbability(N, K, n, k)
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
        
def HypergeometricRandomVariable(N, K, n):
    "Returns a random value according to the hypergeometric distribution"
    p    = numpy.random.uniform()
    k    = HypergeometricVariableMinimum(N, K, n)
    kmax = HypergeometricVariableMaximum(N, K, n)
    prob = HypergeometricProbability(N, K, n, k)
    intg = prob;
    while intg < p and k < kmax:
        prob *= float((K-k)*(n-k))/float((k+1)*(N-K-n+k+1))
        intg += prob
        k    += 1
    return k
