import numpy as np
from functools import reduce

def nauc(y_test: np.ndarray, prediction: np.ndarray):
    
    label_num = y_test.shape[1]
    auc = np.empty(label_num)
    for k in range(label_num):
        r_ = tiedrank(prediction[:, k])
        s_ = y_test[:, k]
        npos = sum(s_ == 1)
        nneg = sum(s_ < 1)
        auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    return 2 * mvmean(auc) - 1

def acc(y_test: np.ndarray, prediction: np.ndarray):
    """Get accuracy"""
    epsilon = 1e-15
    # normalize prediction
    prediction_normalized = \
        prediction / (np.sum(np.abs(prediction), axis=1, keepdims=True) + epsilon)
    return np.sum(y_test * prediction_normalized) / y_test.shape[0]

def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x))[
        1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))

def tiedrank(a):
    '''
    Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.
    '''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        k0 = 0
        oldval = sa[k0]
        for k in range(1, m):
            newval = sa[k]
            if newval == oldval:
                # moving average
                R[k0:k + 1] = R[k - 1] * (k - k0) / (k - k0 + 1) + R[k] / (k - k0 + 1)
            else:
                k0 = k
                oldval = sa[k0]
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S