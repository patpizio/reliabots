import numpy as np
from scipy.special import softmax

def negative_logit(y_proba, y):
    return -y_proba[np.arange(len(y_proba)), y]

def negative_softmax(y_proba, y):
    y_proba = softmax(y_proba, axis=1)
    return -y_proba[np.arange(len(y_proba)), y]

def logit_ratio(y_proba, y):
    idx_of_true = np.zeros(y_proba.shape, dtype=bool)
    idx_of_true[np.arange(len(y_proba)), y] = True
    num = y_proba[np.invert(idx_of_true)]
    if y_proba.shape[1] > 2:
        num = np.sum(num, axis=1)
    den = y_proba[idx_of_true] + 1e6
    return np.divide(num, den)
