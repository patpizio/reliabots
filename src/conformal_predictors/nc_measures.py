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

def margin_error_func(y_proba, y):  # y is:  i) true y_cal  and  ii) y_test candidate
    prob_true = y_proba[np.arange(len(y_proba)), y]
    print(prob_true)
    y_probba = y_proba.copy()
    np.put_along_axis(y_probba, y[:, None], -np.inf, axis=1)  # "exclude" probs of true labels
    print(f'y_probba now: \n{y_probba}')
    max_among_false = np.max(y_probba, axis=1)
    print(f'Max among false: \n{max_among_false}')
    return 0.5 - ((prob_true - max_among_false) / 2)
