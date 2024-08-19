from sklearn.feature_selection import mutual_info_regression
import numpy as np


def node_pair_mutual_info_regression(a, b):
    a = a.reshape(-1, 1)
    if np.sum(a) == 0 or np.sum(b) == 0:
        return np.NaN
    return mutual_info_regression(a, b)[0]
