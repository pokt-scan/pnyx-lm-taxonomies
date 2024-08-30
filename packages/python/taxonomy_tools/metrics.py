from sklearn.feature_selection import mutual_info_regression
import numpy as np
import itertools
import pandas as pd
from typing import Tuple

# All methods will be calculated on every node combination (using function
# "apply_to_pairs") if they are in this list, otherwise it means that they will
# only be calculated on the triangular elements, not on the diagonal or the
# mirrored pairs.
permutation_methods = ["mutual_information", "success_association"]


def apply_to_pairs(df, func):
    """Applies a function to all combinations of pairs of columns in a DataFrame.

    Args:
    df: The pandas DataFrame.
    func: The function to apply to each pair of columns.

    Returns:
    A DataFrame containing the results of applying the function to each pair.
    """

    columns = df.columns.values
    num_cols = len(columns)
    columns_idxs = [i for i in range(num_cols)]
    pairs = itertools.permutations(columns_idxs, 2)

    metric_mat = np.zeros((num_cols, num_cols))
    # Triang
    for col1, col2 in pairs:
        result = func(df[columns[col1]].values, df[columns[col2]].values)
        metric_mat[col1, col2] = result
    # Diag
    for col in columns_idxs:
        result = func(df[columns[col]].values, df[columns[col]].values)
        metric_mat[col, col] = result

    return pd.DataFrame(metric_mat, columns=df.columns.values, index=df.columns.values)


def remove_nans(a: np.array, b: np.array) -> Tuple[np.array, np.array]:
    these = 0 == (np.isnan(a) + np.isnan(b))
    a = a[these]
    b = b[these]
    return a, b


################################################################################
# Node Pair Metrics
################################################################################
# All metrics starting with "node_pair_" are meant to be used with the function
# "apply_to_pairs" or the "corr" method of a Pandas.DataFrame


def node_pair_mutual_info_regression(a: np.array, b: np.array) -> float:
    a, b = remove_nans(a, b)
    a = a.reshape(-1, 1)
    if np.sum(a) == 0 or np.sum(b) == 0:
        return np.NaN
    return mutual_info_regression(a, b)[0]


def node_pair_success_association(a: np.array, b: np.array) -> float:
    """
    This metric calculates a value that reflects the association between A and B
    defined as the proportion of the total where B has a higher value than A.
    """
    a, b = remove_nans(a, b)

    # Number of A smaller than B, ignore NaNs
    num_asb = np.nan_to_num(np.sum(a <= b), 0)
    # Number of total non-NaN entries
    support = np.sum(0 == (np.isnan(a) + np.isnan(b)))

    return num_asb / support
