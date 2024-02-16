import numpy as np
from numba import jit


def mse(y_pred: np.array, y_true: np.array) -> float:
    squared_diff = (y_pred - y_true) ** 2
    mse = np.mean(squared_diff)
    return mse


def rmse(y_true, y_pred):
    return np.sqrt(mse)


def r2(y_true, y_pred):
    pass


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc
