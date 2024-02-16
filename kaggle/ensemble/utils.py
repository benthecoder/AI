import numpy as np


def coefficients(preds):
    A = np.asarray(preds[:, 0], dtype=bool)
    B = np.asarray(preds[:, 1], dtype=bool)

    a = np.sum(A * B)  # A right, B right
    b = np.sum(~A * B)  # A wrong, B right
    c = np.sum(A * ~B)  # A right, B wrong
    d = np.sum(~A * ~B)  # A wrong, B wrong

    return a, b, c, d


def disagreement(preds, i, j):
    L = preds.shape[1]
    a, b, c, d = coefficients(preds[:, [i, j]])
    return float(b + c) / (a + b + c + d)


def paired_q(preds, i, j):
    L = preds.shape[1]
    # div = np.zeros((L * (L - 1)) // 2)
    a, b, c, d = coefficients(preds[:, [i, j]])
    return float(a * d - b * c) / ((a * d + b * c) + 10e-24)


def entropy(preds):
    L = preds.shape[1]
    tmp = np.sum(preds, axis=1)
    tmp = np.minimum(tmp, L - tmp)
    ent = np.mean((1.0 / (L - np.ceil(0.5 * L))) * tmp)
    return ent
