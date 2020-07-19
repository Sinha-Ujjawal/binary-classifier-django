from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


def get_model(maxIter, solver, penalty, c, l1Ratio):
    c = 10 ** c
    return LogisticRegression(
        max_iter=maxIter, solver=solver, penalty=penalty, C=c, l1_ratio=l1Ratio
    )
