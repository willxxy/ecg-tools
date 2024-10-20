from scipy.linalg import cholesky
import numpy as np

def multivariate_zscore_normalization(X):
    C = np.cov(X)
    mu = np.mean(X, axis=1).reshape(-1, 1)
    C_inv_sqrt = np.linalg.inv(cholesky(C, lower=True))
    normalized_X = C_inv_sqrt @ (X - mu)
    return normalized_X
