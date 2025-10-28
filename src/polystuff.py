import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import time
import os
from numba import njit
from main import SEED

# -----------------------------------------------------------------------------------------
# New polynomial features function that allows for a creation of a matrix without the intercept column
def polynomial_features(x, p, intercept=True):
    """
    Generate polynomial features up to degree p.

    Parameters
    ----------
    x : array_like of shape (n_samples,)
        Input feature vector.
    p : int
        Polynomial degree.
    intercept : bool, optional
        If True, includes a column of ones as intercept. Default is True.

    Returns
    -------
    X : ndarray of shape (n_samples, p+1) if intercept else (n_samples, p)
        Polynomial feature matrix.
    """
    x = np.array(x).reshape(-1)
    n = len(x)
    if intercept==False:
        X = np.zeros((n, p))
        for i in range(p):
            X[:, i] = x**(i+1)
    else:
        X = np.zeros((n, p+1))
        for i in range(p+1):
            X[:, i] = x**(i)
    return X

# -----------------------------------------------------------------------------------------
# New polynomial features function that allows for a creation of a matrix without the intercept column. It also scales the feature matrix
def polynomial_features_scaled(x, degree, intercept=True, col_means=None, col_stds=None, return_stats=False):
    """
    Generate a polynomial feature matrix and automatically scale each column.
    
    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input 1D feature vector.
    degree : int
        Maximum degree of polynomial.
    intercept : bool
        Whether to include a column of ones as the intercept term.
    col_means : array-like, optional
        Means of training polynomial columns for scaling test set. Ignored if None.
    col_stds : array-like, optional
        Standard deviations of training polynomial columns for scaling test set. Ignored if None.
    return_stats : bool, optional
        If True, return the column means and stds for reuse on test set.
    
    Returns
    -------
    X_poly_scaled : ndarray of shape (n_samples, degree+1 or degree)
        Scaled polynomial feature matrix.
    col_means, col_stds : ndarray, ndarray (only if return_stats=True)
        Means and stds of polynomial columns (for consistent test scaling).
    """
    x = np.array(x).reshape(-1)
    n = len(x)
    
    # Build raw polynomial features
    if intercept:
        X = np.zeros((n, degree + 1))
        X[:, 0] = 1
        for i in range(1, degree + 1):
            X[:, i] = x ** i
    else:
        X = np.zeros((n, degree))
        for i in range(degree):
            X[:, i] = x ** (i + 1)
    
    # Scale columns (except intercept) using training stats if provided
    if intercept:
        start_col = 1
    else:
        start_col = 0
    
    if col_means is None or col_stds is None:
        # Compute stats from current data (usually train set)
        col_means = X[:, start_col:].mean(axis=0)
        col_stds = X[:, start_col:].std(axis=0)
        col_stds[col_stds == 0] = 1  # safeguard
    
    # Apply scaling
    X[:, start_col:] = (X[:, start_col:] - col_means) / col_stds
    
    if return_stats:
        return X, col_means, col_stds
    else:
        return X

# -----------------------------------------------------------------------------------------
# Splitting and scaling function for the data
def split_scale(x, y, random_state=None):
    """
    Split dataset into train/test sets and scale features.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input features.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    X_train_scaled : ndarray of shape (n_train, 1)
        Scaled training features.
    X_test_scaled : ndarray of shape (n_test, 1)
        Scaled test features.
    y_train_centered : ndarray of shape (n_train,)
        Centered training targets.
    y_test_centered : ndarray of shape (n_test,)
        Centered test targets.
    """
    rs = SEED if random_state is None else random_state

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rs)
    # reshape the x datas to make them a 2d matrix
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)
    # calculate mean and std on the training set for X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1 # safe guard
    # scale the x data sets
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    # calculate mean of y
    y_mean = y_train.mean()
    # center the y data sets
    y_train_centered = y_train - y_mean
    y_test_centered = y_test - y_mean
    
    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered
