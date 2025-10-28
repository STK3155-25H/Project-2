import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
import time
import os
from main import SEED
from typing import Callable 
from functions import cost_function

# -----------------------------------------------------------------------------------------
# Gradient descent code for advanced methods
def Gradient_descent_advanced(X, y, cost_func: cost_function, hard_coded_version = False, lam=0.01, lr=0.01, n_iter=1000, tol=1e-6, method='vanilla', beta=0.9, epsilon=1e-8, batch_size=1, use_sgd=False, theta_history=False):
    """
    Gradient Descent for OLS (Type=0), Ridge (Type=1) or LASSO (Type=2)
    With advanced optimizers: Momentum, AdaGrad, RMSProp, Adam.
    With option to use stochastic gradient descent

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    Type : int, optional
        0 = OLS, 1 = Ridge, 2 = LASSO. Default is 0.
    lam : float, optional
        Regularization strength (only for Ridge). Default is 0.01.
    lr : float, optional
        Base learning rate. Default is 0.01.
    n_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    method : str, optional
        Optimization method. Options are 'vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam'. Default is 'vanilla'.
    beta : float, optional
        Momentum or decay parameter (used in momentum, RMSProp, Adam). Default is 0.9.
    epsilon : float, optional
        Small number to avoid division by zero in adaptive methods. Default is 1e-8.
    batch_size : int, optional
        The size of the batches used for sgd
    use_sgd : bool, optional
        If True, use stochastic gradient descent with random mini-batches.
        If False, use full-batch gradient descent. Default is False.
    theta_history : bool, optional
        If False, return only the final parameter vector.
        If True, return the full history of parameters over all iterations.
        Default is False.

    Returns
    -------
    theta : ndarray of shape (n_features,)
        Estimated parameters after gradient descent (if `theta_history=False`).
    history : ndarray of shape (n_iter, n_features)
        Full trajectory of parameters during training (if `theta_history=True`).
    """
    n, p = X.shape
    theta = np.zeros(p)
    history = []
    
    # Initialize variables for optimizers
    v = np.zeros(p)      # momentum
    G = np.zeros(p)      # AdaGrad
    S = np.zeros(p)      # RMSProp
    m = np.zeros(p)      # Adam first moment
    v_adam = np.zeros(p) # Adam second moment
    t = 0
    
    for epoch in range(n_iter):
        theta_old = theta.copy()

        # Choose batch
        if use_sgd:
            # Pick random mini-batch
            indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
        else:
            # Full dataset
            X_batch = X
            y_batch = y
            
        grad = cost_func.grad(hard_coded_version=hard_coded_version)
        
        t += 1
        # Calculate theta
        if method == 'vanilla':
            theta -= lr * grad
            
        elif method == 'momentum':
            v = beta * v + (1 - beta) * grad
            theta -= lr * v
            
        elif method == 'adagrad':
            G += grad**2
            theta -= lr * grad / (np.sqrt(G) + epsilon)
            
        elif method == 'rmsprop':
            S = beta * S + (1 - beta) * grad**2
            theta -= lr * grad / (np.sqrt(S) + epsilon)
            
        elif method == 'adam':
            m = beta * m + (1 - beta) * grad
            v_adam = beta * v_adam + (1 - beta) * (grad**2)
            m_hat = m / (1 - beta**t)
            v_hat = v_adam / (1 - beta**t)
            theta -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
        else:
            print("Method not found")
            break
            
        # Add the theta value to the history list only if requested
        if theta_history:
            history.append(theta.copy())
            
        # Convergence check
        if np.linalg.norm(theta - theta_old) < tol:
            # print(f"Converged after {epoch+1} iterations, with {method}")
            break
    
    if theta_history == False:
        return theta
    else:
        return np.array(history)
