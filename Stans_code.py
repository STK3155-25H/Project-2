# ======================================================================================
# FUNCTIONS
# ======================================================================================
import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.simplefilter("error")
# ======================================================================================
# DATA FUNCTIONS
# ======================================================================================
# Runge function
# --------------------------------------------------------------------------------------
def runge_function(x, noise=0.0):
    """
    Compute the Runge function f(x) = 1 / (1 + 25x^2).
    
    Parameters
    ----------
    x : array_like
        Input values.
    noise : bool, optional
        If True, Gaussian noise (mean=0, std=0.5) is added. Default is False.
    
    Returns
    -------
    y : ndarray
        Output values of the Runge function (with optional noise).
    """
    y = 1 / (1 + 25 * x**2) + np.random.normal(0, noise, size=len(x), )
    return y

# polynomial_features_scaled function used to create a feature matrix that is already scaled
# --------------------------------------------------------------------------------------
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
        
# Polynomial features function with Min-Max scaling to [-1, 1]
# --------------------------------------------------------------------------------------
def polynomial_features_minmax(x, degree, intercept=True, col_mins=None, col_maxs=None, return_stats=False):
    """
    Generate a polynomial feature matrix and scale each column to [-1, 1].
    
    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input 1D feature vector.
    degree : int
        Maximum degree of polynomial.
    intercept : bool
        Whether to include a column of ones as the intercept term.
    col_mins : array-like, optional
        Minimum values of columns (for test set scaling)
    col_maxs : array-like, optional
        Maximum values of columns (for test set scaling)
    return_stats : bool, optional
        If True, return the column min/max for reuse on test set.
    
    Returns
    -------
    X_poly_scaled : ndarray of shape (n_samples, degree+1 or degree)
        Scaled polynomial feature matrix.
    col_mins, col_maxs : ndarray, ndarray (only if return_stats=True)
        Minimum and maximum of polynomial columns (for consistent test scaling).
    """
    x = np.array(x).reshape(-1)
    n = len(x)
    
    # Build raw polynomial features
    if intercept:
        X = np.zeros((n, degree + 1))
        X[:, 0] = 1
        for i in range(1, degree + 1):
            X[:, i] = x ** i
        start_col = 1
    else:
        X = np.zeros((n, degree))
        for i in range(degree):
            X[:, i] = x ** (i + 1)
        start_col = 0
    
    # Compute or use provided min/max
    if col_mins is None or col_maxs is None:
        col_mins = X[:, start_col:].min(axis=0)
        col_maxs = X[:, start_col:].max(axis=0)
        # safeguard against zero range
        col_maxs[col_maxs - col_mins == 0] += 1e-8
    
    # Scale columns to [-1,1]
    X[:, start_col:] = 2 * (X[:, start_col:] - col_mins) / (col_maxs - col_mins) - 1
    
    if return_stats:
        return X, col_mins, col_maxs
    else:
        return X

        
# Splitting and scaling function for the data
# --------------------------------------------------------------------------------------
def split_scale(x, y, seed):
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
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
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
    
    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered, y_mean

# Splitting and scaling function for the data with Min-Max scaling to [-1, 1]
# --------------------------------------------------------------------------------------
def split_scale_minmax(x, y, seed=42):
    """
    Split dataset into train/test sets and scale features to [-1, 1].
    
    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input features.
    y : ndarray of shape (n_samples,)
        Target values.
    
    Returns
    -------
    X_train_scaled : ndarray of shape (n_train, 1)
    X_test_scaled : ndarray of shape (n_test, 1)
    y_train_centered : ndarray of shape (n_train,)
    y_test_centered : ndarray of shape (n_test,)
    """
    from sklearn.model_selection import train_test_split

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

    # Reshape
    X_train = x_train.reshape(-1, 1)
    X_test  = x_test.reshape(-1, 1)

    # Min-max scale to [-1, 1] using training data
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train_scaled = 2 * (X_train - X_min) / (X_max - X_min) - 1
    X_test_scaled  = 2 * (X_test - X_min) / (X_max - X_min) - 1

    # Center y
    y_mean = y_train.mean()
    y_train_centered = y_train - y_mean
    y_test_centered  = y_test  - y_mean

    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered, y_mean

# ======================================================================================
# COST/LOST FUNCTIONS
# ======================================================================================

# Mean Squared Error (MSE) for regression
# --------------------------------------------------------------------------------------
def CostMSE(target, l1=0.0, l2=0.0):
    """
    Creates a Mean Squared Error (MSE) cost function with optional L1 and L2 regularization.
    
    Parameters
    ----------
    target : ndarray of shape (n_samples,)
        The true target values for regression.
    l1 : float, default=0.0
        Strength of L1 regularization (Lasso). Penalizes the absolute values of the parameters.
    l2 : float, default=0.0
        Strength of L2 regularization (Ridge). Penalizes the squared values of the parameters.
    
    Returns
    -------
    func : function
        A function that computes the regularized MSE for a given set of predictions and model parameters.
        
        Signature:
            func(predictions, params=None) -> float
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples,)
            Predicted target values.
        params : ndarray of shape (n_features,), optional
            Model parameters (e.g., regression coefficients) used for regularization. If None, no regularization is applied.
        
        Returns
        -------
        mse : float
            The mean squared error with optional regularization.
    
    Notes
    -----
    - The function is compatible with autograd for automatic gradient computation.
    """
    def func(predictions, params=None):
        mse = np.mean((target - predictions) ** 2)
        if params is not None:
            mse += l1 * np.sum(np.abs(params))
            mse += l2 * np.sum(params ** 2)
        return mse

    return func

# Binary Cross-Entropy (Log Loss) for binary classification
# --------------------------------------------------------------------------------------
def CostLogReg(target, l1=0.0, l2=0.0):
    """
    Creates a binary cross-entropy (log loss) cost function with optional L1 and L2 regularization.
    
    Parameters
    ----------
    target : ndarray of shape (n_samples,)
        True binary labels (0 or 1).
    l1 : float, default=0.0
        Strength of L1 regularization (Lasso).
    l2 : float, default=0.0
        Strength of L2 regularization (Ridge).
    
    Returns
    -------
    func : function
        A function that computes the regularized binary cross-entropy loss for a given set of predicted probabilities and model parameters.
        
        Signature:
            func(predictions, params=None) -> float
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples,)
            Predicted probabilities for the positive class (values in [0, 1]).
        params : ndarray of shape (n_features,), optional
            Model parameters used for regularization. If None, no regularization is applied.
        
        Returns
        -------
        loss : float
            The binary cross-entropy loss with optional L1/L2 regularization.
    
    Notes
    -----
    - A small epsilon (1e-10) is added to predictions to prevent log(0) errors.
    - Compatible with autograd for automatic gradient computation.
    """
    def func(predictions, params=None):
        epsilon = 1e-10
        bce = -(1.0 / target.size) * np.sum(
            target * np.log(predictions + epsilon) +
            (1 - target) * np.log(1 - predictions + epsilon)
        )
        if params is not None:
            bce += l1 * np.sum(np.abs(params))
            bce += l2 * np.sum(params ** 2)
        return bce

    return func

# Multiclass Cross-Entropy (Softmax Loss) for classification
# --------------------------------------------------------------------------------------
def CostCrossEntropy(target, l1=0.0, l2=0.0):
    """
    Creates a multiclass cross-entropy (Softmax) loss function with optional L1 and L2 regularization.
    
    Parameters
    ----------
    target : ndarray of shape (n_samples, n_classes)
        One-hot encoded true labels for multiclass classification.
    l1 : float, default=0.0
        Strength of L1 regularization (Lasso).
    l2 : float, default=0.0
        Strength of L2 regularization (Ridge).
    
    Returns
    -------
    func : function
        A function that computes the regularized multiclass cross-entropy loss for a given set of predicted probabilities and model parameters.
        
        Signature:
            func(predictions, params=None) -> float
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class (Softmax outputs).
        params : ndarray of shape (n_features,), optional
            Model parameters used for regularization. If None, no regularization is applied.
        
        Returns
        -------
        loss : float
            The multiclass cross-entropy loss with optional L1/L2 regularization.
    
    Notes
    -----
    - A small epsilon (1e-10) is added to predictions to prevent log(0) errors.
    - Compatible with autograd for automatic gradient computation.
    """
    def func(predictions, params=None):
        epsilon = 1e-10
        ce = -(1.0 / target.shape[0]) * np.sum(target * np.log(predictions + epsilon))
        if params is not None:
            ce += l1 * np.sum(np.abs(params))
            ce += l2 * np.sum(params ** 2)
        return ce

    return func

# ======================================================================================
# ACTIVATION FUNCTIONS
# ======================================================================================

# Sigmoid Activation
# --------------------------------------------------------------------------------------
def Sigmoid(x):
    """
    Sigmoid (logistic) activation function.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray
        Sigmoid of the input.
    
    Notes
    -----
    Sigmoid maps any real number to the interval (0, 1).
    """
    return 1 / (1 + np.exp(-x))

def Sigmoid_derivative(x):
    """
    Derivative of the Sigmoid function.

    Parameters
    ----------
    x : ndarray
        Input array (same as used in Sigmoid).

    Returns
    -------
    dy_dx : ndarray
        Derivative of the Sigmoid function.
    
    Notes
    -----
    For backpropagation, it is often more efficient to compute as:
        sigmoid(x) * (1 - sigmoid(x))
    """
    s = Sigmoid(x)
    return s * (1 - s)

# ReLU Activation
# --------------------------------------------------------------------------------------
def ReLU(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray
        ReLU applied element-wise: max(0, x).
    
    Notes
    -----
    Commonly used in hidden layers due to simplicity and gradient flow.
    """
    return np.maximum(0, x)

def ReLU_derivative(x):
    """
    Derivative of the ReLU function.

    Parameters
    ----------
    x : ndarray
        Input array (same as used in ReLU).

    Returns
    -------
    dy_dx : ndarray
        Derivative of ReLU: 1 if x > 0, else 0.
    """
    return (x > 0).astype(float)

# Leaky ReLU Activation
# --------------------------------------------------------------------------------------
def LeakyReLU(x, alpha=0.01):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Parameters
    ----------
    x : ndarray
        Input array.
    alpha : float, default=0.01
        Slope for x < 0 to avoid "dying ReLU" problem.

    Returns
    -------
    y : ndarray
        Leaky ReLU applied element-wise.
    
    Notes
    -----
    Leaky ReLU allows a small, non-zero gradient for negative inputs.
    """
    return np.where(x > 0, x, alpha * x)

def LeakyReLU_derivative(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function.

    Parameters
    ----------
    x : ndarray
        Input array (same as used in LeakyReLU).
    alpha : float, default=0.01
        Slope for negative inputs.

    Returns
    -------
    dy_dx : ndarray
        Derivative of Leaky ReLU: 1 if x > 0, else alpha.
    """
    return np.where(x > 0, 1.0, alpha)

# Linear Activation
# --------------------------------------------------------------------------------------
def Linear(x):
    """
    Linear (identity) activation function.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray
        Same as input (identity function).
    """
    return x

def Linear_derivative(x):
    """
    Derivative of the Linear function.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    dy_dx : ndarray
        Ones, same shape as input.
    """
    return np.ones_like(x)

# Function to determine the derivatives
# --------------------------------------------------------------------------------------
def derivate(f):
    """Return the derivative function for a given activation function."""
    if f == Sigmoid:
        return Sigmoid_derivative
    elif f == ReLU:
        return ReLU_derivative
    elif f == LeakyReLU:
        return LeakyReLU_derivative
    elif f == Linear:
        return Linear_derivative
    else:
        raise ValueError(f"No derivative defined for {f}")
# ======================================================================================
# SCHEDULER CLASSES
# ======================================================================================

# Abstract scheduler class, shared by all schedulers
# --------------------------------------------------------------------------------------
class Scheduler:
    """
    Abstract base class for learning rate schedulers used in optimization algorithms.

    Attributes:
        eta (float): The learning rate used to scale parameter updates.
    """

    def __init__(self, eta):
        """
        Initialize the scheduler.

        Args:
            eta (float): The learning rate.
        """
        self.eta = eta

    def update_change(self, gradient):
        """
        Compute the change to apply to the parameters based on the gradient.

        Args:
            gradient (float or np.ndarray): The current gradient.

        Returns:
            float or np.ndarray: The update to apply to the parameters.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset any internal state variables of the scheduler.

        Used between training runs or epochs if needed.
        """
        pass


# RMS_prop scheduler
# --------------------------------------------------------------------------------------
class RMS_prop(Scheduler):
    """
    RMSprop (Root Mean Square Propagation) learning rate scheduler.

    This method adapts the learning rate for each parameter by maintaining
    a moving average of squared gradients.

    Attributes:
        eta (float): Learning rate.
        rho (float): Decay rate for the moving average of squared gradients.
        second (float): Running average of squared gradients.
    """

    def __init__(self, eta, rho):
        """
        Initialize RMSprop scheduler.

        Args:
            eta (float): Learning rate.
            rho (float): Decay rate for moving average of squared gradients.
        """
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        """
        Compute the RMSprop update for a given gradient.

        Args:
            gradient (float or np.ndarray): The gradient to be used for update.

        Returns:
            float or np.ndarray: The RMSprop-adjusted update value.
        """
        delta = 1e-8  # avoid division by zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        """
        Reset the running average of squared gradients.
        """
        self.second = 0.0


# Adam scheduler
# --------------------------------------------------------------------------------------
class Adam(Scheduler):
    """
    Adam (Adaptive Moment Estimation) learning rate scheduler.

    Combines momentum and RMSprop techniques to adapt learning rates for
    each parameter using first (mean) and second (uncentered variance) moments
    of the gradient.

    Attributes:
        eta (float): Learning rate.
        rho (float): Exponential decay rate for the first moment estimates.
        rho2 (float): Exponential decay rate for the second moment estimates.
        moment (float or np.ndarray): First moment (mean) estimate.
        second (float or np.ndarray): Second moment (variance) estimate.
        n_epochs (int): Counter for bias correction terms.
    """

    def __init__(self, eta, rho, rho2):
        """
        Initialize Adam scheduler.

        Args:
            eta (float): Learning rate.
            rho (float): Decay rate for first moment estimate.
            rho2 (float): Decay rate for second moment estimate.
        """
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        """
        Compute the Adam update for a given gradient.

        Args:
            gradient (float or np.ndarray): The gradient to be used for update.

        Returns:
            float or np.ndarray: The Adam-adjusted update value.
        """
        delta = 1e-8  # avoid division by zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        """
        Increment epoch count and reset moment and second-moment estimates.

        Typically called after each training epoch.
        """
        self.n_epochs += 1
        self.moment = 0
        self.second = 0
    
# ======================================================================================
# FEED FORWARD NEURAL NETWORK CLASS
# ======================================================================================

# Object-Oriented Implementation Of The Neural Network
# --------------------------------------------------------------------------------------
class FFNN:
    """
    Feed Forward Neural Network (FFNN) supporting flexible architecture, activation functions,
    and cost functions. Can be used for regression or classification tasks.
    """
    
    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = Sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostMSE,
        seed: int = None
    ):
        """
        Initialize the FFNN object.
        
        Parameters
        ----------
        dimensions : tuple[int]
            Number of nodes in each layer (input, hidden layers..., output).
        hidden_func : Callable, optional
            Activation function to use in hidden layers (default: Sigmoid).
        output_func : Callable, optional
            Activation function for the output layer (default: linear identity).
        cost_func : Callable, optional
            Loss/cost function for training (default: CostMSE for regression).
        seed : int, optional
            Random seed for reproducible weight initialization.
        """
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self.reset_weights()
        self._set_classification()
        
    def reset_weights(self):
        """
        Initialize or reset the network's weights with small random values.
        Bias weights are also initialized separately.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1])
            
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)
            
    def _set_classification(self):
        """
        Determine if the network is performing classification based on the cost function.
        False = Regression
        True = Classification
        """
        self.classification = False
        if (self.cost_func.__name__ == "CostLogReg" or self.cost_func.__name__ == "CostCrossEntropy"):
            self.classification = True

    def _feedforward(self, X: np.ndarray):
        """
        Perform a feedforward pass through the network.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        
        Returns
        -------
        a : np.ndarray
            Output of the network after the forward pass.
        """
        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order
        # to add bias to our data
        bias = np.ones((X.shape[0], 1)) * 0.01
        X = np.hstack([bias, X])

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the
        # exponent indicates layer number).
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                # bias column again added to the data here
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a])
                self.a_matrices.append(a)
            else:
                try:
                    # a^L, the nodes in our output layers
                    z = a @ self.weights[i]
                    a = self.output_func(z)
                    self.a_matrices.append(a)
                    self.z_matrices.append(z)
                except Exception as OverflowError:
                    print("OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling")
        return a
        
    def predict(self, X: np.ndarray, threshold=0.5):
        """
        Compute network predictions for given input X.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        threshold : float, optional
            Threshold for classification outputs (default: 0.5).
        
        Returns
        -------
        np.ndarray
            Predictions of the network (continuous for regression, binary for classification).
        """
        predict = self._feedforward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Compute accuracy for classification tasks.
        
        Parameters
        ----------
        prediction : np.ndarray
            Network predictions.
        target : np.ndarray
            True labels.
        
        Returns
        -------
        float
            Fraction of correct predictions.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))

    def _backpropagate(self, X, t, l1, l2):
        """
        Perform backpropagation and update network weights using gradients.
        
        Parameters
        ----------
        X : np.ndarray
            Input data for the batch.
        t : np.ndarray
            Target outputs for the batch.
        l1 : float
            L1 regularization parameter.
        l2 : float
            L2 regularization parameter.
        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)

        for i in range(len(self.weights) - 1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                # for multi-class classification
                if (
                    self.output_func.__name__ == "softmax"
                ):
                    delta_matrix = self.a_matrices[i + 1] - t
                # for single class classification
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            # delta terms for hidden layer
            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])

            # calculate gradient
            gradient_weights = self.a_matrices[i][:, 1:].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )

            # regularization term
            gradient_weights += np.sign(self.weights[i][1:, :]) * l1
            gradient_weights += self.weights[i][1:, :] * l2

            # use scheduler
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(gradient_bias),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )

            # update weights and bias
            self.weights[i] -= update_matrix

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        l1: float = 0,
        l2: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
    ):
        """
        Train the neural network using batch gradient descent and schedulers.
        
        Parameters
        ----------
        X : np.ndarray
            Training input data.
        t : np.ndarray
            Training targets.
        scheduler : Scheduler
            Learning rate scheduler (e.g., RMS_prop, Adam).
        batches : int, optional
            Number of mini-batches per epoch.
        epochs : int, optional
            Number of training epochs.
        l1 : float, optional
            L1 regularization parameter.
        l2 : float, optional
            L2 regularization parameter.
        X_val : np.ndarray, optional
            Validation input data.
        t_val : np.ndarray, optional
            Validation targets.
        
        Returns
        -------
        scores : dict
            Dictionary containing training/validation errors and accuracies.
        """
        # setup 
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        X, t = resample(X, t)

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)
        if val_set:
            cost_function_val = self.cost_func(t_val)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        print(f"{scheduler.__class__.__name__}: L1={l1}, L2={l2}")

        try:
            for e in range(epochs):
                for i in range(batches):
                    # allows for minibatch gradient descent
                    if i == batches - 1:
                        # If the for loop has reached the last batch, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, l1, l2)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # computing performance metrics
                pred_train = self.predict(X)
                train_error = cost_function_train(pred_train)

                train_errors[e] = train_error
                if val_set:
                    
                    pred_val = self.predict(X_val)
                    val_error = cost_function_val(pred_val)
                    val_errors[e] = val_error

                if self.classification:
                    train_acc = self._accuracy(self.predict(X), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(pred_val, t_val)
                        val_accs[e] = val_acc

                # printing progress bar
                progression = e / epochs
                print_length = self._progress_bar(
                    progression,
                    train_error=train_errors[e],
                    train_acc=train_accs[e],
                    val_error=val_errors[e],
                    val_acc=val_accs[e],
                )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            1,
            train_error=train_errors[e],
            train_acc=train_accs[e],
            val_error=val_errors[e],
            val_acc=val_accs[e],
        )
        sys.stdout.write("")

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores

    def _progress_bar(self, progression, **kwargs):
        """
        Print a console progress bar during training.
        
        Parameters
        ----------
        progression : float
            Progress fraction (0.0 to 1.0).
        kwargs : dict
            Metrics to display alongside the progress bar.
        
        Returns
        -------
        int
            Length of the printed progress bar line.
        """
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._format(progression * 100, decimals=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if not np.isnan(kwargs[key]):
                value = self._format(kwargs[key], decimals=4)
                line += f"| {key}: {value} "
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        return len(line)

    def _format(self, value, decimals=4):
        """
        Format a float for display in the progress bar.
        
        Parameters
        ----------
        value : float
            Value to format.
        decimals : int, optional
            Number of decimals to show.
        
        Returns
        -------
        str
            Formatted string representation of the value.
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= decimals - 1:
            return str(round(value))
        return f"{value:.{decimals-n-1}f}"

# ======================================================================================
# EXPERIMENT AND RESULT FUNCTIONS
# ======================================================================================

# Function to run the experiment
# --------------------------------------------------------------------------------------
def Run_Experiment(X_train, y_train, X_test, y_test,
                    layers_config, activation_funcs, activation_names,
                    regularizations, eta, rho, rho2, scheduler_class, 
                    epochs=200, seed=42, cost_func=CostMSE, output_func=Linear):
    """
    Runs a series of feedforward neural network experiments over different architectures,
    activation functions, and regularization parameters, and returns the training and test
    Mean Squared Error (MSE) for each configuration.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_samples_train, n_features)
    y_train : np.ndarray
        Training targets, shape (n_samples_train,)
    X_test : np.ndarray
        Test features, shape (n_samples_test, n_features)
    y_test : np.ndarray
        Test targets, shape (n_samples_test,)
    layers_config : list of tuples
        Each tuple specifies the number of neurons in each hidden layer.
        Example: [(10,), (20, 10)].
    activation_funcs : list of callables
        List of activation functions to use in hidden layers.
    activation_names : list of str
        Names corresponding to activation functions, for tracking results.
    regularizations : list of tuples
        List of (l1, l2) regularization strengths.
    eta : float
        Learning rate for the optimizer.
    rho : float
        First momentum parameter (e.g., for Adam).
    rho2 : float
        Second momentum parameter (e.g., for Adam).
    scheduler_class : class
        Optimizer class to use (e.g., Adam).
    epochs : int, optional
        Number of training epochs (default is 200).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    cost_func : callable, optional
        Cost function to use (default is CostMSE).
    output_func : callable, optional
        Activation function for the output layer (default is Linear).

    Returns
    -------
    results : list of dicts
        Each dictionary contains:
            - 'layers': tuple, hidden layer sizes
            - 'activation': str, activation function name
            - 'l1': float, L1 regularization strength
            - 'l2': float, L2 regularization strength
            - 'mse_train': float, training MSE
            - 'mse_test': float, test MSE
    """
    results = []
    for layers in layers_config:
        for act_func, act_name in zip(activation_funcs, activation_names):
            for l1, l2 in regularizations:
                # Build network dimensions: input layer + hidden layers + output layer
                dims = (X_train.shape[1],) + layers + (1,)
            
                nn = FFNN(
                    dimensions=dims,
                    hidden_func=act_func,
                    output_func=output_func,
                    cost_func=cost_func,
                    seed=seed
                )

                # Initialize scheduler
                scheduler = scheduler_class(eta=eta, rho=rho, rho2=rho2)
            
                # Train
                scores = nn.fit(
                    X_train, y_train.reshape(-1,1),
                    scheduler=scheduler,
                    epochs=epochs,
                    l1=l1,
                    l2=l2  # L2 applied in backprop
                )
            
                # Evaluate
                pred_train = nn.predict(X_train)
                pred_test = nn.predict(X_test)
                mse_train = np.mean((y_train - pred_train.flatten())**2)
                mse_test = np.mean((y_test - pred_test.flatten())**2)
            
                results.append({
                    "layers": layers,
                    "activation": act_name,
                    "l1": l1,
                    "l2": l2,
                    "mse_train": mse_train,
                    "mse_test": mse_test
                })
                
    return results

# Function to filter the results for 2 variables
# --------------------------------------------------------------------------------------
def filter_results(var1_name, var2_name, fixed_params, resulst):
    var1_values = sorted(list({r[var1_name] for r in resulst}))
    var2_values = sorted(list({r[var2_name] for r in resulst}))

    heatmap = np.full((len(var1_values), len(var2_values)), np.nan)

    for i, v1 in enumerate(var1_values):
        for j, v2 in enumerate(var2_values):
            for r in resulst:
                if r[var1_name] == v1 and r[var2_name] == v2:
                    # check fixed params
                    match = all(r[k] == v for k,v in fixed_params.items())
                    if match:
                        heatmap[i,j] = r['mse_test']
    return heatmap, var1_values, var2_values

# Function to plot a heatmap for 2 given results parameters
# --------------------------------------------------------------------------------------
def plot_heatmap(heatmap, y_labels, x_labels, title="Heatmap", xlabel="", ylabel=""):
    plt.figure(figsize=(8,6))
    im = plt.imshow(heatmap, origin='upper', cmap='viridis_r')
    
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.yticks(np.arange(len(y_labels)), [str(l) for l in y_labels])
    
    
    plt.colorbar(im, label="Test MSE")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()