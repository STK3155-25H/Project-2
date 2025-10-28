
import scipy as scp
from typing import List, Dict, Tuple, Callable, Any
import autograd.numpy as np
# import numpy as np
weights_n_biases = Tuple[np.ndarray, np.ndarray]
layer = Dict[str,weights_n_biases]

# x, y = sy.symbols('x,y')

from autograd import elementwise_grad, grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



class activation_function:
    def __init__(self, expr: Callable[[np.ndarray|np.number], np.ndarray|np.number], der = None):
        self.func = expr
        self.der = der
        self.__name__ = expr.__name__
    def __call__(self, value: np.ndarray|np.number = None):
        if value is None:
            return self.func
        else : 
            return self.func(value)
        
    def diff(self, value: np.ndarray|np.number = None):
        diff = elementwise_grad(self.func, 0) 
        if value is None:
            return diff
        else: return diff(value)
        
        
def _ReLU(z):
    return np.where(z > 0, z, 0)

def _ReLU_der(z):
    return np.where(z > 0, 1, 0)  

def _leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)            

def _leaky_ReLU_der(z,  alpha=0.01):        
    return np.where(z > 0, 1, alpha)

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _sigmoid_der(z):
    s = _sigmoid(z)
    return s * (1 - s)  

def _softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def _softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


ReLU = activation_function(_ReLU, der=_ReLU_der)
leaky_ReLU = activation_function(_leaky_ReLU, der=_leaky_ReLU_der)
sigmoid = activation_function(_sigmoid, der=_sigmoid_der)
softmax = activation_function(_softmax)
softmax_vec = activation_function(_softmax_vec)

class cost_function:
    def __init__(self, expr, der = None):
        self.func = expr
        self.der = der
        self.__name__ = expr.__name__
    def __call__(self, prediction:np.ndarray|np.number = None  ,target: np.ndarray|np.number = None):
        if target is None or prediction is None:
            return self.func
        else : 
            return self.func(prediction, target)
        
    def grad(self, value: np.ndarray|np.number = None, target = None, hard_coded_version = False):
        if hard_coded_version:
            if self.der is not None:
                diff = self.der
            else:
                raise ValueError("Hard-coded derivative requested but 'der' is None")
        else:
            diff = grad(self.func, argnum=0)
        if value is None or target is None:
            return diff
        else: return diff(value)
        
#def _mse( prediction, target):
#    return np.mean((prediction - target)**2)

#def _mse_der( prediction, target):
#    return 2 * (prediction - target) / target.size

# MSE loss with optional L1 and L2 regularization
def _mse(prediction, target, weights=None, l1=0.0, l2=0.0):

    mse_loss = np.mean((prediction - target) ** 2)
    reg_term = 0.0
    if weights is not None:
        reg_term = l1 * np.sum(np.abs(weights)) + l2 * np.sum(weights ** 2)
    return mse_loss + reg_term

def _mse_der(prediction, target, weights=None, l1=0.0, l2=0.0):

    grad_pred = 2 * (prediction - target) / target.size

    grad_weights = None
    if weights is not None:
        grad_weights = l1 * np.sign(weights) + 2 * l2 * weights

    return grad_pred, grad_weights

def _binary_cross_entropy(prediction, target, weights=None, l1=0.0, l2=0.0, eps=1e-12):

    prediction = np.clip(prediction, eps, 1.0 - eps)
    loss = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

    if weights is not None:
        reg_term = l1 * np.sum(np.abs(weights)) + l2 * np.sum(weights ** 2)
        loss += reg_term    

    return loss

def _binary_cross_entropy_der(prediction, target, eps=1e-12, weights=None, l1=0.0, l2=0.0):
    prediction = np.clip(prediction, eps, 1.0 - eps)
    grad_pred = - (target / prediction - (1 - target) / (1 - prediction)) / target.shape[0]

    grad_weights = None
    if weights is not None:
        grad_weights = l1 * np.sign(weights) + 2 * l2 * weights
    
    return grad_pred, grad_weights

def _cross_entropy(prediction, target, eps=1e-12):
    prediction = np.clip(prediction, eps, 1.0)
    return -np.sum(target * np.log(prediction))

def _cross_entropy_der(prediction, target, eps=1e-12):
    return - (target / prediction) / target.shape[0]

mse = cost_function(_mse, der=_mse_der)
binary_cross_entropy = cost_function(_binary_cross_entropy, der=_binary_cross_entropy_der)
cross_entropy = cost_function(_cross_entropy, der=_cross_entropy_der)