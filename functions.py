import sympy as sy
import scipy as scp
from typing import List, Dict, Tuple, Callable, Any
import autograd.numpy as np
import numpy as npy
weights_n_biases = Tuple[np.ndarray, np.ndarray]
layer = Dict[str,weights_n_biases]

# x, y = sy.symbols('x,y')

from autograd import elementwise_grad, grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



class activation_function:
    def __init__(self, expr: Callable[[np.ndarray|np.number], np.ndarray|np.number]):
        self.func = expr
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


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


ReLU = activation_function(_ReLU)
sigmoid = activation_function(_sigmoid)
softmax = activation_function(_softmax)
softmax_vec = activation_function(_softmax_vec)

class cost_function:
    def __init__(self, expr: Callable[[np.ndarray|np.number , np.ndarray|np.number], np.ndarray|np.number]):
        self.func = expr
    def __call__(self, prediction:np.ndarray|np.number = None  ,target: np.ndarray|np.number = None):
        if target is None or prediction is None:
            return self.func
        else : 
            return self.func(prediction, target)
        
    def grad(self, value: np.ndarray|np.number = None):
        diff = grad(self.func, 0) 
        if value is None:
            return diff
        else: return diff(value)
        
def _mse( prediction, target):
    return np.mean((prediction - target)**2)

def _cross_entropy(prediction, target, eps=1e-12):
    prediction = np.clip(prediction, eps, 1.0)
    return -np.sum(target * np.log(prediction))

mse = cost_function(_mse)