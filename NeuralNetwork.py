import numpy as np
import sympy as sy
import scipy as scp
from typing import List, Dict, Tuple, Callable

from autograd import elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from functions import (ReLU, sigmoid, softmax, mse, activation_function, cost_function)
der = Callable[[np.number | np.ndarray], np.ndarray]

np.random.seed(314)

class Layer:
    def __init__(self, id ,weights ,bias ,name ):
        self.id: int = id
        self.weights: np.ndarray = weights
        self.bias: np.ndarray = bias
        self.last_activation: np.ndarray = None
        self.last_pre_activation: np.ndarray = None
        self.name:str = name
        self.last_weights_gradient: np.ndarray = None
        self.last_bias_gradient: np.ndarray = None

class NeuralNetwork:
    def __init__(
        self,
        layers: list[Layer] | list[int]= None,
        activation_funcs: Tuple[activation_function] | activation_function = [sigmoid],
        activation_ders: Tuple[der] | der = None,
        cost_fun: cost_function = mse,
        cost_der: Callable[[np.ndarray],np.ndarray] = None,
    ):
        self.layers = None
        if isinstance(layers, list) and (not layers or isinstance(layers[0], Layer)):
            self.layers = layers
            self.layout = self.get_layout()
        elif isinstance(layers, list) and (not layers or isinstance(layers[0], int)):
            layout = layers
            self.layout = layout
            self.layers = self.create_layers(layout)
        else:
            raise TypeError("layers deve essere List[Layer] oppure List[int]")
            
        self.activation_mode = len(activation_funcs)
        if(self.activation_mode > 2 | self.activation_mode < 0 | self.activation_mode != len(activation_ders)): raise RuntimeError('Incorrect parameters')
        elif(self.activation_mode == 2):
            self.hidden_activation = activation_funcs[0]
            self.output_activation = activation_funcs[1]
        elif():
            self.activation_funcs = activation_funcs
            
        if activation_ders is None:
            self.activation_ders = self.activation_funcs.diff()    
        else: 
            self.activation_ders = activation_ders
            self.cost_fun = cost_fun
        if cost_der is None:
            self.cost_fun = cost_fun.grad()
        else:
            self.cost_der = cost_der
        

    

    def predict(self, inputs):
        for layer in self.layers:
            z = layer.weights @ inputs + layer.bias
            if layer == self.layers[-1]:
                a = self.output_activation(z)
            else:
                a = self.hidden_activation(z)
            inputs = a
        return inputs

    def cost(self, inputs, targets):
        return self.cost_fun(inputs, targets)

    def _feed_forward_saver(self, inputs):
        activations= [] 
        pre_activations = []
        
        for layer in self.layers:
            z = layer.weights @ inputs + layer.bias
            layer.last_pre_activation = z
            pre_activations.append(z)
            
            if layer == self.layers[-1]:
                a = self.output_activation(z)
            else:
                a = self.hidden_activation(z)
            layer.last_activation = a
            activations.append(a)
            
            inputs = a
        return activations, pre_activations

    def compute_gradient(self, inputs, targets):
        
        self._feed_forward_saver(inputs)
        
        m = inputs.shape[1] if inputs.ndim == 2 else 1
        
        d_act_hidden = self.hidden_activation.diff
        d_act_out = self.output_activation.diff
        
        dC_da_L = 2 * (self.layers[-1].last_activation - targets) / m
        delta = dC_da_L * d_act_out(self.layers[-1].last_pre_activation) 
        
        self.layers[-1].last_weights_gradient = (delta @ self.layers[-2].last_activation.T)    
        self.layers[-1].last_bias_gradient = np.sum(delta, axis=1, keepdims=True)

        for i in range(len(self.layers)-2, -1, -1):
            W_next = self.layers[i+1].weights     
            z_ell = self.layers[i+1].last_pre_activation
            a_prev = self.layers[i-1].last_activation

            delta = (W_next.T @ delta) * d_act_hidden(z_ell)
            
            self.layers[i].last_weights_gradient = (delta @ a_prev.T)
            self.layers[i].last_bias_gradient = np.sum(delta, axis=1, keepdims=True)

        # self.layers[0].last_weights_gradient = delta @ inputs.T
        # self.layers[0].last_bias_gradient = delta
        
        weights_gradient = [layer.last_weights_gradient for layer in self.layers]
        bias_gradient  = [layer.last_bias_gradient for layer in self.layers]
        return weights_gradient, bias_gradient

    def update_weights(self, layer_grads, lr, lrb):
        for i in range(len(self.layers)):
            self.layers[i].weights -= lr*self.layers[i].last_weights_gradient
            self.layers[i].bias -= lrb*self.layers[i].last_bias_gradient

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass
    
    
    def get_layout(self):
        if self.layout is not None:
            return self.layout
        else:
            layout = [self.layers[0].weights.shape[1]]
            for layer in self.layers:
                layout.append(layer.weights.shape[0])
            return layout

    def create_layers(self, layout: list[int] = None):
        if layout is None:
            layout = self.layout
        layers = []
        for i in range(len(layout)-1):
            W = np.random.randn(layout[i+1], layout[i]) * 0.01
            b = np.random.randn(layout[i+1], 1) * 0.01    
            if i == 0:
                name = "Input Layer"
            elif i >0 and i < len(layout)-2:
                name = f"Hidden Layer {i}"
            else:
                name = "Output Layer"
            layers.append(Layer(id=i,name=name,weights=W,bias=b))
        return layers