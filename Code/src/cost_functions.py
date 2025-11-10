import autograd.numpy as np

def CostOLS(target):
    
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func

import autograd.numpy as np
from autograd import grad as autograd_grad, elementwise_grad


def analytical_grad(cost_factory, target):
    """
    Return the analytic gradient function dL/dX for the given cost factory,
    with `target` captured via closure—just like Cost* does.

    The returned function takes a single argument X (model predictions) and
    returns the gradient with the same shape as X.
    """
    eps = 1.0e-9  # small constant for numerical stability

    if cost_factory.__name__ == "CostOLS":
        # L = (1/N) * sum (y - X)^2  -> dL/dX = (2/N) * (X - y)
        N = target.shape[0]
        def g(X):
            return (2.0 / N) * (X - target)
        return g

    elif cost_factory.__name__ == "CostLogReg":
        # Binary cross-entropy on probabilities X in (0,1):
        # L = -(1/N) sum [ y*log(X) + (1-y)*log(1-X) ]
        # dL/dX = -(1/N) * [ y/(X) - (1-y)/(1-X) ]
        N = target.shape[0]
        def g(X):
            return -(1.0 / N) * (target / (X + eps) - (1.0 - target) / (1.0 - X + eps))
        return g

    elif cost_factory.__name__ == "CostCrossEntropy":
        # Multiclass (one-hot) cross-entropy on softmax probs X:
        # L = -(1/target.size) * sum target * log(X)
        # dL/dX = -(1/target.size) * target / X
        Z = target.size
        def g(X):
            return -(1.0 / Z) * (target / (X + eps))
        return g

    else:
        # Fallback: use Autograd to differentiate the scalar loss w.r.t. X
        loss = cost_factory(target)     # bind target -> func(X) -> scalar
        return autograd_grad(loss)
