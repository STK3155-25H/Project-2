# page.py
# Confronti "scientifici" tra forme analitiche e Autograd:
# - Cost functions: OLS, LogReg, CrossEntropy (grad wrt predictions)
# - Activation functions: identity, sigmoid, tanh, ReLU, Leaky-ReLU (derivate)
# - Gradient check rete (MSE + L2): backprop manuale vs Autograd
#
# Requisiti: autograd, numpy

import autograd.numpy as np
from autograd import grad as autograd_grad, elementwise_grad
import numpy as onp

# ================================================================
# Cost functions (stesse interfacce del tuo codice)
# ================================================================
def CostOLS(target):
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)
    return func

def CostLogReg(target):
    def func(X):
        eps = 1.0e-9
        return -(1.0 / target.shape[0]) * np.sum(
            target * np.log(X + eps) + (1 - target) * np.log(1 - X + eps)
        )
    return func

def CostCrossEntropy(target):
    # tipicamente target è one-hot e X sono probabilità (softmax)
    def func(X):
        eps = 1.0e-9
        return -(1.0 / target.size) * np.sum(target * np.log(X + eps))
    return func

def analytical_grad(cost_factory, target):
    """
    Restituisce ∂L/∂X (gradiente della loss rispetto alle predizioni X)
    per la corrispondente cost factory, catturando 'target' via closure.
    """
    eps = 1.0e-9

    if cost_factory.__name__ == "CostOLS":
        N = target.shape[0]
        def g(X):
            return (2.0 / N) * (X - target)
        return g

    elif cost_factory.__name__ == "CostLogReg":
        N = target.shape[0]
        def g(X):
            return -(1.0 / N) * (target / (X + eps) - (1.0 - target) / (1.0 - X + eps))
        return g

    elif cost_factory.__name__ == "CostCrossEntropy":
        Z = target.size
        def g(X):
            return -(1.0 / Z) * (target / (X + eps))
        return g

    else:
        loss = cost_factory(target)
        return autograd_grad(loss)

# ================================================================
# Activation functions + derivate (come nel tuo file, con fix tanh)
# ================================================================
def identity(X):
    return X

def sigmoid(X):
    # forma numericamente stabile
    return 1.0 / (1.0 + np.exp(-X))

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    eps = 1.0e-9
    ex = np.exp(X)
    return ex / (np.sum(ex, axis=-1, keepdims=True) + eps)

def tanh(X):
    return np.tanh(X)

def RELU(X):
    return np.where(X > 0, X, 0.0)

def LRELU(X):
    delta = 1.0e-4
    return np.where(X > 0, X, delta * X)

def derivate(func):
    # NB: per softmax la derivata corretta è una matrice (jacobiano);
    # qui testiamo solo funzioni element-wise (non softmax).
    if func.__name__ == "RELU":
        def d(X):
            return np.where(X > 0, 1.0, 0.0)
        return d
    elif func.__name__ == "LRELU":
        def d(X):
            delta = 1.0e-4
            return np.where(X > 0, 1.0, delta)
        return d
    elif func.__name__ == "sigmoid":
        def d(X):
            s = sigmoid(X)
            return s * (1.0 - s)
        return d
    elif func.__name__ == "tanh":
        def d(X):
            t = np.tanh(X)
            return 1.0 - t**2
        return d
    elif func.__name__ == "identity":
        def d(X):
            return np.ones_like(X)
        return d
    else:
        # fallback: derivata element-wise via autograd
        return elementwise_grad(func)

# ================================================================
# Utilities di confronto
# ================================================================
def compare_arrays(a, b, name="", atol=1e-8, rtol=1e-6, verbose=True):
    a_np = onp.asarray(a)
    b_np = onp.asarray(b)
    diff = a_np - b_np
    max_abs = onp.max(onp.abs(diff))
    rel = onp.linalg.norm(diff) / (onp.linalg.norm(a_np) + 1e-12)
    passed = (max_abs <= atol) or (rel <= rtol)
    if verbose:
        print(f"[{name}] max|diff| = {max_abs:.3e}, rel_err = {rel:.3e} -> "
              f"{'PASSED' if passed else 'FAILED'} (atol={atol}, rtol={rtol})")
    return passed, max_abs, rel

# ================================================================
# 1) Test: gradienti delle cost function wrt predizioni
# ================================================================
def run_cost_grad_checks(seed=314, atol=1e-8, rtol=1e-6):
    rng = onp.random.default_rng(seed)
    print("\n=== COST GRADIENT CHECKS (∂L/∂X) ===")

    # -------- OLS --------
    N = 128
    y = rng.normal(size=(N, 1)).astype(onp.float64)
    X = rng.normal(size=(N, 1)).astype(onp.float64)

    loss = CostOLS(np.array(y))
    dL_dX_auto = autograd_grad(loss)(np.array(X))
    dL_dX_anal = analytical_grad(CostOLS, np.array(y))(np.array(X))
    compare_arrays(dL_dX_anal, dL_dX_auto, name="CostOLS")

    # -------- LogReg (binaria) --------
    N = 256
    y_bin = rng.integers(0, 2, size=(N, 1)).astype(onp.float64)
    # pred in (0,1) via sigmoid su gaussiane
    logits = rng.normal(size=(N, 1)).astype(onp.float64)
    Xp = sigmoid(np.array(logits))

    loss = CostLogReg(np.array(y_bin))
    dL_dX_auto = autograd_grad(loss)(np.array(Xp))
    dL_dX_anal = analytical_grad(CostLogReg, np.array(y_bin))(np.array(Xp))
    compare_arrays(dL_dX_anal, dL_dX_auto, name="CostLogReg")

    # -------- CrossEntropy (multiclasse, one-hot) --------
    N, C = 64, 5
    # target one-hot
    idx = rng.integers(0, C, size=(N,))
    y_oh = onp.zeros((N, C), dtype=onp.float64)
    y_oh[onp.arange(N), idx] = 1.0
    # pred = softmax(logits)
    logits = rng.normal(size=(N, C)).astype(onp.float64)
    Xs = softmax(np.array(logits))

    loss = CostCrossEntropy(np.array(y_oh))
    dL_dX_auto = autograd_grad(loss)(np.array(Xs))
    dL_dX_anal = analytical_grad(CostCrossEntropy, np.array(y_oh))(np.array(Xs))
    compare_arrays(dL_dX_anal, dL_dX_auto, name="CostCrossEntropy")

# ================================================================
# 2) Test: derivate delle activation vs Autograd (elementwise)
#     (escludiamo softmax perché richiede il jacobiano completo)
# ================================================================
def run_activation_deriv_checks(seed=2718, atol=1e-8, rtol=1e-6):
    rng = onp.random.default_rng(seed)
    print("\n=== ACTIVATION DERIVATIVE CHECKS (element-wise) ===")

    # Campioni lontani da 0 per ReLU/LReLU (evita punto non differenziabile)
    X = rng.normal(size=(1024,)).astype(onp.float64)
    X = onp.where(onp.abs(X) < 1e-6, X + 1e-3, X)

    act_list = [identity, sigmoid, tanh, RELU, LRELU]

    for act in act_list:
        dan = derivate(act)
        # Autograd elementwise_grad calcola la derivata di out_i wrt x_i
        d_auto = elementwise_grad(act)(np.array(X))
        d_anal = dan(np.array(X))
        compare_arrays(d_anal, d_auto, name=f"d{act.__name__}")

# ================================================================
# 3) Rete: forward/backward + gradient check (come tuo file, con fix __main__)
# ================================================================
def make_runge(n_samples=100, seed=314):
    rng = onp.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = 1.0 / (1.0 + 25.0 * x**2)
    return x.astype(onp.float64), y.astype(onp.float64)

def unpack_theta(theta, d_in, d_hidden, d_out):
    idx = 0
    n_W1 = d_in * d_hidden
    W1 = theta[idx : idx + n_W1].reshape((d_in, d_hidden)); idx += n_W1
    n_b1 = d_hidden
    b1 = theta[idx : idx + n_b1].reshape((1, d_hidden)); idx += n_b1
    n_W2 = d_hidden * d_out
    W2 = theta[idx : idx + n_W2].reshape((d_hidden, d_out)); idx += n_W2
    n_b2 = d_out
    b2 = theta[idx : idx + n_b2].reshape((1, d_out)); idx += n_b2
    assert idx == theta.size
    return W1, b1, W2, b2

def forward_autograd(theta, X, d_in, d_hidden, d_out):
    W1, b1, W2, b2 = unpack_theta(theta, d_in, d_hidden, d_out)
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z2  # regressione

def mse_loss_autograd(theta, X, y, d_in, d_hidden, d_out, lam_l2=0.0):
    y_hat = forward_autograd(theta, X, d_in, d_hidden, d_out)
    mse = np.mean((y_hat - y) ** 2)
    l2 = lam_l2 * np.sum(theta ** 2)
    return mse + l2

def manual_grad(theta, X, y, d_in, d_hidden, d_out, lam_l2=0.0):
    X = onp.asarray(X); y = onp.asarray(y); theta = onp.asarray(theta)
    W1, b1, W2, b2 = unpack_theta(theta, d_in, d_hidden, d_out)
    N = X.shape[0]
    Z1 = X @ W1 + b1
    A1 = onp.tanh(Z1)
    Y_hat = A1 @ W2 + b2
    dY = (2.0 / N) * (Y_hat - y)
    dW2 = A1.T @ dY
    db2 = onp.sum(dY, axis=0, keepdims=True)
    dA1 = dY @ W2.T
    dZ1 = dA1 * (1.0 - onp.tanh(Z1) ** 2)
    dW1 = X.T @ dZ1
    db1 = onp.sum(dZ1, axis=0, keepdims=True)
    dW1 += 2.0 * lam_l2 * W1; db1 += 2.0 * lam_l2 * b1
    dW2 += 2.0 * lam_l2 * W2; db2 += 2.0 * lam_l2 * b2
    grads = onp.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
    assert grads.shape == theta.shape
    return grads

def gradient_check(d_in=1, d_hidden=10, d_out=1, lam_l2=1e-4, n_samples=100, seed=314,
                   atol=1e-8, rtol=1e-6):
    X_raw, y_raw = make_runge(n_samples=n_samples, seed=seed)
    X = np.array(X_raw); y = np.array(y_raw)
    rng = onp.random.default_rng(seed)
    n_params = d_in * d_hidden + d_hidden + d_hidden * d_out + d_out
    theta0 = rng.normal(loc=0.0, scale=0.1, size=n_params)
    loss_grad = autograd_grad(mse_loss_autograd)
    grad_auto = loss_grad(theta0, X, y, d_in, d_hidden, d_out, lam_l2)
    grad_manual = manual_grad(theta0, X_raw, y_raw, d_in, d_hidden, d_out, lam_l2)

    print("\n=== NETWORK GRADIENT CHECK (manual vs autograd) ===")
    print(f"d_in={d_in}, d_hidden={d_hidden}, d_out={d_out}, λ2={lam_l2}, n={n_samples}")
    _, max_abs, rel = compare_arrays(grad_manual, onp.asarray(grad_auto),
                                     name="Backprop vs Autograd",
                                     atol=atol, rtol=rtol, verbose=True)
    idx_worst = int(onp.argmax(onp.abs(grad_manual - onp.asarray(grad_auto))))
    print(f"Worst index = {idx_worst}")
    print(f"grad_manual[{idx_worst}]  = {grad_manual[idx_worst]:.6e}")
    print(f"grad_autograd[{idx_worst}] = {grad_auto[idx_worst]:.6e}")
    if rel < rtol:
        print("Result: PASSED (relative error within tolerance)")
    else:
        print("Result: FAILED (relative error above tolerance)")

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    # Soglie “scientifiche” conservative (puoi stringere se vuoi)
    ATOL = 1e-8
    RTOL = 1e-6

    run_cost_grad_checks(seed=314, atol=ATOL, rtol=RTOL)
    run_activation_deriv_checks(seed=2718, atol=ATOL, rtol=RTOL)
    gradient_check(d_in=1, d_hidden=10, d_out=1, lam_l2=1e-4, n_samples=100, seed=314,
                   atol=ATOL, rtol=RTOL)
