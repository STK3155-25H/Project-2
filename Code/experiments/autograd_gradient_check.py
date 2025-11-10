from ..config import MODELS_DIR, get_output_subdir
BASE_DIR = MODELS_DIR
OUTPUT_DIR = get_output_subdir("complexity_analysis")

import numpy as onp  
from autograd import grad
import autograd.numpy as np 


# ---------------------------------------------------------------------
# 1. Generate test data (Runge function)
# ---------------------------------------------------------------------
def make_runge(n_samples=100, seed=314):
    rng = onp.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = 1.0 / (1.0 + 25.0 * x**2)
    return x.astype(onp.float64), y.astype(onp.float64)


# ---------------------------------------------------------------------
# 2. Feed-forward network definition and parameter unpacking
# ---------------------------------------------------------------------
def unpack_theta(theta, d_in, d_hidden, d_out):
    """
    Splits the parameter vector `theta` into:
        W1 (d_in x d_hidden), b1 (1 x d_hidden),
        W2 (d_hidden x d_out), b2 (1 x d_out)
    """
    idx = 0
    n_W1 = d_in * d_hidden
    W1 = theta[idx : idx + n_W1].reshape((d_in, d_hidden))
    idx += n_W1

    n_b1 = d_hidden
    b1 = theta[idx : idx + n_b1].reshape((1, d_hidden))
    idx += n_b1

    n_W2 = d_hidden * d_out
    W2 = theta[idx : idx + n_W2].reshape((d_hidden, d_out))
    idx += n_W2

    n_b2 = d_out
    b2 = theta[idx : idx + n_b2].reshape((1, d_out))
    idx += n_b2

    assert idx == theta.size
    return W1, b1, W2, b2


def forward_autograd(theta, X, d_in, d_hidden, d_out):
    """
    Forward pass using autograd.numpy (np).
    This version is differentiable by Autograd.
    """
    W1, b1, W2, b2 = unpack_theta(theta, d_in, d_hidden, d_out)
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z2  # linear output for regression


def mse_loss_autograd(theta, X, y, d_in, d_hidden, d_out, lam_l2=0.0):
    """
    MSE + L2 regularization loss written with autograd.numpy.
    """
    y_hat = forward_autograd(theta, X, d_in, d_hidden, d_out)
    mse = np.mean((y_hat - y) ** 2)
    l2 = lam_l2 * np.sum(theta ** 2)
    return mse + l2


# ---------------------------------------------------------------------
# 3. Manual gradient computation (explicit backprop)
# ---------------------------------------------------------------------
def manual_grad(theta, X, y, d_in, d_hidden, d_out, lam_l2=0.0):
    """
    Explicitly computes the gradient of MSE + L2 w.r.t. all parameters.
    Returns a flattened gradient vector with the same shape as theta.
    """
    X = onp.asarray(X)
    y = onp.asarray(y)
    theta = onp.asarray(theta)

    W1, b1, W2, b2 = unpack_theta(theta, d_in, d_hidden, d_out)
    N = X.shape[0]

    # Forward
    Z1 = X @ W1 + b1
    A1 = onp.tanh(Z1)
    Y_hat = A1 @ W2 + b2

    # dMSE/dY_hat = 2/N * (Y_hat - y)
    dY = (2.0 / N) * (Y_hat - y)

    # Gradients for W2 and b2
    dW2 = A1.T @ dY
    db2 = onp.sum(dY, axis=0, keepdims=True)

    # Backprop to hidden layer
    dA1 = dY @ W2.T
    dZ1 = dA1 * (1.0 - onp.tanh(Z1) ** 2)

    dW1 = X.T @ dZ1
    db1 = onp.sum(dZ1, axis=0, keepdims=True)

    # Add L2 regularization term
    dW1 += 2.0 * lam_l2 * W1
    db1 += 2.0 * lam_l2 * b1
    dW2 += 2.0 * lam_l2 * W2
    db2 += 2.0 * lam_l2 * b2

    # Flatten all gradients into a single vector
    grads = onp.concatenate(
        [dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()]
    )
    assert grads.shape == theta.shape
    return grads


# ---------------------------------------------------------------------
# 4. Gradient check routine
# ---------------------------------------------------------------------
def gradient_check(
    d_in=1,
    d_hidden=10,
    d_out=1,
    lam_l2=1e-4,
    n_samples=100,
    seed=314,
):
    # Generate data
    X_raw, y_raw = make_runge(n_samples=n_samples, seed=seed)
    X = np.array(X_raw)
    y = np.array(y_raw)

    # Initialize parameters
    rng = onp.random.default_rng(seed)
    n_params = d_in * d_hidden + d_hidden + d_hidden * d_out + d_out
    theta0 = rng.normal(loc=0.0, scale=0.1, size=n_params)

    # Gradient from Autograd
    loss_grad = grad(mse_loss_autograd)
    grad_auto = loss_grad(theta0, X, y, d_in, d_hidden, d_out, lam_l2)

    # Gradient from manual backprop
    grad_manual = manual_grad(theta0, X_raw, y_raw, d_in, d_hidden, d_out, lam_l2)

    # Compare
    diff = grad_manual - onp.asarray(grad_auto)
    max_abs_diff = onp.max(onp.abs(diff))
    rel_error = onp.linalg.norm(diff) / (onp.linalg.norm(grad_manual) + 1e-12)

    print("-------------------------------------------------")
    print("Gradient check: Manual Backprop vs Autograd")
    print(f"d_in      = {d_in}")
    print(f"d_hidden  = {d_hidden}")
    print(f"d_out     = {d_out}")
    print(f"lambda L2 = {lam_l2}")
    print(f"n_samples = {n_samples}")
    print("-------------------------------------------------")
    print(f"||grad_manual||   = {onp.linalg.norm(grad_manual):.6e}")
    print(f"||grad_autograd|| = {onp.linalg.norm(grad_auto):.6e}")
    print(f"Max |diff|        = {max_abs_diff:.6e}")
    print(f"Relative error    = {rel_error:.6e}")

    idx_worst = onp.argmax(onp.abs(diff))
    print(f"Worst mismatch index: {idx_worst}")
    print(
        f"grad_manual[{idx_worst}]  = {grad_manual[idx_worst]:.6e}, "
        f"grad_autograd[{idx_worst}] = {grad_auto[idx_worst]:.6e}"
    )

    if rel_error < 1e-6:
        print("Gradient check PASSED (relative error < 1e-6)")
    else:
        print("Gradient check FAILED (relative error >= 1e-6)")


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    gradient_check()
