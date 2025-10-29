import numpy as np
import os
SEED = os.environ.get("SEED")

if SEED is not None:
    SEED = int(SEED) 
    print("SEED from env:", SEED)
else:
    SEED = 314
    print("SEED from hard-coded value in file ml_core.py :", SEED)
    print("If you want a specific SEED set the SEED environment variable")
np.random.seed(SEED)

import matplotlib.pyplot as plt
from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import sigmoid, identity, LRELU

# ---- RUNGE FUNCTION DATA ---- #
def runge(x, noise_std=0.02):
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1 / (1 + 25 * x**2) + noise

# Dataset in [-1, 1]
X = np.linspace(-1, 1, 200)
y = runge(X)

# Shape in input column vector shape: (n_samples, n_features)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# ---- Model Settings ---- #
layout = [1, 20, 20, 1]  # More hidden units for better approximation
epochs = 1
lr = 0.001
lam = 0.0
rho = 0.9   
rho2 = 0.999

net = FFNN(
    dimensions=layout,
    hidden_func=LRELU,
    output_func=identity,
    cost_func=CostOLS,
    seed=SEED,
)

scheduler = Adam(lr, rho, rho2)

# ---- TRAIN ---- #
scores = net.fit(X=X, t=y, scheduler=scheduler, batches=100, epochs=epochs)

# ---- PLOT RESULTS ---- #
y_pred = net.predict(X)

plt.figure(figsize=(8,5))
plt.plot(X, y, label="Runge Function", linewidth=2)
plt.scatter(X, y_pred, s=10, label="NN Approximation")
plt.legend()
plt.title("Runge Function Fit with FFNN")
plt.grid(True)
plt.show()


train_losses = scores["train_errors"]
epochs_axis = np.arange(1, len(train_losses) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_axis, train_losses, label="Train loss")
if "val_errors" in scores:
    plt.plot(epochs_axis, scores["val_errors"], label="Val loss", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss (Cost)")
plt.title("Training Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.yscale("log")
plt.show()