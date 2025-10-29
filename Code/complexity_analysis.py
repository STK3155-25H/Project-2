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
from src.activation_functions import sigmoid, identity, LRELU,RELU,tanh,softmax

# ---- RUNGE FUNCTION DATA ---- #
def runge(x):
    return 1 / (1 + 25 * x**2)

# Dataset in [-1, 1]
X = np.linspace(-1, 1, 200)
y = runge(X)

# Shape in input column vector shape: (n_samples, n_features)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# ---- Model Settings ---- #
layout = [1, 20, 20, 1]  # More hidden units for better approximation
epochs = 2000
lr = 0.001
lam = 0.0
rho = 0.9   
rho2 = 0.999

scores = {}
activation_func = [sigmoid, LRELU,RELU,tanh,softmax]

for act_func in activation_func:
    net = FFNN(
        dimensions=layout,
        hidden_func=act_func,
        output_func=identity,
        cost_func=CostOLS,
        seed=SEED,
    )
    scheduler = Adam(lr, rho, rho2)
    scores[act_func.__name__] = net.fit(X=X, t=y, scheduler=scheduler, batches=100, epochs=epochs, lam=lam)
# ---- PLOT TRAIN / VAL LOSSES ---- #
# plt.figure(figsize=(9,5))

for name, history in scores.items():
    # prova entrambe le convenzioni di chiave
    train = history.get("train_loss", history.get("train_errors"))
    val   = history.get("val_loss",   history.get("val_errors"))

    if train is not None:
        plt.plot(range(1, len(train)+1), train, linestyle="--", label=f"{name} train")
    if val is not None:
        plt.plot(range(1, len(val)+1), val, label=f"{name} val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training curves per activation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()