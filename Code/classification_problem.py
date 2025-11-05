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


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.FFNN import FFNN
from src.scheduler import Scheduler,Momentum, Adam
from src.cost_functions import CostCrossEntropy, CostLogReg
from src.activation_functions import sigmoid 

X, y = make_moons(n_samples=500, noise=0.15, random_state=314)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

layout = [2, 20,20, 1]
epochs = 200
lr = 0.01
rho = 0.9
lam1 = 0.0
lam2 = 0.0
rho2 = 0.999

net = FFNN(
        dimensions = layout,
        hidden_func = sigmoid,
        output_func= sigmoid,
        cost_func =  CostLogReg,
        seed= SEED,
)

scheduler = Adam(lr, rho, rho2)
net.fit(X=X_train, t=y_train, scheduler=scheduler, batches=5, epochs=100, lam_l1=lam1, lam_l2=lam2, X_val= X_test,t_val=y_test)

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = net.predict(grid)
zz = (zz > 0.5).astype(int)
plt.contourf(xx, yy, zz.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), cmap="coolwarm", edgecolor="k")
plt.title("NeuralNetwork decision boundary on make_moons")
plt.show()
