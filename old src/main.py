import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from functions import (ReLU, sigmoid, softmax, mse, activation_function, cost_function)

X, y = make_moons(n_samples=500, noise=0.15, random_state=314)
X = X.T                               # (2, N)
y = y.reshape(1, -1)                  # (1, N)

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T

layout = [2, 5, 1]


net = NeuralNetwork(
    layers=layout,
    activation_funcs=(sigmoid, sigmoid),
    activation_ders=(sigmoid.diff, sigmoid.diff),
    cost_fun=mse,
    cost_der=mse.grad()
)


lr = 0.1
lrb = 0.2
epochs = 200

for epoch in range(epochs):
    wg, bg = net.compute_gradient(X_train, y_train)
    net.update_weights((wg, bg), lr, lrb)
    if epoch % 20 == 0:
        preds = net.predict(X_train)
        loss = net.cost(preds, y_train)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")


xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()].T
zz = net.predict(grid)
zz = (zz > 0.5).astype(int)
plt.contourf(xx, yy, zz.reshape(xx.shape), alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[0], X_test[1], c=y_test.flatten(), cmap="coolwarm", edgecolor="k")
plt.title("NeuralNetwork decision boundary on make_moons")
plt.show()
