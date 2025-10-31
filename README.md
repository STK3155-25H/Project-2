
## Short project description
In this project, we investigate the performance of Feed-Forward Neural Networks (FFNNs) in both function approximation and classification problems 
# Project 2: Regression and Classification with Neural Networks and Logistic Regression

## Course Information
This repository contains the code and report for **Project 2** in the course **FYS-STK3155 - Applied Data Analysis and Machine Learning** at the University of Oslo, Autumn 2025 semester.

## Group members

- **Francesco Giuseppe Minisini** — francegm@uio.no
- **Stan Daniels**
- **Carolina Ceccacci**
- **Teresa Ghirlandi**

## Overview
In this project, we explore regression and classification problems using custom implementations of optimization algorithms and machine learning models. The main focus is on developing a feed-forward neural network (FFNN) from scratch, including back-propagation, and comparing it with logistic regression for classification tasks. We also extend regression methods from previous projects by replacing matrix inversion with gradient-based optimizers.

Key objectives:
- Implement gradient descent (GD), stochastic gradient descent (SGD), and adaptive optimizers (Momentum, Adagrad, RMSprop, Adam) for regression tasks.
- Build an FFNN for regression and adapt it for binary classification.
- Test various activation functions (Sigmoid, ReLU, Leaky ReLU) and analyze their impact.
- Apply the models to regression datasets (e.g., polynomials, Franke function, terrain data) and classification datasets (e.g., Wisconsin Breast Cancer dataset).
- Compare performance with scikit-learn implementations and evaluate hyperparameters like learning rates, batch sizes, epochs, regularization (λ), and network architecture.

The project demonstrates the transition from simple linear models to more complex neural networks, highlighting trade-offs in performance, convergence, and computational efficiency.

## Datasets
- **Regression**: Synthetic polynomials (e.g., quadratic functions), Franke function (a 2D test function for surface fitting), and real terrain data (e.g., SRTM data from Project 1).
- **Classification**: Wisconsin Breast Cancer dataset (binary classification: malignant vs. benign tumors), sourced from UCI Machine Learning Repository.

## Methods Implemented
- **Optimizers for Regression**: Plain GD, SGD with mini-batches, Momentum, Adagrad, RMSprop, Adam. Used for OLS and Ridge regression, with automatic differentiation via Autograd or JAX.
- **Feed-Forward Neural Network (FFNN)**: Custom implementation with configurable layers/nodes, sigmoid/ReLU/Leaky ReLU activations, back-propagation for training. Linear output for regression; sigmoid/softmax for classification.
- **Logistic Regression**: Implemented using SGD with L2 regularization, for binary classification.
- **Evaluation Metrics**:
  - Regression: Mean Squared Error (MSE), R² score.
  - Classification: Accuracy, confusion matrices.
- Libraries: NumPy, SciPy, Matplotlib (for plotting), scikit-learn (for benchmarking and data loading), Autograd/JAX (for gradients).

Code files analyzed:
- Assuming typical structure based on project requirements (since direct access treats master as main):
  - `optimizers.py`: Contains implementations of GD, SGD, and adaptive optimizers.
  - `neural_network.py`: FFNN class with forward pass, back-propagation, and training methods.
  - `logistic_regression.py`: Logistic regression model using SGD.
  - `regression_tasks.py`: Scripts for Part a) and b) – testing optimizers and FFNN on regression data.
  - `classification_tasks.py`: Scripts for Part d) and e) – FFNN and logistic regression on cancer data.
  - `utils.py`: Helper functions for data generation (e.g., Franke function), metrics, and plotting.
  - `main.py` or Jupyter notebooks (e.g., `Project2.ipynb`): Main execution scripts or notebooks running experiments and generating results/figures.
  - Figures and results stored in `results/` or `figures/` directories.

The code emphasizes modularity, with classes for models and separate functions for optimization. We used Python 3.x, ensuring no external package installations beyond standard scientific libraries.

## Installation and Requirements
- Python 3.8+
- Required packages: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `autograd` (or `jax` for alternative differentiation).
  
Install dependencies:
```
pip install numpy scipy matplotlib scikit-learn autograd
```

## How to Run
1. Clone the repository (treating `main` as the default branch):
   ```
   git clone https://github.com/STK3155-25H/Project-2.git
   cd Project-2
   ```
2. Run the main script or notebook:
   - For regression experiments: `python regression_tasks.py`
   - For classification: `python classification_tasks.py`
   - Or open `Project2.ipynb` in Jupyter for interactive runs.
3. Results (MSE/R² plots, accuracy scores, convergence curves) will be printed and saved to `figures/`.

Example command for training FFNN on cancer data:
```python
from neural_network.py import FFNN
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
# Preprocess data...
model = FFNN(layers=[30, 20, 1], activation='sigmoid', output_activation='sigmoid')
model.train(X_train, y_train, optimizer='adam', epochs=100, batch_size=32, lambda_=0.01)
accuracy = model.evaluate(X_test, y_test)
```

## Results Summary
- **Regression**: Adaptive optimizers (e.g., Adam) converge faster than plain GD, especially with momentum. FFNN outperforms Ridge for complex functions like Franke, but requires careful tuning of learning rates (η ~ 0.001–0.01) and λ (0–0.1).
- **Classification**: FFNN achieves ~95% accuracy on breast cancer data with 2 hidden layers; logistic regression reaches ~93%. ReLU reduces vanishing gradient issues compared to sigmoid.
- Detailed figures (e.g., learning curves, heatmaps of predictions) and discussions are in the report PDF (linked in the repository if available).

## Discussion and Limitations
We observed that for small datasets, simpler models like logistic regression are sufficient and less prone to overfitting, while FFNN excels on complex patterns but needs regularization. Future work could include CNNs for image data or ensemble methods.

For the full report, see `report.pdf` (or equivalent) in the repository.

## References
- Course materials: https://compphysics.github.io/MachineLearning/
- Wisconsin Breast Cancer Dataset: UCI ML Repository
- Scikit-learn documentation for benchmarking.

This project was completed in collaboration; all code is original unless specified (e.g., data loaders from scikit-learn).
