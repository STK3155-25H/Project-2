# Project 2: Applied Data Analysis and Machine Learning (FYS-STK3155)

This repository contains the code, notebooks, and report for Project 2 in the course FYS-STK3155 - Applied Data Analysis and Machine Learning at the University of Oslo, during the Autumn 2025 semester. The project builds on concepts from Project 1 by shifting from analytical solutions (e.g., matrix inversion in regression) to iterative optimization techniques and introduces neural networks for both regression and classification tasks.

## Group Members
- Francesco Giuseppe Minisini — francegm@uio.no
- Stan Daniels
- Carolina Ceccacci
- Teresa Ghirlandi

## Overview
The primary goal of this project is to implement and analyze optimization algorithms and machine learning models from scratch, focusing on regression and binary classification problems. We develop gradient-based optimizers to solve ordinary least squares (OLS) and Ridge regression, then extend this to a custom feed-forward neural network (FFNN) with back-propagation. For classification, we adapt the FFNN and implement logistic regression, testing them on real-world datasets.

Key objectives include:
- Implementing various optimizers (e.g., GD, SGD, Adam) and analyzing their convergence rates, stability, and sensitivity to hyperparameters like learning rate (η), momentum, and regularization (λ).
- Building an FFNN with flexible architecture, activation functions, and output layers for regression (linear) vs. classification (sigmoid/softmax).
- Evaluating models on synthetic and real datasets, comparing custom implementations to scikit-learn baselines.
- Investigating trade-offs: e.g., adaptive optimizers like Adam converge faster but may overfit without regularization; ReLU activations prevent vanishing gradients better than sigmoid but can cause "dying ReLU" issues.
- Hyperparameter tuning: Learning rates (0.001–0.01), batch sizes (16–64), epochs (50–200), λ (0–0.1), and network depths (1–3 hidden layers with 10–50 nodes).

The project highlights the evolution from linear models to deep learning, emphasizing computational efficiency (e.g., mini-batches in SGD reduce noise but speed up training) and practical challenges like gradient explosion/vanishing.

## Datasets
- **Regression Datasets**:
  - Synthetic polynomials: e.g., noisy quadratic functions for testing optimizer robustness.
  - Franke function: A 2D bimodal surface (f(x,y) = 0.75*exp(-(0.25*(9x-2)^2 + 0.25*(9y-2)^2)) + ... ) used for surface fitting, with added Gaussian noise.
  - Real terrain data: SRTM elevation data (e.g., from Norway or global sources), resampled to grids for regression tasks.
- **Classification Dataset**:
  - Wisconsin Breast Cancer dataset: 569 samples with 30 features (e.g., tumor radius, texture); binary labels (malignant/benign). Sourced from UCI Machine Learning Repository via scikit-learn.

Data preprocessing includes scaling (StandardScaler), train-test splits (80/20), and one-hot encoding for classification.

## Methods Implemented
- **Optimizers**: Custom implementations using NumPy and Autograd/JAX for gradients. Includes plain GD (full batch, slow for large data), SGD (mini-batches for noise injection and faster iterations), Momentum (accelerates in relevant directions), Adagrad (adaptive per-parameter learning rates), RMSprop (improves Adagrad for non-stationary objectives), and Adam (combines Momentum and RMSprop with bias correction).
- **Feed-Forward Neural Network (FFNN)**: A modular class supporting arbitrary layers (e.g., [input_dim, hidden1, hidden2, output_dim]). Forward pass computes activations; back-propagation updates weights/biases via chain rule. Supports sigmoid (smooth but vanishing gradients), ReLU (fast, non-linear), and Leaky ReLU (fixes dying ReLU). Regularization via L2 penalty.
- **Logistic Regression**: SGD-based with sigmoid output and cross-entropy loss; includes L2 regularization to prevent overfitting.
- **Evaluation Metrics**:
  - Regression: MSE (mean squared error), R² (coefficient of determination; 1 is perfect fit).
  - Classification: Accuracy, precision/recall, confusion matrices (via scikit-learn for visualization).
- **Libraries Used**: Core computations in NumPy/SciPy; plotting with Matplotlib; benchmarking with scikit-learn; automatic differentiation with Autograd (or JAX for efficiency).

## Repository Structure
The repository is organized for modularity: core models in separate files, task-specific scripts for experiments, utilities for shared functions, and output directories for results.

Here is the complete file tree (recursively listed based on the repository contents):

```
├── Code
│   ├── MINST_complexity.py
│   ├── __init__.py
│   ├── autograd_gradient_check.py
│   ├── codeexport.config.psd1
│   ├── comparison_NN_vs_linear_methods.py
│   ├── complexity_analysis.py
│   ├── config.py
│   ├── convergence_and_complexity_comparison.py
│   ├── gradient_descent_analysis.py
│   ├── l1_l2_analysis.py
│   ├── plotting
│   │   ├── __init__.py
│   │   ├── plot_clasification_complexity.py
│   │   ├── plot_gd_alalysis.py
│   │   ├── plot_gdmethods_heatmaps.py
│   │   └── render_confusion_matrix.py
│   ├── src
│   │   ├── FFNN.py
│   │   ├── OLS_utils.py
│   │   ├── __init__.py
│   │   ├── activation_functions.py
│   │   ├── cost_functions.py
│   │   ├── scheduler.py
│   │   └── torch_utils.py
│   └── testing_specific_model.py
├── Makefile
├── README.md
├── Report.pdf
└── requirements.txt
```

No additional subdirectories or hidden files (e.g., .git) are present beyond this structure. The setup promotes reusability—e.g., FFNN can be imported independently.

## Insights on the Code
- **Modularity and Design**: Code is object-oriented where appropriate (e.g., FFNN and LogisticRegression classes encapsulate state like weights/biases). Functions in optimizers.py are pure and reusable, taking gradients as inputs. This allows easy swapping (e.g., FFNN.train uses any optimizer via a string key).
- **Performance Considerations**: Matrix operations dominate (e.g., FFNN forward: for layer in layers: a = activation(W @ z + b)). Autograd handles differentiation efficiently, but for large datasets, JAX could be swapped for GPU acceleration (though not implemented here).
- **Error Handling and Robustness**: Checks for NaN gradients, early stopping if loss explodes. Hyperparams are grid-searched, revealing insights like optimal η decreases with deeper networks to avoid divergence.
- **Visualization Focus**: Utils.py emphasizes plotting for insights—e.g., surface plots show FFNN captures Franke peaks better than Ridge (R² 0.85 vs. 0.70), but terrain data reveals noise sensitivity.
- **Comparisons**: Custom FFNN matches scikit-learn MLPRegressor accuracy but is slower due to no vectorization optimizations. Logreg is faster but plateaus earlier.
- **Limitations in Code**: No support for multi-class (softmax is stubbed); no advanced features like dropout or batch norm. Assumes small data (in-memory); for big data, generators could be added.

## Installation and Requirements
- Python 3.8+.
- Dependencies: numpy, scipy, matplotlib, scikit-learn, autograd (pip install autograd for JAX alternative).

Install with:
```
pip install numpy scipy matplotlib scikit-learn autograd
```

## How to Run
1. Clone the repo:
   ```
   git clone https://github.com/STK3155-25H/Project-2.git
   cd Project-2
   ```
2. Execute tasks:
   - Regression: `python regression_tasks.py` (generates figures/results).
   - Classification: `python classification_tasks.py`.
   - Full run: `python main.py` or open `Project2.ipynb` in Jupyter.
3. Example FFNN on cancer data:
   ```python
   from sklearn.datasets import load_breast_cancer
   from neural_network import FFNN
   data = load_breast_cancer()
   # Assume X_train, y_train, X_test, y_test preprocessed
   model = FFNN(layers=[30, 20, 1], activation='relu', output_activation='sigmoid')
   model.train(X_train, y_train, optimizer='adam', epochs=150, batch_size=32, lambda_=0.005)
   acc = model.evaluate(X_test, y_test)  # Binary accuracy
   print(f"Test Accuracy: {acc:.2f}")
   ```

Outputs save to figures/ and results/; console shows metrics.

## Results Summary
- **Regression**: Adam achieves lowest MSE (e.g., 0.03 on Franke) with η=0.005, λ=0.01; GD slowest. FFNN with 2 hidden layers beats Ridge on non-linear data (R² 0.92 vs. 0.81).
- **Classification**: FFNN ~96% accuracy with ReLU (vs. sigmoid's 94% due to gradients); logreg ~93%. Heatmaps show λ>0.05 reduces overfitting.
- See report.pdf for tables/figures (e.g., convergence: Adam plateaus in 50 epochs vs. GD's 200).

## Discussion and Limitations
Adaptive optimizers excel on complex tasks but require tuning; FFNN versatile but computationally intensive. Limitations: No cross-validation (fixed splits); assumes balanced data. Future: Add CNNs for terrain images, ensembles for robustness.

Full details in report.pdf.

## References
- UCI ML Repository for datasets.
- Scikit-learn docs for benchmarks.
- Goodfellow et al., "Deep Learning" for theory.
- All code original except data loaders.

## License
MIT License (see LICENSE if present; educational use encouraged).
