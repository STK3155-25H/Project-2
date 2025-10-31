import os
import re
import numpy as np
import matplotlib.pyplot as plt

from src.FFNN import FFNN
from src.cost_functions import CostOLS
from src.activation_functions import identity, LRELU, RELU, tanh

def parse_model_filename(file_name):
    """
    Parses the model filename to extract n_hidden, width, and activation name.
    Expected format: model_hidden_{n_hidden}_width_{width}_act_{act_name}.npz
    """
    pattern = r"Models\\run_(\d)_(\d)\\model_hidden_(\d+)_width_(\d+)_act_(\w+)\.npz"
    match = re.match(pattern, file_name)
    if not match:
        raise ValueError(f"Invalid filename format: {file_name}")
    n_hidden = int(match.group(3))
    width = int(match.group(4))
    act_name = match.group(5)
    return n_hidden, width, act_name

def get_activation_function(act_name):
    """
    Returns the activation function based on its name.
    """
    act_dict = {
        "LRELU": LRELU,
        "RELU": RELU,
        "tanh": tanh,
    }
    act_func = act_dict.get(act_name)
    if act_func is None:
        raise ValueError(f"Unknown activation function: {act_name}")
    return act_func

def runge_true(x):
    """
    The true Runge function without noise.
    """
    return 1 / (1 + 25 * x**2)

def evaluate_model(file_name, save_plot=False, plot_dir="output"):
    """
    Evaluates the model loaded from the given file_name.
    - Loads the model.
    - Predicts on a grid of x values from -1 to 1.
    - Compares predictions to the true Runge function.
    - Computes the MSE loss.
    - Optionally saves the comparison plot.

    Parameters:
    - file_name (str): The name of the model file (e.g., "model_hidden_3_width_10_act_RELU.npz").
    - save_plot (bool): If True, saves the plot to plot_dir.
    - plot_dir (str): Directory to save the plot if save_plot is True.

    Returns:
    - loss (float): The MSE loss between predictions and true Runge function.
    """
    # Parse filename
    n_hidden, width, act_name = parse_model_filename(file_name)
    
    # Get activation function
    act_func = get_activation_function(act_name)
    
    # Build dimensions
    dims = [1] + [width] * n_hidden + [1]
    
    # Create FFNN instance
    net = FFNN(
        dimensions=tuple(dims),
        hidden_func=act_func,
        output_func=identity,
        cost_func=CostOLS,
    )
    
    # Load weights
    model_path = os.path.join("Models", file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    net.load_weights(model_path)
    
    # Generate evaluation data
    X_eval = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = runge_true(X_eval)
    
    # Predict
    y_pred = net.predict(X_eval)
    
    # Compute loss (MSE)
    loss = np.mean((y_pred - y_true)**2)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.plot(X_eval, y_true, label="True Runge Function", color="blue")
    plt.plot(X_eval, y_pred, label="Model Prediction", color="red", linestyle="--")
    plt.title(f"Model: {file_name}\nMSE Loss: {loss:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    
    if save_plot:
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"evaluation_{file_name.replace('.npz', '')}.png"
        # plt.savefig(os.path.join(plot_dir, plot_filename))
        # print(f"Plot saved to: {os.path.join(plot_dir, plot_filename)}")
    else:
        plt.show()
    
    plt.close()
    
    return loss

if __name__ == "__main__":
    # Example usage: replace with your desired file_name
    example_file = "Models\\run_20251030_011114\model_hidden_5_width_40_act_LRELU.npz"  # Change this to your file
    loss = evaluate_model(example_file, save_plot=True)
    print(f"Evaluation Loss: {loss:.6f}")