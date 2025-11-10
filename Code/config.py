# config.py
import os
# Base directories (customize as needed)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Project root
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
# Function for centralized subdir paths (prevents repetition)
def get_output_subdir(subdir: str) -> str:
    """Returns a full path to an output subdirectory."""
    full_path = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(full_path, exist_ok=True) # Create if missing
    return full_path
# Examples of centralized subdirs (add more as needed)
COMPLEXITY_OUTPUT_DIR = get_output_subdir("complexity_analysis")
OLS_VS_FFNN_OUTPUT_DIR = get_output_subdir("OLS_vs_FFNN")
COMPLEXITY_ANALYSIS_OUTPUT_DIR = get_output_subdir("ComplexityAnalysis")
BENCHMARK_OUTPUT_DIR = get_output_subdir("benchmark")
OPTIMIZER_SWEEP_OUTPUT_DIR = get_output_subdir("OptimizerSweep")
L1_L2_ANALYSIS_OUTPUT_DIR = get_output_subdir("l1_l2_analysis")
MNIST_COMPLEXITY_OUTPUT_DIR = get_output_subdir("MNIST_ComplexityAnalysis")
SPECIFIC_MODEL_EVAL_OUTPUT_DIR = get_output_subdir("specific_model_eval")
PLOTS_OUTPUT_DIR = get_output_subdir("plots")
MNIST_CONFUSIONS_OUTPUT_DIR = get_output_subdir("MNIST_confusions")