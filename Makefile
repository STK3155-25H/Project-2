# Makefile for testing experiment and plotting scripts
# Usage:
#   make all          - Run all scripts
#   make autograd     - Run autograd_gradient_check
#   make comparison   - Run comparison_NN_vs_linear_methods
#   make complexity   - Run complexity_analysis
#   make convergence  - Run convergence_and_complexity_comparison
#   make gradient     - Run gradient_descent_analysis
#   make l1l2         - Run l1_l2_analysis
#   make mnist        - Run MINST_complexity
#   make testing      - Run testing_specific_model
#   make plot_class   - Run plot_clasification_complexity
#   make plot_gd      - Run plot_gd_alalysis
#   make plot_heat    - Run plot_gdmethods_heatmaps
#   make clean        - Clean up generated files (if needed)

PYTHON = python3 -m
EXP_DIR = Code.experiments
PLOT_DIR = Code.plotting

all: autograd comparison complexity convergence gradient l1l2 mnist testing plot_class plot_gd plot_heat

autograd:
	$(PYTHON) $(EXP_DIR).autograd_gradient_check

comparison:
	$(PYTHON) $(EXP_DIR).comparison_NN_vs_linear_methods

complexity:
	$(PYTHON) $(EXP_DIR).complexity_analysis

convergence:
	$(PYTHON) $(EXP_DIR).convergence_and_complexity_comparison

gradient:
	$(PYTHON) $(EXP_DIR).gradient_descent_analysis

l1l2:
	$(PYTHON) $(EXP_DIR).l1_l2_analysis

mnist:
	$(PYTHON) $(EXP_DIR).MINST_complexity

testing:
	$(PYTHON) $(EXP_DIR).testing_specific_model

plot_class:
	$(PYTHON) $(PLOT_DIR).plot_clasification_complexity

plot_gd:
	$(PYTHON) $(PLOT_DIR).plot_gd_alalysis

plot_heat:
	$(PYTHON) $(PLOT_DIR).plot_gdmethods_heatmaps

clean:
	@echo "Add clean commands if needed (e.g., rm -rf output.*)"