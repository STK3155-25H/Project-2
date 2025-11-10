# Makefile for testing experiment and plotting scripts
# Usage:
#   make all          - Run all scripts
#   make autograd     - Run autograd_gradient_check.py
#   make comparison   - Run comparison_NN_vs_linear_methods.py
#   make complexity   - Run complexity_analysis.py
#   make convergence  - Run convergence_and_complexity_comparison.py
#   make gradient     - Run gradient_descent_analysis.py
#   make l1l2         - Run l1_l2_analysis.py
#   make mnist        - Run MINST_complexity.py
#   make testing      - Run testing_specific_model.py
#   make plot_class   - Run plot_clasification_complexity.py
#   make plot_gd      - Run plot_gd_alalysis.py
#   make plot_heat    - Run plot_gdmethods_heatmaps.py
#   make clean        - Clean up generated files (if needed)

PYTHON = python3
EXP_DIR = Code
PLOT_DIR = Code/plotting

all: autograd comparison complexity convergence gradient l1l2 mnist testing plot_class plot_gd plot_heat

autograd:
	$(PYTHON) $(EXP_DIR)/autograd_gradient_check.py

comparison:
	$(PYTHON) $(EXP_DIR)/comparison_NN_vs_linear_methods.py

complexity:
	$(PYTHON) $(EXP_DIR)/complexity_analysis.py

convergence:
	$(PYTHON) $(EXP_DIR)/convergence_and_complexity_comparison.py

gradient:
	$(PYTHON) $(EXP_DIR)/gradient_descent_analysis.py

l1l2:
	$(PYTHON) $(EXP_DIR)/l1_l2_analysis.py

mnist:
	$(PYTHON) $(EXP_DIR)/MINST_complexity.py

testing:
	$(PYTHON) $(EXP_DIR)/testing_specific_model.py

plot_class:
	$(PYTHON) $(PLOT_DIR)/plot_clasification_complexity.py

plot_gd:
	$(PYTHON) $(PLOT_DIR)/plot_gd_alalysis.py

plot_heat:
	$(PYTHON) $(PLOT_DIR)/plot_gdmethods_heatmaps.py

clean:
	@echo "Add clean commands if needed (e.g., rm -rf output/*)"