import numpy as np
import os
SEED = 314

seed = os.environ.get("SEED")

if SEED is not None:
    SEED = int(SEED) 
    print("SEED from env:", SEED)
else:
    SEED = 314
    print("SEED from hard-coded value in file ml_core.py :", SEED)
    print("If you want a specific SEED set the SEED environment variable")
np.random.seed(SEED)
