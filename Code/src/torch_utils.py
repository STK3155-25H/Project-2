# torch_complexity_experiment.py
# PyTorch port of your complexity sweep with resume & device-aware setup.
# Models and results are stored under: output/ComplexityAnalysisTORCH

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# -------------------- Device detection --------------------
def detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        tag = "cuda_" + name.replace(" ", "_").replace("/", "_")
        return torch.device("cuda:0"), name, tag
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS", "mps_Apple"
    return torch.device("cpu"), "CPU", "cpu"


# -------------------- Helpers --------------------
def runge(x, noise_std=0.0, rng=None):
    rng = rng or np.random
    noise = rng.normal(0, noise_std, size=x.shape) if noise_std > 0 else 0.0
    return 1.0 / (1.0 + 25.0 * x**2) + noise

def build_layout(n_hidden: int, width: int):
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]

def mse_torch(yhat, y):
    return F.mse_loss(yhat, y)

def extract_losses_from_history(history, final_noisy_loss, final_clean_loss, mode="avg_last_n", last_n=50):
    val_noisy = float(final_noisy_loss)  # fallback
    val_clean = float(final_clean_loss)  # always report final-on-clean
    vhist = history.get("val_loss")
    if vhist and len(vhist) > 0:
        if mode == "min":
            val_noisy = float(np.nanmin(vhist))
        elif mode == "final":
            val_noisy = float(vhist[-1])
        elif mode == "avg_last_n":
            val_noisy = float(np.mean(vhist[-last_n:]))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return val_noisy, val_clean

def newest_run_dir(base_dir):
    runs = sorted([d for d in os.listdir(base_dir) if d.startswith("run_")])
    return runs[-1] if runs else None

def start_new_run(models_base, output_base):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{current_time}"
    os.makedirs(os.path.join(models_base, run_dir), exist_ok=True)
    os.makedirs(os.path.join(output_base, run_dir), exist_ok=True)
    return run_dir

def has_incomplete_work(run_dir, output_base, activation_names):
    for act in activation_names:
        for suffix in ["noisy", "clean"]:
            p = os.path.join(output_base, run_dir, f"temp_heat_{suffix}_{act}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, index_col='hidden_layers')
                if np.isnan(df.values).any():
                    return True
    return False


# -------------------- Model --------------------
class MLP(nn.Module):
    def __init__(self, layout, act_name):
        super().__init__()
        # Map to match old names for filenames
        # LRELU -> LeakyReLU, RELU -> ReLU, tanh -> Tanh, sigmoid -> Sigmoid
        act_map = {
            "LRELU": nn.LeakyReLU(negative_slope=0.01),
            "RELU": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.act = act_map[act_name]
        self.layers = nn.ModuleList([nn.Linear(layout[i], layout[i+1]) for i in range(len(layout)-1)])
        self.out_act = nn.Identity()  # regression

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return self.out_act(x)


# -------------------- Train one model --------------------
def train_one_model(
    layout, act_name, device,
    X_train_t, y_train_noisy_t, X_val_t, y_val_noisy_t, y_val_clean_t,
    epochs=1500, lr=1e-3, lam_l1=0.0, lam_l2=0.0, batches=100, betas=(0.9, 0.999), seed=314
):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(layout, act_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=lam_l2)

    N = X_train_t.shape[0]
    batch_size = max(1, N // max(1, batches))
    train_dl = DataLoader(TensorDataset(X_train_t, y_train_noisy_t), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}

    try:
        for ep in range(epochs):
            model.train()
            run_loss = 0.0
            for xb, yb in train_dl:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = mse_torch(pred, yb)
                if lam_l1 > 0.0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lam_l1 * l1 / N
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * xb.size(0)
            train_epoch_loss = run_loss / N

            model.eval()
            with torch.no_grad():
                vpred_noisy = model(X_val_t.to(device))
                val_loss_noisy = mse_torch(vpred_noisy, y_val_noisy_t.to(device)).item()
            history["train_loss"].append(train_epoch_loss)
            history["val_loss"].append(val_loss_noisy)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — this model will be retrained from scratch on resume.")
        raise

    model.eval()
    with torch.no_grad():
        vpred_noisy = model(X_val_t.to(device))
        vpred_clean = model(X_val_t.to(device))
        final_noisy = mse_torch(vpred_noisy, y_val_noisy_t.to(device)).item()
        final_clean = mse_torch(vpred_clean, y_val_clean_t.to(device)).item()

    return model, history, final_noisy, final_clean

