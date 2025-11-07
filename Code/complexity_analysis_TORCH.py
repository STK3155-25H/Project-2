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


# -------------------- Main --------------------
if __name__ == "__main__":
    # ---- Config ----
    SEED = int(os.environ.get("SEED", 314))
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED); torch.manual_seed(SEED)

    # Data
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    noise_global = 0.00
    noise_train_extra = 0.03
    y_noisy = runge(X, noise_std=noise_global, rng=rng).reshape(-1, 1)
    y_clean = runge(X, noise_std=0.0, rng=rng).reshape(-1, 1)

    # Training settings
    epochs = 1500
    lr = 0.001
    lam_l1 = 0.0
    lam_l2 = 0.0
    rho = 0.9
    rho2 = 0.999
    batches = 100

    # EXACT activation names to match your filenames
    activation_funcs = ["LRELU", "RELU", "tanh", "sigmoid"]
    n_hidden_list = list(range(1, 6))             # 1..5
    n_perceptrons_list = [2 * i for i in range(1, 21)]  # 2..40 step 2

    VAL_LOSS_MODE = "avg_last_n"
    LAST_N = 50

    # ---- Device ----
    device, device_name, device_tag = detect_device()
    print(f"[Device] {device_name} ({device.type})")

    # ---- Folders (ALL under output/ComplexityAnalysisTORCH) ----
    ROOT = os.path.join("output", "ComplexityAnalysisTORCH")
    MODELS_BASE = os.path.join(ROOT, "Models")    # models live here
    OUTPUT_BASE = ROOT                             # results (csv/png/temp) live here
    os.makedirs(MODELS_BASE, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ---- Run handling ----
    last_run = newest_run_dir(MODELS_BASE)
    if last_run and has_incomplete_work(last_run, OUTPUT_BASE, activation_funcs):
        run_dir = last_run
        is_continuing = True
        print(f"[Resume] {run_dir}")
    else:
        run_dir = start_new_run(MODELS_BASE, OUTPUT_BASE)
        is_continuing = False
        print(f"[Start] {run_dir}")

    # Config file in models folder
    config_path = os.path.join(MODELS_BASE, run_dir, "config.json")
    if is_continuing and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        SEED = int(config["SEED"])
        np.random.seed(SEED); torch.manual_seed(SEED)
        epochs = int(config["epochs"]); lr = float(config["lr"])
        lam_l1 = float(config["lam_l1"]); lam_l2 = float(config["lam_l2"])
        rho = float(config["rho"]); rho2 = float(config["rho2"])
        batches = int(config["batches"])
        activation_funcs = list(config["activation_funcs"])
        n_hidden_list = list(config["n_hidden_list"])
        n_perceptrons_list = list(config["n_perceptrons_list"])
        VAL_LOSS_MODE = config["VAL_LOSS_MODE"]
        LAST_N = int(config.get("LAST_N", LAST_N))
        noise_global = float(config["noise_global"])
        noise_train_extra = float(config["noise_train_extra"])
        # regenerate targets with the reloaded seed/noise
        y_noisy = runge(X, noise_std=noise_global, rng=np.random.default_rng(SEED)).reshape(-1, 1)
        y_clean = runge(X, noise_std=0.0, rng=np.random.default_rng(SEED)).reshape(-1, 1)
    else:
        config = {
            "SEED": SEED,
            "epochs": epochs,
            "lr": lr,
            "lam_l1": lam_l1,
            "lam_l2": lam_l2,
            "rho": rho,
            "rho2": rho2,
            "batches": batches,
            "activation_funcs": activation_funcs,
            "n_hidden_list": n_hidden_list,
            "n_perceptrons_list": n_perceptrons_list,
            "VAL_LOSS_MODE": VAL_LOSS_MODE,
            "LAST_N": LAST_N,
            "noise_global": noise_global,
            "noise_train_extra": noise_train_extra,
            "device": device_name
        }
        os.makedirs(os.path.join(MODELS_BASE, run_dir), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    # ---- Split, add extra train noise, tensors ----
    X_train, X_val, y_train_noisy, y_val_noisy = train_test_split(
        X, y_noisy, test_size=0.2, random_state=SEED, shuffle=True
    )
    y_train_noisy = y_train_noisy + np.random.normal(0, noise_train_extra, y_train_noisy.shape)
    y_val_clean = runge(X_val, noise_std=0.0, rng=np.random.default_rng(SEED))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_noisy_t = torch.tensor(y_train_noisy, dtype=torch.float32)
    y_val_noisy_t = torch.tensor(y_val_noisy, dtype=torch.float32)
    y_val_clean_t = torch.tensor(y_val_clean, dtype=torch.float32)

    betas = (rho, rho2)
    interrupted = False

    # ---- Main sweep ----
    for act_name in activation_funcs:
        # CSV/temps live under OUTPUT_BASE/run_dir
        csv_noisy = os.path.join(OUTPUT_BASE, run_dir, f"val_loss_data_noisy_{act_name}.csv")
        csv_clean = os.path.join(OUTPUT_BASE, run_dir, f"val_loss_data_clean_{act_name}.csv")
        temp_noisy = os.path.join(OUTPUT_BASE, run_dir, f"temp_heat_noisy_{act_name}.csv")
        temp_clean = os.path.join(OUTPUT_BASE, run_dir, f"temp_heat_clean_{act_name}.csv")

        # Skip if finished
        if os.path.exists(csv_noisy) and os.path.exists(csv_clean):
            print(f"[{act_name}] already complete. Skipping.")
            continue

        # Load/create temp heatmaps
        if os.path.exists(temp_noisy):
            dfN = pd.read_csv(temp_noisy, index_col='hidden_layers')
            heat_noisy = dfN.values if list(dfN.index.astype(int)) == n_hidden_list and list(dfN.columns.astype(int)) == n_perceptrons_list \
                         else np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)
        else:
            heat_noisy = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)

        if os.path.exists(temp_clean):
            dfC = pd.read_csv(temp_clean, index_col='hidden_layers')
            heat_clean = dfC.values if list(dfC.index.astype(int)) == n_hidden_list and list(dfC.columns.astype(int)) == n_perceptrons_list \
                         else np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)
        else:
            heat_clean = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)

        try:
            for i_h, n_hidden in enumerate(n_hidden_list):
                for j_w, width in enumerate(n_perceptrons_list):
                    if not np.isnan(heat_noisy[i_h, j_w]):
                        continue

                    layout = build_layout(n_hidden, width)

                    # *** EXACT model filename format preserved (.npz) ***
                    model_filename = f"model_hidden_{n_hidden}_width_{width}_act_{act_name}.npz"
                    model_path = os.path.join(MODELS_BASE, run_dir, model_filename)
                    done_marker = model_path + ".done"

                    print(f"[Train] {model_filename} on {device_name}")
                    model, history, final_noisy, final_clean = train_one_model(
                        layout=layout, act_name=act_name, device=device,
                        X_train_t=X_train_t, y_train_noisy_t=y_train_noisy_t,
                        X_val_t=X_val_t, y_val_noisy_t=y_val_noisy_t, y_val_clean_t=y_val_clean_t,
                        epochs=epochs, lr=lr, lam_l1=lam_l1, lam_l2=lam_l2,
                        batches=batches, betas=betas, seed=SEED
                    )

                    # Save weights (extension kept as .npz as requested)
                    torch.save(model.state_dict(), model_path)
                    with open(done_marker, "w") as f:
                        f.write("ok")

                    val_noisy_est, val_clean_est = extract_losses_from_history(
                        history, final_noisy, final_clean, mode=VAL_LOSS_MODE, last_n=LAST_N
                    )
                    heat_noisy[i_h, j_w] = val_noisy_est
                    heat_clean[i_h, j_w] = val_clean_est

                    # Persist temps
                    dfN = pd.DataFrame(heat_noisy, index=n_hidden_list, columns=n_perceptrons_list)
                    dfN.index.name = 'hidden_layers'; dfN.columns.name = 'neurons_per_layer'
                    os.makedirs(os.path.join(OUTPUT_BASE, run_dir), exist_ok=True)
                    dfN.to_csv(temp_noisy)

                    dfC = pd.DataFrame(heat_clean, index=n_hidden_list, columns=n_perceptrons_list)
                    dfC.index.name = 'hidden_layers'; dfC.columns.name = 'neurons_per_layer'
                    dfC.to_csv(temp_clean)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Current model will retrain from scratch next time.")
            interrupted = True

        if not interrupted:
            # Finalize CSVs & plots
            if not os.path.exists(csv_noisy):
                dfN = pd.DataFrame(heat_noisy, index=n_hidden_list, columns=n_perceptrons_list)
                dfN.index.name = 'hidden_layers'; dfN.columns.name = 'neurons_per_layer'
                dfN.to_csv(csv_noisy)
                if os.path.exists(temp_noisy): os.remove(temp_noisy)

                plt.figure(figsize=(10, 5))
                im = plt.imshow(heat_noisy, aspect="auto", origin="upper", interpolation="nearest")
                plt.colorbar(im, label="Validation Loss (MSE) - Noisy")
                plt.title(f"Validation Loss Heatmap (Noisy) — activation: {act_name} — {device_name}")
                plt.xlabel("Neurons per hidden layer")
                plt.ylabel("Number of hidden layers")
                plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list, rotation=45)
                plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_BASE, run_dir, f"val_loss_heatmap_noisy_{act_name}.png"), dpi=160)
                plt.close()

            if not os.path.exists(csv_clean):
                dfC = pd.DataFrame(heat_clean, index=n_hidden_list, columns=n_perceptrons_list)
                dfC.index.name = 'hidden_layers'; dfC.columns.name = 'neurons_per_layer'
                dfC.to_csv(csv_clean)
                if os.path.exists(temp_clean): os.remove(temp_clean)

                plt.figure(figsize=(10, 5))
                im = plt.imshow(heat_clean, aspect="auto", origin="upper", interpolation="nearest")
                plt.colorbar(im, label="Validation Loss (MSE) - Clean")
                plt.title(f"Validation Loss Heatmap (Clean) — activation: {act_name} — {device_name}")
                plt.xlabel("Neurons per hidden layer")
                plt.ylabel("Number of hidden layers")
                plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list, rotation=45)
                plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_BASE, run_dir, f"val_loss_heatmap_clean_{act_name}.png"), dpi=160)
                plt.close()
        else:
            break

    print("Done or paused. Resume will continue from the first missing cell.")
