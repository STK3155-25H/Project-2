import numpy as np
import os
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import sigmoid, identity, LRELU, RELU, tanh, softmax

# Mappatura delle funzioni di attivazione
act_func_map = {
    'sigmoid': sigmoid,
    'identity': identity,
    'LRELU': LRELU,
    'RELU': RELU,
    'tanh': tanh,
    'softmax': softmax
}

# -------------------- FUNZIONI UTILI --------------------
def runge(x, noise_std=0.0):
    """Funzione di Runge con noise opzionale."""
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1 / (1 + 25 * x**2) + noise

def build_layout(n_hidden: int, width: int):
    """Costruisce il layout della rete: input + hidden + output."""
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]

def extract_losses(history: dict, net: FFNN, X_val, y_val_noisy, y_val_clean, mode="min", last_n=100):
    """Estrae la validation loss dal history o fallback predict, per noisy e clean."""
    y_pred = net.predict(X_val)
   
    # Loss su y_val noisy (fallback)
    val_loss_noisy = float(CostOLS(y_val_noisy)(y_pred))
   
    # Loss su y_val clean (sempre final predict)
    val_loss_clean = float(CostOLS(y_val_clean)(y_pred))
   
    # Se history ha val_loss, usa come specificato dal mode
    val_losses_hist = history.get("val_loss", history.get("val_errors"))
    if val_losses_hist is not None and len(val_losses_hist) > 0:
        if mode == "min":
            val_loss_noisy_hist = float(np.nanmin(val_losses_hist))
        elif mode == "final":
            val_loss_noisy_hist = float(val_losses_hist[-1])
        elif mode == "avg_last_n":
            val_loss_noisy_hist = float(np.mean(val_losses_hist[-last_n:]))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        val_loss_noisy = val_loss_noisy_hist
    return val_loss_noisy, val_loss_clean

# -------------------- GESTIONE RUN E FOLDER --------------------
def newest_run_dir(base_dir="Models_Reg"):
    """Trova l'ultima run directory."""
    runs = sorted([d for d in os.listdir(base_dir) if d.startswith("run_")])
    return runs[-1] if runs else None

def start_new_run(base_dir="Models_Reg", output_dir="output_reg"):
    """Crea una nuova run directory."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{current_time}"
    os.makedirs(os.path.join(base_dir, run_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, run_dir), exist_ok=True)
    return run_dir

def has_incomplete_work(run_dir, output_dir="output_reg", models=None):
    """Controlla se ci sono temp files incompleti con NaN per noisy o clean."""
    if models is None:
        return False
    for model_name in models:
        for suffix in ["noisy", "clean"]:
            temp_path = os.path.join(output_dir, run_dir, f"temp_heat_{suffix}_{model_name}.csv")
            if os.path.exists(temp_path):
                df = pd.read_csv(temp_path, index_col='lam_l1')
                if np.isnan(df.values).any():
                    return True
    return False

# -------------------- MAIN SCRIPT --------------------
# Parametri fissi e configurabili
SEED = int(os.environ.get("SEED", 314)) # Da env o default
np.random.seed(SEED)

# Dati
X = np.linspace(-1, 1, 200).reshape(-1, 1)
noise_global = 0.1  # Noise su tutti i dati (per y_noisy)
noise_train_extra = 0.03  # Noise extra solo su y_train (per regularization)
y_noisy = runge(X, noise_std=noise_global).reshape(-1, 1)
y_clean = runge(X, noise_std=0.0).reshape(-1, 1)  # Versione clean per eval

# Training settings
epochs = 1500
lr = 0.001
rho = 0.9
rho2 = 0.999
batches = 100
activation_func = RELU  # Funzione da usare
VAL_LOSS_MODE = "avg_last_n"  # "min", "final", "avg_last_n"
LAST_N = 100  # Per avg_last_n

# Definire i modelli: (name, n_hidden, width)
models = [
    ("simple", 1, 4),
    ("intermediate", 3, 20),
    ("overfitting", 5, 40)
]

# Griglia per L1 e L2
lam_l1_list = np.logspace(-6, -1, 10).tolist()
lam_l2_list = np.logspace(-6, -1, 10).tolist()

# Cartelle base
BASE_DIR = "Models_Reg"
OUTPUT_DIR = "output/l1_l2_analysis"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Decidi se continuare o nuova run
last_run = newest_run_dir(BASE_DIR)
if last_run and has_incomplete_work(last_run, OUTPUT_DIR, [m[0] for m in models]):
    run_dir = last_run
    is_continuing = True
    print(f"Continuing existing run: {run_dir}")
else:
    run_dir = start_new_run(BASE_DIR, OUTPUT_DIR)
    is_continuing = False
    print(f"Starting new run: {run_dir}")

# Path config
config_path = os.path.join(BASE_DIR, run_dir, "config.json")

# Carica o crea config
if is_continuing and os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Applica config caricati
    SEED = int(config['SEED'])
    np.random.seed(SEED)
    epochs = int(config['epochs'])
    lr = float(config['lr'])
    rho = float(config['rho'])
    rho2 = float(config['rho2'])
    batches = int(config['batches'])
    activation_func = act_func_map[config['activation_func']]
    VAL_LOSS_MODE = config['VAL_LOSS_MODE']
    if 'LAST_N' in config:
        LAST_N = int(config['LAST_N'])
    noise_global = float(config['noise_global'])
    noise_train_extra = float(config['noise_train_extra'])
    models = [(m['name'], m['n_hidden'], m['width']) for m in config['models']]
    lam_l1_list = list(config['lam_l1_list'])
    lam_l2_list = list(config['lam_l2_list'])
    # Rigenera y_noisy e y_clean con seed
    y_noisy = runge(X, noise_std=noise_global).reshape(-1, 1)
    y_clean = runge(X, noise_std=0.0).reshape(-1, 1)
else:
    config = {
        'SEED': SEED,
        'epochs': epochs,
        'lr': lr,
        'rho': rho,
        'rho2': rho2,
        'batches': batches,
        'activation_func': activation_func.__name__,
        'VAL_LOSS_MODE': VAL_LOSS_MODE,
        'LAST_N': LAST_N,
        'noise_global': noise_global,
        'noise_train_extra': noise_train_extra,
        'models': [{'name': name, 'n_hidden': n_hidden, 'width': width} for name, n_hidden, width in models],
        'lam_l1_list': lam_l1_list,
        'lam_l2_list': lam_l2_list
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Split dati (dopo seed) - usa y_noisy per split e train
X_train, X_val, y_train_noisy, y_val_noisy = train_test_split(
    X, y_noisy, test_size=0.2, random_state=SEED, shuffle=True
)

# Aggiungi noise extra solo su train
y_train_noisy += np.random.normal(0, noise_train_extra, y_train_noisy.shape)

# Genera y_val_clean corrispondente (deterministico)
y_val_clean = runge(X_val, noise_std=0.0)

# -------------------- LOOP SU MODELS --------------------
for model_name, n_hidden, width in models:
    act = activation_func
    # Path per noisy
    csv_noisy = f"val_loss_data_noisy_{model_name}.csv"
    csv_path_noisy = os.path.join(OUTPUT_DIR, run_dir, csv_noisy)
    temp_heat_path_noisy = os.path.join(OUTPUT_DIR, run_dir, f"temp_heat_noisy_{model_name}.csv")
   
    # Path per clean
    csv_clean = f"val_loss_data_clean_{model_name}.csv"
    csv_path_clean = os.path.join(OUTPUT_DIR, run_dir, csv_clean)
    temp_heat_path_clean = os.path.join(OUTPUT_DIR, run_dir, f"temp_heat_clean_{model_name}.csv")

    # Skip se entrambi già completati
    if os.path.exists(csv_path_noisy) and os.path.exists(csv_path_clean):
        print(f"[{model_name}] already completed for both noisy and clean. Skipping.")
        continue

    # Carica o crea heatmaps temporanee per noisy
    if os.path.exists(temp_heat_path_noisy):
        df_temp_noisy = pd.read_csv(temp_heat_path_noisy, index_col='lam_l1')
        if not np.array_equal(df_temp_noisy.index.values, lam_l1_list) or not np.array_equal(df_temp_noisy.columns.values, lam_l2_list):
            heat_noisy = np.full((len(lam_l1_list), len(lam_l2_list)), np.nan, dtype=float)
        else:
            heat_noisy = df_temp_noisy.values
    else:
        heat_noisy = np.full((len(lam_l1_list), len(lam_l2_list)), np.nan, dtype=float)

    # Carica o crea heatmaps temporanee per clean
    if os.path.exists(temp_heat_path_clean):
        df_temp_clean = pd.read_csv(temp_heat_path_clean, index_col='lam_l1')
        if not np.array_equal(df_temp_clean.index.values, lam_l1_list) or not np.array_equal(df_temp_clean.columns.values, lam_l2_list):
            heat_clean = np.full((len(lam_l1_list), len(lam_l2_list)), np.nan, dtype=float)
        else:
            heat_clean = df_temp_clean.values
    else:
        heat_clean = np.full((len(lam_l1_list), len(lam_l2_list)), np.nan, dtype=float)

    interrupted = False
    try:
        for i_l1, lam_l1 in enumerate(lam_l1_list):
            for j_l2, lam_l2 in enumerate(lam_l2_list):
                # Skip se già calcolato
                if not np.isnan(heat_noisy[i_l1, j_l2]):
                    continue
                layout = build_layout(n_hidden, width)
                model_filename = f"model_{model_name}_l1_{lam_l1:.1e}_l2_{lam_l2:.1e}.npz"
                model_path = os.path.join(BASE_DIR, run_dir, model_filename)
                done_marker = model_path + ".done"

                # Crea rete e scheduler
                net = FFNN(
                    dimensions=layout,
                    hidden_func=act,
                    output_func=identity,
                    cost_func=CostOLS,
                    seed=SEED,
                )
                scheduler = Adam(lr, rho, rho2)
                print(f"Training {model_filename}")

                # Fit senza salvare su interrupt
                history = net.fit(
                    X=X_train, t=y_train_noisy,
                    scheduler=scheduler,
                    batches=batches,
                    epochs=epochs,
                    lam_l1=lam_l1,
                    lam_l2=lam_l2,
                    X_val=X_val, t_val=y_val_noisy,  # Usa noisy per val durante training
                    save_on_interrupt=None,
                )

                # Salva solo se completato
                net.save_weights(model_path)
                with open(done_marker, "w") as _f:
                    _f.write("ok")

                # Estrai entrambe le losses
                val_loss_noisy, val_loss_clean = extract_losses(
                    history, net, X_val, y_val_noisy, y_val_clean, mode=VAL_LOSS_MODE, last_n=LAST_N
                )
                heat_noisy[i_l1, j_l2] = val_loss_noisy
                heat_clean[i_l1, j_l2] = val_loss_clean

                # Salva temp heatmaps
                df_temp_noisy = pd.DataFrame(heat_noisy, index=lam_l1_list, columns=lam_l2_list)
                df_temp_noisy.index.name = 'lam_l1'
                df_temp_noisy.columns.name = 'lam_l2'
                df_temp_noisy.to_csv(temp_heat_path_noisy)
               
                df_temp_clean = pd.DataFrame(heat_clean, index=lam_l1_list, columns=lam_l2_list)
                df_temp_clean.index.name = 'lam_l1'
                df_temp_clean.columns.name = 'lam_l2'
                df_temp_clean.to_csv(temp_heat_path_clean)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Current model will be retrained from scratch next time.")
        interrupted = True

    # Se non interrotto, finalizza entrambe
    if not interrupted:
        # Noisy
        if not os.path.exists(csv_path_noisy):
            df_noisy = pd.DataFrame(heat_noisy, index=lam_l1_list, columns=lam_l2_list)
            df_noisy.index.name = 'lam_l1'
            df_noisy.columns.name = 'lam_l2'
            df_noisy.to_csv(csv_path_noisy)
            if os.path.exists(temp_heat_path_noisy):
                os.remove(temp_heat_path_noisy)
            # Plot noisy
            plt.figure(figsize=(10, 5))
            im = plt.imshow(heat_noisy, aspect="auto", origin="upper", interpolation="nearest")
            plt.colorbar(im, label="Validation Loss (OLS) - Noisy")
            plt.title(f"Validation Loss Heatmap (Noisy) — Model: {model_name}")
            plt.xlabel("lam_l2")
            plt.ylabel("lam_l1")
            plt.xticks(ticks=np.arange(len(lam_l2_list)), labels=[f"{v:.1e}" for v in lam_l2_list], rotation=45)
            plt.yticks(ticks=np.arange(len(lam_l1_list)), labels=[f"{v:.1e}" for v in lam_l1_list])
            plt.tight_layout()
            plot_filename_noisy = f"val_loss_heatmap_noisy_{model_name}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, run_dir, plot_filename_noisy))
            plt.close()

        # Clean
        if not os.path.exists(csv_path_clean):
            df_clean = pd.DataFrame(heat_clean, index=lam_l1_list, columns=lam_l2_list)
            df_clean.index.name = 'lam_l1'
            df_clean.columns.name = 'lam_l2'
            df_clean.to_csv(csv_path_clean)
            if os.path.exists(temp_heat_path_clean):
                os.remove(temp_heat_path_clean)
            # Plot clean
            plt.figure(figsize=(10, 5))
            im = plt.imshow(heat_clean, aspect="auto", origin="upper", interpolation="nearest")
            plt.colorbar(im, label="Validation Loss (OLS) - Clean")
            plt.title(f"Validation Loss Heatmap (Clean) — Model: {model_name}")
            plt.xlabel("lam_l2")
            plt.ylabel("lam_l1")
            plt.xticks(ticks=np.arange(len(lam_l2_list)), labels=[f"{v:.1e}" for v in lam_l2_list], rotation=45)
            plt.yticks(ticks=np.arange(len(lam_l1_list)), labels=[f"{v:.1e}" for v in lam_l1_list])
            plt.tight_layout()
            plot_filename_clean = f"val_loss_heatmap_clean_{model_name}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, run_dir, plot_filename_clean))
            plt.close()
    else:
        # Interrotto: esci dal loop models
        break

print("Done or paused. Resume will automatically continue from first missing cell.")