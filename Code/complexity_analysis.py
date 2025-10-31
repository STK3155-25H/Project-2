import numpy as np
import os
import pandas as pd
from datetime import datetime
import json
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import sigmoid, identity, LRELU, RELU, tanh, softmax

act_func_map = {'sigmoid': sigmoid, 'identity':identity, 'LRELU':LRELU, 'RELU':RELU, 'tanh':tanh, 'softmax':softmax}

# -------------------- DATA --------------------
def runge(x, noise_std=0.05):
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1 / (1 + 25 * x**2) + noise

X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge(X, noise_std=0.03).reshape(-1, 1)

# -------------------- TRAINING SETTINGS --------------------
epochs = 1500
lr = 0.001
lam1 = 0.0
lam2 = 0.0
rho = 0.9
rho2 = 0.999
batches = 100
noise_std = 0.02  # Noise only on training labels
activation_funcs = [LRELU, RELU, tanh]
n_hidden_list = list(range(1, 6))
n_perceptrons_list = [2*i for i in range(1, 21)]
VAL_LOSS_MODE = "min"  # or "final"

def build_layout(n_hidden: int, width: int):
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]

def extract_val_loss(history: dict, net: FFNN, X_val, y_val):
    val = history.get("val_loss", history.get("val_errors"))
    if val is not None and len(val) > 0:
        return float(np.nanmin(val) if VAL_LOSS_MODE == "min" else val[-1])
    # fallback
    y_pred = net.predict(X_val)
    return float(CostOLS(y_val)(y_pred))

# -------------------- RUN FOLDERS --------------------
os.makedirs("Models", exist_ok=True)
os.makedirs("output", exist_ok=True)

def newest_run_dir():
    runs = sorted([d for d in os.listdir("Models") if d.startswith("run_")])
    return runs[-1] if runs else None

def start_new_run():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{current_time}"
    os.makedirs(os.path.join("Models", run_dir), exist_ok=True)
    os.makedirs(os.path.join("output", run_dir), exist_ok=True)
    return run_dir

def has_incomplete_work(run_dir):
    # Se esiste almeno un temp_heat_*.csv con NaN -> incompleta
    temp_files = glob.glob(os.path.join("output", run_dir, "temp_heat_*.csv"))
    for tf in temp_files:
        df = pd.read_csv(tf, index_col='hidden_layers')
        if np.isnan(df.values).any():
            return True
    return False

# Strategy: se l’ultima run ha temp_heat con buchi => continua; altrimenti crea run nuova
last = newest_run_dir()
if last and has_incomplete_work(last):
    run_dir = last
    is_continuing = True
    print(f"Continuing existing run: {run_dir}")
else:
    run_dir = start_new_run()
    is_continuing = False
    print(f"Starting new run: {run_dir}")

config_path = os.path.join("Models", run_dir, "config.json")

# -------------------- CONFIG SEED --------------------
if is_continuing and os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    SEED = int(config['SEED'])
    np.random.seed(SEED)
    epochs = int(config['epochs'])
    lr = float(config['lr'])
    lam1 = float(config['lam1'])
    lam2 = float(config['lam2'])
    rho = float(config['rho'])
    rho2 = float(config['rho2'])
    batches = int(config['batches'])
    activation_funcs = [act_func_map[name] for name in config['activation_funcs']]
    n_hidden_list = list(config['n_hidden_list'])
    n_perceptrons_list = list(config['n_perceptrons_list'])
    VAL_LOSS_MODE = config['VAL_LOSS_MODE']
    noise_std = float(config.get('noise_std', 0.05))
else:
    SEED = os.environ.get("SEED")
    if SEED is not None:
        SEED = int(SEED)
        print("SEED from env:", SEED)
    else:
        SEED = 314
        print("SEED from hard-coded value in file ml_core.py :", SEED)
        print("If you want a specific SEED set the SEED environment variable")
    np.random.seed(SEED)
    config = {
        'SEED': SEED,
        'epochs': epochs,
        'lr': lr,
        'lam1': lam1,
        'lam2': lam2,
        'rho': rho,
        'rho2': rho2,
        'batches': batches,
        'activation_funcs': [f.__name__ for f in activation_funcs],
        'n_hidden_list': n_hidden_list,
        'n_perceptrons_list': n_perceptrons_list,
        'VAL_LOSS_MODE': VAL_LOSS_MODE,
        'noise_std': noise_std
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# -------------------- DATA SPLIT --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)
y_train = y_train + np.random.normal(0, noise_std, y_train.shape)

# -------------------- LOOP --------------------
for act in activation_funcs:
    csv_filename = f"val_loss_data_{act.__name__}.csv"
    csv_path = os.path.join("output", run_dir, csv_filename)
    temp_heat_path = os.path.join("output", run_dir, f"temp_heat_{act.__name__}.csv")

    # Se l’activation è già completata (csv finale esiste), passa oltre
    if os.path.exists(csv_path):
        print(f"[{act.__name__}] already completed. Skipping.")
        continue

    # Carica/crea heat temporanea
    if os.path.exists(temp_heat_path):
        df_temp = pd.read_csv(temp_heat_path, index_col='hidden_layers')
        # Se dimensioni cambiate tra run, riallinea
        if list(df_temp.index.astype(int)) != n_hidden_list or list(df_temp.columns.astype(int)) != n_perceptrons_list:
            heat = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan, dtype=float)
        else:
            heat = df_temp.values
    else:
        heat = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan, dtype=float)

    interrupted = False

    try:
        for i_h, n_hidden in enumerate(n_hidden_list):
            for j_w, width in enumerate(n_perceptrons_list):
                # Già calcolato? Vai avanti
                if not np.isnan(heat[i_h, j_w]):
                    continue

                layout = build_layout(n_hidden, width)
                model_filename = f"model_hidden_{n_hidden}_width_{width}_act_{act.__name__}.npz"
                model_path = os.path.join("Models", run_dir, model_filename)
                done_marker = model_path + ".done"

                # Non fidarti di pesi parziali: si ricalcola sempre finché heat è NaN
                # (Se vuoi sfruttarli, abilita il blocco facoltativo più sotto)

                net = FFNN(
                    dimensions=layout,
                    hidden_func=act,
                    output_func=identity,
                    cost_func=CostOLS,
                    seed=SEED,
                )
                scheduler = Adam(lr, rho, rho2)

                print(f"Training {model_filename}")

                # NB: NON salviamo i pesi su interrupt -> riparte da zero
                history = net.fit(
                    X=X_train, t=y_train,
                    scheduler=scheduler,
                    batches=batches,
                    epochs=epochs,
                    lam_l1=lam1,
                    lam_l2=lam2,
                    X_val=X_val, t_val=y_val,
                    save_on_interrupt=None,  # <= fondamentale per il tuo requisito
                )

                net.save_weights(model_path)  # salviamo solo se ha finito
                # marker “completato” per sicurezza opzionale
                with open(done_marker, "w") as _f:
                    _f.write("ok")

                val_loss = extract_val_loss(history, net, X_val, y_val)

                heat[i_h, j_w] = val_loss

                # salva temp ogni volta
                df_temp = pd.DataFrame(heat, index=n_hidden_list, columns=n_perceptrons_list)
                df_temp.index.name = 'hidden_layers'
                df_temp.columns.name = 'neurons_per_layer'
                df_temp.to_csv(temp_heat_path)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Current model will be retrained from scratch next time.")
        interrupted = True

    # Se arriviamo qui, o abbiamo completato l’activation o siamo stati interrotti.
    # In entrambi i casi, salviamo lo stato attuale del calcolo (temp_heat è già salvata).
    # Se NON interrotto, finalizziamo la heat per questa activation.
    if not interrupted:
        df = pd.DataFrame(heat, index=n_hidden_list, columns=n_perceptrons_list)
        df.index.name = 'hidden_layers'
        df.columns.name = 'neurons_per_layer'
        df.to_csv(csv_path)
        if os.path.exists(temp_heat_path):
            os.remove(temp_heat_path)

        # Plot
        plt.figure(figsize=(10, 5))
        im = plt.imshow(
            heat,
            aspect="auto",
            origin="upper",
            interpolation="nearest"
        )
        plt.colorbar(im, label="Validation Loss (OLS)")
        plt.title(f"Validation Loss Heatmap — activation: {act.__name__}")
        plt.xlabel("Neurons per hidden layer")
        plt.ylabel("Number of hidden layers")

        plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list, rotation=45)
        plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)

        plt.tight_layout()
        plot_filename = f"val_loss_heatmap_{act.__name__}.png"
        plt.savefig(os.path.join("output", run_dir, plot_filename))
        plt.show()
    else:
        # Interrotto: non produciamo CSV finale né plot, per indicare che questa activation non è completa.
        break

print("Done or paused. Resume will automatically continue from first missing cell.")
