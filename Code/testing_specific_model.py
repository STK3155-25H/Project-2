import os
import re
import numpy as np
import matplotlib.pyplot as plt

from src.FFNN import FFNN
from src.cost_functions import CostOLS
from src.activation_functions import identity, LRELU, RELU, tanh

SEED = 314
np.random.seed(SEED)

def runge(x, noise_std=0.05):
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1 / (1 + 25 * x**2) + noise

X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge(X, noise_std=0.03).reshape(-1, 1)



# ---------- Name & path handling ----------



def parse_model_filename(file_path: str):
    """
    Estrae (n_hidden, width, act_name) dal NOME FILE (basename),
    indipendentemente dalla cartella in cui si trova.

    Formato atteso (case-insensitive):
        model_hidden_{n}_width_{w}_act_{act}.npz
    """
    base = os.path.basename(file_path)
    pattern = r"^model_hidden_(\d+)_width_(\d+)_act_([A-Za-z0-9_+-]+)\.npz$"
    match = re.match(pattern, base, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid filename format: {base}")
    n_hidden = int(match.group(1))
    width = int(match.group(2))
    act_name = match.group(3)
    # normalizziamo l'activation per la lookup (es. relu/LReLU/TANH -> maiuscolo tranne 'tanh')
    return n_hidden, width, act_name

def get_activation_function(act_name: str):
    """
    Restituisce la funzione di attivazione a partire dal nome (case-insensitive).
    Aggiungi qui eventuali nuove attivazioni.
    """
    key = act_name.strip().upper()
    act_dict = {
        "LRELU": LRELU,
        "LEAKYRELU": LRELU,   # alias comodo
        "RELU": RELU,
        "TANH": tanh,
        "HYPERBOLICTANGENT": tanh,  # alias
    }
    act_func = act_dict.get(key)
    if act_func is None:
        raise ValueError(f"Unknown activation function: {act_name}")
    return act_func

# ---------- Data ----------

def runge_true(x: np.ndarray) -> np.ndarray:
    """Runge function (senza rumore)."""
    return 1 / (1 + 25 * x**2)

# ---------- Eval ----------

def evaluate_model(file_path: str, save_plot: bool = False, plot_dir: str = "output") -> float:
    """
    Valuta il modello salvato in file_path:
      - carica i pesi
      - predice su [-1, 1]
      - calcola MSE contro la Runge
      - opzionalmente salva il plot
    Ritorna: loss MSE (float)
    """
    # Parse dal nome file (non importa la cartella)
    n_hidden, width, act_name = parse_model_filename(file_path)
    act_func = get_activation_function(act_name)

    # Costruzione dimensioni rete
    dims = [1] + [width] * n_hidden + [1]

    # Istanza rete
    net = FFNN(
        dimensions=tuple(dims),
        hidden_func=act_func,
        output_func=identity,
        cost_func=CostOLS,
    )

    # Path: se file_path è già esistente usalo; altrimenti prova a cercarlo sotto "Models/"
    model_path = os.path.normpath(file_path)
    if not os.path.exists(model_path):
        candidate = os.path.normpath(os.path.join("Models", file_path))
        if os.path.exists(candidate):
            model_path = candidate
        else:
            raise FileNotFoundError(f"Model file not found: {file_path} (also tried {candidate})")

    # Carica pesi
    net.load_weights(model_path)

    # Dati di valutazione
    X_eval = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = runge_true(X_eval)

    # Predizione
    y_pred = net.predict(X_eval)

    # MSE
    loss = float(np.mean((y_pred - y_true) ** 2))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(X, y, label = "True with noise")
    plt.plot(X_eval, y_true, label="True Runge Function")
    plt.plot(X_eval, y_pred, label="Model Prediction", linestyle="--")
    plt.title(f"Model: {os.path.basename(model_path)}\nMSE Loss: {loss:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    if save_plot:
        os.makedirs(plot_dir, exist_ok=True)
        # filename safe: sostituisco separatori dir con '__'
        safe_name = model_path.replace(os.sep, "__").replace("/", "__")
        plot_filename = f"evaluation_{os.path.splitext(os.path.basename(safe_name))[0]}.png"
        out_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()

    plt.close()
    return loss

# ---------- CLI example ----------

if __name__ == "__main__":
    # Esempio: percorso completo o relativo, funziona in entrambi i casi
    example_file = "Models/run_20251031_171843/model_hidden_5_width_38_act_LRELU.npz"
    loss = evaluate_model(example_file, save_plot=True)
    print(f"Evaluation Loss: {loss:.6f}")
