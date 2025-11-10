import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from src.FFNN import FFNN
from src.cost_functions import CostCrossEntropy
from src.activation_functions import LRELU, RELU, softmax

# -----------------------------
# Config / reproducibility
# -----------------------------
SEED = 314
np.random.seed(SEED)

# -----------------------------
# Activation helpers
# -----------------------------
def normalize_act_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", (name or "")).lower()

def get_activation_function(act_name: str):
    key = normalize_act_name(act_name)
    mapping = {
        "relu": RELU,
        "lrelu": LRELU,
        "leakyrelu": LRELU,
        "leaky": LRELU,
    }
    func = mapping.get(key)
    if func is None:
        raise ValueError(f"Unknown activation function: {act_name!r}")
    return func

def canonical_file_token(act_name: str) -> str:
    key = normalize_act_name(act_name)
    token_map = {
        "relu": "RELU",
        "lrelu": "LRELU",
        "leakyrelu": "LRELU",
        "leaky": "LRELU",
    }
    return token_map.get(key, act_name)

# -----------------------------
# Filenames & dirs
# -----------------------------
def model_filename(n_hidden: int, width: int, act_name: str) -> str:
    token = canonical_file_token(act_name)
    return f"model_hidden_{n_hidden}_width_{width}_act_{token}.npz"

def layout_dir_name(n_hidden: int, width: int) -> str:
    return f"hidden{n_hidden}_width{width}"

def run_layout_output_dir(base_out: str, run_dir: str, n_hidden: int, width: int) -> str:
    run_name = os.path.basename(os.path.normpath(run_dir))
    return os.path.join(base_out, run_name, layout_dir_name(n_hidden, width))

# -----------------------------
# MNIST loading (same pipeline as training)
# -----------------------------
def load_mnist_split(seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ricarica MNIST da OpenML, normalizza in [0,1], one-hot a 10 classi,
    e usa il medesimo split usato in training (test_size=0.2, stratify, random_state=seed).
    Ritorna X_val, y_val (one-hot).
    """
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int64)
    y_onehot = np.eye(10)[y]
    # stesso split del tuo script: train_test_split(..., test_size=0.2, random_state=SEED, stratify=y)
    _, X_val, _, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=seed, stratify=y
    )
    return X_val, y_val

# -----------------------------
# Utils: labels & plotting
# -----------------------------
def y_true_as_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(int)
    elif y.ndim == 2:
        return np.argmax(y, axis=1).astype(int)
    raise ValueError("y must be 1D (labels) or 2D (one-hot)")

def preds_to_labels(y_pred: np.ndarray) -> np.ndarray:
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    # MNIST: multiclass → argmax
    return np.argmax(y_pred, axis=1).astype(int)

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    out_png: str,
    out_pdf: Optional[str] = None,
):
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(
                j, i, str(val),
                ha="center", va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    # if out_pdf:
        # plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

# -----------------------------
# Core: evaluate a single model
# -----------------------------
def evaluate_one_model_mnist(
    model_path: str,
    n_hidden: int,
    width: int,
    hidden_activation_name: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    layout_out_dir: str,
) -> Dict[str, Any]:
    """
    Carica un modello MNIST (softmax in output), predice su X_eval,
    costruisce confusion matrix e salva grafici + report.
    """
    hidden_act = get_activation_function(hidden_activation_name)
    input_dim = X_eval.shape[1]         # 784
    num_classes = y_eval.shape[1]       # 10
    dims = [input_dim] + [width] * n_hidden + [num_classes]

    net = FFNN(
        dimensions=tuple(dims),
        hidden_func=hidden_act,
        output_func=softmax,         # <- stessa scelta del training
        cost_func=CostCrossEntropy,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    net.load_weights(model_path)

    y_pred = net.predict(X_eval)            # (N, 10) softmax
    y_true_labels = y_true_as_labels(y_eval)
    y_pred_labels = preds_to_labels(y_pred)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    macro_f1 = f1_score(y_true_labels, y_pred_labels, average="macro")
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(num_classes)))
    class_names = [str(i) for i in range(num_classes)]

    base_name = f"confmat_{os.path.basename(model_path)[:-4]}"
    cm_png = os.path.join(layout_out_dir, base_name + ".png")
    cm_pdf = os.path.join(layout_out_dir, base_name + ".pdf")
    title = f"{os.path.basename(model_path)}\nAcc={acc:.4f}  Macro-F1={macro_f1:.4f}"
    plot_confusion_matrix(cm, class_names, title, cm_png, out_pdf=cm_pdf)
    print(f"[Saved] {cm_png}")
    print(f"[Saved] {cm_pdf}")

    report = classification_report(
        y_true_labels, y_pred_labels, labels=list(range(num_classes)), target_names=class_names
    )
    rep_path = os.path.join(layout_out_dir, base_name + "_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(report)
    print(f"[Saved] {rep_path}")

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "model_path": model_path,
        "num_classes": num_classes,
        "confusion_matrix": cm,
        "report_path": rep_path,
    }

# -----------------------------
# Batch: evaluate all activations for a layout
# -----------------------------
def evaluate_all_activations_for_layout_mnist(
    run_dir: str,
    n_hidden: int,
    width: int,
    activations: Optional[List[str]] = None,
    base_out_dir: str = "output/MNIST_confusions",
    save_csv: bool = True,
    seed: int = SEED,
) -> Dict[str, Dict[str, Any]]:
    """
    Per un run_dir e un layout (n_hidden, width), carica tutti i modelli
    delle attivazioni richieste (default: RELU, LRELU), ricarica MNIST e
    calcola confusion matrix + report, salvando tutto in:
      base_out_dir/<run_name>/hidden<n>_width<w>/
    """
    if activations is None:
        activations = ["RELU", "LRELU"]

    run_dir = os.path.normpath(run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    # X_eval / y_eval = validation split ricreato identico al training
    X_eval, y_eval = load_mnist_split(seed=seed)

    layout_out_dir = run_layout_output_dir(base_out_dir, run_dir, n_hidden, width)
    os.makedirs(layout_out_dir, exist_ok=True)
    print(f"== Evaluating MNIST (confusion) run='{run_dir}', layout=hidden{n_hidden}, width={width} ==")
    print(f"   Output -> {layout_out_dir}")

    results: Dict[str, Dict[str, Any]] = {}

    for act in activations:
        fname = model_filename(n_hidden, width, act)
        model_path = os.path.join(run_dir, fname)
        try:
            r = evaluate_one_model_mnist(
                model_path=model_path,
                n_hidden=n_hidden,
                width=width,
                hidden_activation_name=act,
                X_eval=X_eval,
                y_eval=y_eval,
                layout_out_dir=layout_out_dir,
            )
            results[act] = r
            print(f"[OK]   {act:<8} Acc={r['acc']:.4f}  Macro-F1={r['macro_f1']:.4f}")
        except FileNotFoundError:
            print(f"[Skip] {act:<8} (file not found)")
        except ValueError as e:
            print(f"[Skip] {act:<8} ({e})")
        except Exception as e:
            print(f"[Skip] {act:<8} (error: {type(e).__name__}: {e})")

    if save_csv and results:
        csv_path = os.path.join(layout_out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["activation", "accuracy", "macro_f1", "model_path"])
            for act, info in sorted(results.items(), key=lambda kv: (-kv[1]["acc"], -kv[1]["macro_f1"])):
                w.writerow([act, f"{info['acc']:.6f}", f"{info['macro_f1']:.6f}", info["model_path"]])
        print(f"[Saved] summary CSV -> {csv_path}")
    elif not results:
        print("[Info] No models were evaluated.")

    return results

# -----------------------------
# CLI example
# -----------------------------
if __name__ == "__main__":
    # Esempio: cambia questi tre parametri
    RUN_DIR = "Models_MNIST/run_20251107_220350"  # cartella dove hai salvato i .npz
    N_HIDDEN = 5
    WIDTH = 512

    # Se vuoi specificare esplicitamente le attivazioni presenti nella cartella:
    ACTIVATIONS = None  # oppure ["RELU", "LRELU"]

    evaluate_all_activations_for_layout_mnist(
        run_dir=RUN_DIR,
        n_hidden=N_HIDDEN,
        width=WIDTH,
        activations=ACTIVATIONS,
        base_out_dir="output/MNIST_confusions",
        save_csv=False,
        seed=SEED,
    )
