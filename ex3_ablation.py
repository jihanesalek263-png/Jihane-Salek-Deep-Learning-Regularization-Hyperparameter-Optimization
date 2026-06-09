# EXERCICE 3 — Analyse de l'Impact des Techniques de Stabilisation

import torch
import matplotlib.pyplot as plt
import pandas as pd

from ex1_dataset import train_loader, val_loader
from ex2_model import DeepFFN, train_model

# ==========================================================
# Q1 — Ablation Study : BatchNorm vs Dropout vs L2
# ==========================================================

configs = {
    "A": {
        "use_bn": True,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4
    },

    "B": {
        "use_bn": False,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4
    },

    "C": {
        "use_bn": True,
        "dropout_rate": 0.0,
        "weight_decay": 1e-4
    },

    "D": {
        "use_bn": True,
        "dropout_rate": 0.2,
        "weight_decay": 0.0
    },

    "E": {
        "use_bn": False,
        "dropout_rate": 0.0,
        "weight_decay": 0.0
    }
}

results = []
all_histories = {}

for name, cfg in configs.items():

    print(f"\n===== CONFIG {name} =====")

    torch.manual_seed(42)

    model = DeepFFN(
        hidden_dims=[128, 64, 32],
        activation="relu",
        use_bn=cfg["use_bn"],
        dropout_rate=cfg["dropout_rate"]
    )

    config = {
        "hidden_dims": [128, 64, 32],
        "activation": "relu",
        "use_bn": cfg["use_bn"],
        "dropout_rate": cfg["dropout_rate"],
        "lr": 1e-3,
        "weight_decay": cfg["weight_decay"],
        "clip_value": 1.0,
        "epochs": 100,
        "early_stopping_patience": 15
    }

    history, best_val_mse, training_time = train_model(
        model,
        train_loader,
        val_loader,
        config
    )

    all_histories[name] = history

    results.append({
        "Configuration": name,
        "Best_Val_MSE": best_val_mse,
        "Training_Time_s": training_time
    })

    print("Best Val MSE :", best_val_mse)
    print("Training Time :", training_time)

# ==========================================================
# Tableau Résultats Q1
# ==========================================================

print("\n=== Tableau Résultats Q1 ===")

df_results = pd.DataFrame(results)

print(df_results)

df_results.to_csv(
    "ablation_results.csv",
    index=False
)

# ==========================================================
# Q2 — A vs B
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    all_histories["A"]["val_mse"],
    label="A Baseline"
)

plt.plot(
    all_histories["B"]["val_mse"],
    label="B Sans BatchNorm"
)

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("A vs B")
plt.legend()
plt.grid(True)

plt.savefig(
    "A_vs_B.png",
    dpi=120
)

plt.show()

# ==========================================================
# Q2 — A vs C
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    all_histories["A"]["val_mse"],
    label="A Baseline"
)

plt.plot(
    all_histories["C"]["val_mse"],
    label="C Sans Dropout"
)

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("A vs C")
plt.legend()
plt.grid(True)

plt.savefig(
    "A_vs_C.png",
    dpi=120
)

plt.show()

# ==========================================================
# Q2 — A vs E
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    all_histories["A"]["val_mse"],
    label="A Baseline"
)

plt.plot(
    all_histories["E"]["val_mse"],
    label="E Sans Régularisation"
)

plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("A vs E")
plt.legend()
plt.grid(True)

plt.savefig(
    "A_vs_E.png",
    dpi=120
)

plt.show()








# ==========================================================
# Q3 — Effet du Gradient Clipping
# ==========================================================

clip_values = [0.1, 0.5, 1.0, 5.0, 10.0]

gradient_histories = {}
clip_results = []

for clip in clip_values:

    print(f"\n===== CLIP VALUE = {clip} =====")

    torch.manual_seed(42)

    model = DeepFFN(
        hidden_dims=[128, 64, 32],
        activation="relu",
        use_bn=True,
        dropout_rate=0.2
    )

    config = {
        "hidden_dims": [128, 64, 32],
        "activation": "relu",
        "use_bn": True,
        "dropout_rate": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_value": clip,
        "epochs": 50,
        "early_stopping_patience": 15
    }

    history, best_val_mse, training_time = train_model(
        model,
        train_loader,
        val_loader,
        config
    )

    gradient_histories[str(clip)] = history["grad_norm"]

    clip_results.append({
        "Clip_Value": clip,
        "Best_Val_MSE": best_val_mse,
        "Training_Time_s": training_time
    })

    print(
        f"clip={clip} | "
        f"best_val_mse={best_val_mse:.4f} | "
        f"time={training_time:.1f}s"
    )

# ==========================================================
# Tableau récapitulatif Q3
# ==========================================================

print("\n=== Résultats Gradient Clipping ===")

df_clip = pd.DataFrame(clip_results)

print(df_clip)

df_clip.to_csv(
    "gradient_clipping_results.csv",
    index=False
)

# ==========================================================
# Courbes Gradient Norm
# ==========================================================

plt.figure(figsize=(10, 6))

for clip, grad_values in gradient_histories.items():

    plt.plot(
        grad_values,
        label=f"clip={clip}"
    )

plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm selon la valeur du clipping")
plt.legend()
plt.grid(True)

plt.savefig(
    "gradient_norm_curves.png",
    dpi=120
)

plt.show()

# ==========================================================
# Q4 — Boxplot des normes de gradient
# ==========================================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [gradient_histories[k] for k in gradient_histories],
    labels=list(gradient_histories.keys())
)

plt.xlabel("Clip Value")
plt.ylabel("Gradient Norm")
plt.title("Distribution des normes de gradient")

plt.savefig(
    "gradient_boxplot.png",
    dpi=120
)

plt.show()