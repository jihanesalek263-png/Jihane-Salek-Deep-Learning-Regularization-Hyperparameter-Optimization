# =====================================================
# EXERCICE 4 — GRID SEARCH
# =====================================================

import itertools
import pandas as pd
import torch
import seaborn as sns





from ex1_dataset import train_loader, val_loader
from ex2_model import DeepFFN, train_model

# =====================================================
# Q1 — Définition de la grille complète
# =====================================================

param_grid = {
    'hidden_dims': [[64,32], [128,64], [128,64,32], [256,128,64]],
    'activation': ['relu', 'leaky_relu', 'elu'],
    'dropout_rate': [0.0, 0.2, 0.3],
    'lr': [1e-3, 5e-4],
    'weight_decay': [0.0, 1e-4],
    'clip_value': [1.0, 5.0],
}

n_configs = 1

for v in param_grid.values():
    n_configs *= len(v)

print(f"Total configurations Grid Search : {n_configs}")

# =====================================================
# Q2 — Fonction Grid Search
# =====================================================

def grid_search(param_grid, train_loader, val_loader, epochs=80):

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combos = list(itertools.product(*values))

    results = []

    print(
        f"Grid Search : {len(combos)} configurations × {epochs} epochs max"
    )

    print("-" * 60)

    for i, combo in enumerate(combos):

        config = dict(zip(keys, combo))

        config["epochs"] = epochs
        config["early_stopping_patience"] = 15
        config["use_bn"] = True

        torch.manual_seed(42)

        model = DeepFFN(
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            use_bn=True,
            dropout_rate=config["dropout_rate"]
        )

        _, best_mse, elapsed = train_model(
            model,
            train_loader,
            val_loader,
            config
        )

        results.append({
            **config,
            "val_mse": best_mse,
            "time_s": elapsed
        })

        print(
            f"[{i+1:3d}/{len(combos)}] "
            f"val_MSE={best_mse:.4f} "
            f"({elapsed:.1f}s)"
        )

    df = pd.DataFrame(results)

    return df.sort_values("val_mse")





param_grid_small = {
    'hidden_dims' : [[64,32], [128,64,32], [256,128,64]],
    'activation' : ['relu', 'leaky_relu'],
    'dropout_rate' : [0.1, 0.3],
    'lr' : [1e-3, 5e-4],
    'weight_decay' : [1e-4, 1e-3],
    'clip_value' : [1.0],
}
gs_results = grid_search(
    param_grid_small,
    train_loader,
    val_loader,
    epochs=80
)

gs_results.to_csv(
    "grid_search_results.csv",
    index=False
)


print("\n=== TOP 10 ===")

print(
    gs_results[
        [
            "hidden_dims",
            "activation",
            "dropout_rate",
            "lr",
            "weight_decay",
            "val_mse"
        ]
    ].head(10).to_string(index=False)
)






import matplotlib.pyplot as plt

# =====================================================
# Q4.a — Boxplot par activation
# =====================================================

plt.figure(figsize=(8,5))

gs_results.boxplot(
    column="val_mse",
    by="activation"
)

plt.title("Validation MSE par activation")
plt.suptitle("")
plt.ylabel("val_mse")

plt.show()



plt.figure(figsize=(8,5))

gs_results.boxplot(
    column="val_mse",
    by="dropout_rate"
)

plt.title("Validation MSE par dropout")
plt.suptitle("")
plt.ylabel("val_mse")

plt.show()


pivot = gs_results.pivot_table(
    values="val_mse",
    index="lr",
    columns="weight_decay",
    aggfunc="mean"
)

plt.figure(figsize=(6,4))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".4f"
)

plt.title("Heatmap : lr × weight_decay")

plt.show()

top15 = gs_results.head(15).copy()

top15["arch"] = top15["hidden_dims"].astype(str)

plt.figure(figsize=(12,6))

plt.bar(
    range(len(top15)),
    top15["val_mse"]
)

plt.xticks(
    range(len(top15)),
    top15["arch"],
    rotation=45
)

plt.ylabel("Validation MSE")
plt.title("Top 15 Configurations")

plt.tight_layout()
plt.show()



gs_results["hidden_dims_str"] = gs_results["hidden_dims"].astype(str)

print("\n=== Impact des hyperparamètres ===")

params = [
    "activation",
    "dropout_rate",
    "hidden_dims_str",
    "lr",
    "weight_decay"
]

for param in params:

    impact = gs_results.groupby(param)["val_mse"].std()

    print(f"\n{param}")
    print(impact.sort_values(ascending=False))