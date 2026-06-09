# =====================================================
# EXERCICE 5 — RANDOM SEARCH
# =====================================================

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ex1 import train_loader, val_loader
from ex2_model import DeepFFN, train_model


# =======================================================
# SEARCH SPACE
# =======================================================

search_space = {
    'lr': ('log_uniform', 1e-4, 1e-2),
    'weight_decay': ('log_uniform', 1e-5, 1e-2),
    'dropout_rate': ('uniform', 0.0, 0.5),
    'clip_value': ('uniform', 0.5, 5.0),

    'hidden_dims': ('choice', [
        [64,32],
        [128,64],
        [128,64,32],
        [256,128,64],
        [256,128,64,32]
    ]),

    'activation': ('choice', ['relu', 'leaky_relu', 'elu', 'selu'])
}


# =====================================================
# SAMPLE CONFIG
# =====================================================

def sample_config(space):
    config = {}

    for key, spec in space.items():
        dist = spec[0]

        if dist == 'log_uniform':
            config[key] = np.exp(
                np.random.uniform(np.log(spec[1]), np.log(spec[2]))
            )

        elif dist == 'uniform':
            config[key] = random.uniform(spec[1], spec[2])

        elif dist == 'choice':
            config[key] = random.choice(spec[1])

    return config


print("Test sample_config :", sample_config(search_space))


# =====================================================
# RANDOM SEARCH
# =====================================================

def random_search(search_space, n_trials, train_loader, val_loader, epochs=80):

    results = []

    print(f"\nRandom Search : {n_trials} trials × {epochs} epochs\n")

    for trial in range(n_trials):

        config = sample_config(search_space)

        config["epochs"] = epochs
        config["early_stopping_patience"] = 8
        config["use_bn"] = True

        random.seed(trial)
        np.random.seed(trial)
        torch.manual_seed(trial)

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
            "trial": trial,
            "time_s": elapsed
        })

        print(
            f"Trial {trial+1}/{n_trials} | "
            f"val_mse={best_mse:.4f} | "
            f"lr={config['lr']:.2e} | "
            f"wd={config['weight_decay']:.2e} | "
            f"dr={config['dropout_rate']:.2f}"
        )

    return pd.DataFrame(results).sort_values("val_mse")


# =====================================================
# EXECUTION
# =====================================================

rs_results = random_search(
    search_space,
    n_trials=10,      # réduit pour exécution rapide
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=40         # réduit pour exécution rapide
)

rs_results.to_csv(
    "random_search_results.csv",
    index=False
)

print("\n=== TOP 10 RANDOM SEARCH ===")
print(rs_results.head(10))

# =====================================================
# CONVERGENCE
# =====================================================

def best_so_far(values):
    return np.minimum.accumulate(values)

plt.figure(figsize=(9,5))

plt.plot(best_so_far(rs_results["val_mse"].values), label="Random Search")

plt.xlabel("Trials")
plt.ylabel("Best val MSE")
plt.title("Random Search Convergence")
plt.legend()
plt.grid()
plt.show()