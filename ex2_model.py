import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import time
import copy


# =====================================================
# Q1-Q3 : DeepFFN
# =====================================================

class DeepFFN(nn.Module):

    def __init__(
        self,
        input_dim=8,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        activation='relu',
        use_bn=True,
        dropout_rate=0.2
    ):

        super().__init__()

        self.activation_name = activation

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )

            if use_bn:
                layers.append(
                    nn.BatchNorm1d(hidden_dim)
                )

            if activation == "relu":
                layers.append(nn.ReLU())

            elif activation == "tanh":
                layers.append(nn.Tanh())

            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())

            elif activation == "elu":
                layers.append(nn.ELU())

            elif activation == "selu":
                layers.append(nn.SELU())

            else:
                raise ValueError(
                    f"Activation inconnue : {activation}"
                )

            if dropout_rate > 0:
                layers.append(
                    nn.Dropout(dropout_rate)
                )

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(
            prev_dim,
            output_dim
        )

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                if self.activation_name in [
                    "relu",
                    "leaky_relu",
                    "elu"
                ]:

                    nn.init.kaiming_normal_(m.weight)

                else:

                    nn.init.xavier_uniform_(m.weight)

                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.hidden_layers(x)

        x = self.output_layer(x)

        return x


# =====================================================
# Q4 : train_one_epoch
# =====================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    clip_value=1.0
):

    model.train()

    total_loss = 0.0
    total_gnorm = 0.0
    n = 0

    for xb, yb in loader:

        optimizer.zero_grad()

        pred = model(xb)

        loss = criterion(pred, yb)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_value
        )

        total_gnorm += grad_norm.item()

        optimizer.step()

        total_loss += loss.item() * len(xb)

        n += len(xb)

    return (
        total_loss / n,
        total_gnorm / len(loader)
    )


# =====================================================
# Q5 : evaluate
# =====================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for xb, yb in loader:

            pred = model(xb)

            preds.extend(
                pred.squeeze().cpu().numpy()
            )

            targets.extend(
                yb.squeeze().cpu().numpy()
            )

    mse = mean_squared_error(
        targets,
        preds
    )

    mae = mean_absolute_error(
        targets,
        preds
    )

    r2 = r2_score(
        targets,
        preds
    )

    return mse, mae, r2


# =====================================================
# Q6 : train_model
# =====================================================

def train_model(
    model,
    train_loader,
    val_loader,
    config
):

    start_time = time.time()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_r2": [],
        "grad_norm": []
    }

    best_val_mse = float("inf")

    patience_counter = 0

    best_state = None

    for epoch in range(config["epochs"]):

        train_loss, grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config["clip_value"]
        )

        val_mse, val_mae, val_r2 = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_mse)

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)
        history["grad_norm"].append(grad_norm)

        if val_mse < best_val_mse:

            best_val_mse = val_mse

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                "best_model.pth"
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= 20:

            print(
                f"Early stopping à l'epoch {epoch+1}"
            )

            break

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch+1:3d} "
                f"| train_loss={train_loss:.4f} "
                f"| val_mse={val_mse:.4f}"
            )

    elapsed_time = time.time() - start_time

    return (
        history,
        best_val_mse,
        elapsed_time
    )


# =====================================================
# TEST Q3
# =====================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    model = DeepFFN()

    print("\n=== Architecture du modèle ===")
    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParamètres entraînables : {total_params:,}"
    )

    x = torch.randn(64, 8)

    out = model(x)

    print("\n=== Test Forward Pass ===")
    print("Entrée :", x.shape)
    print("Sortie :", out.shape)

    print(
        f"Plage : [{out.min().item():.3f}, "
        f"{out.max().item():.3f}]"
    )