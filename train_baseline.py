from ex1_dataset import train_loader, val_loader

from ex2_model import (
    DeepFFN,
    train_model
)

config_baseline = {
    'hidden_dims': [128, 64, 32],
    'activation': 'relu',
    'use_bn': True,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'epochs': 200,
    'early_stopping_patience': 25
}

model = DeepFFN(
    hidden_dims=config_baseline['hidden_dims'],
    activation=config_baseline['activation'],
    use_bn=config_baseline['use_bn'],
    dropout_rate=config_baseline['dropout_rate']
)

history, best_val_mse, training_time = train_model(
    model,
    train_loader,
    val_loader,
    config_baseline
)

print("Best Val MSE:", best_val_mse)
print("Training Time:", training_time)