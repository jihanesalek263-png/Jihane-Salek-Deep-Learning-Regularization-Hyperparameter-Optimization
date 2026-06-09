from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# =====================================================
# Q1 : Chargement du dataset
# =====================================================

housing = fetch_california_housing()

X = housing.data
y = housing.target

df = pd.DataFrame(
    X,
    columns=housing.feature_names
)

df["MedHouseVal"] = y

print("Nombre d'exemples :", df.shape[0])
print("Nombre de features :", len(housing.feature_names))

print("\nFeatures :")
print(housing.feature_names)

print("\nPremières lignes :")
print(df.head())

print("\nStatistiques descriptives :")
print(df.describe())

# =====================================================
# Q2 : Histogramme et Boxplot
# =====================================================

plt.figure(figsize=(8,5))
plt.hist(df["MedHouseVal"], bins=50)
plt.title("Distribution de MedHouseVal")
plt.show()

plt.figure(figsize=(8,3))
plt.boxplot(df["MedHouseVal"], vert=False)
plt.title("Boxplot de MedHouseVal")
plt.show()

# =====================================================
# Q3 : Heatmap de corrélation
# =====================================================

corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matrice de corrélation")
plt.show()

# =====================================================
# Q4 : Split 70 / 15 / 15
# =====================================================

df["target_bin"] = pd.cut(
    df["MedHouseVal"],
    bins=5,
    labels=False
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["target_bin"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["target_bin"]
)

print("\nTrain :", len(train_df))
print("Validation :", len(val_df))
print("Test :", len(test_df))

# =====================================================
# Q5 : StandardScaler
# =====================================================

X_train = train_df.drop(
    ["MedHouseVal", "target_bin"],
    axis=1
)

y_train = train_df["MedHouseVal"]

X_val = val_df.drop(
    ["MedHouseVal", "target_bin"],
    axis=1
)

y_val = val_df["MedHouseVal"]

X_test = test_df.drop(
    ["MedHouseVal", "target_bin"],
    axis=1
)

y_test = test_df["MedHouseVal"]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =====================================================
# Q6 : TensorDataset et DataLoader
# =====================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values.reshape(-1,1), dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values.reshape(-1,1), dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False
)

xb, yb = next(iter(train_loader))

print("\n=== Vérification DataLoader ===")
print("X batch :", xb.shape)
print("y batch :", yb.shape)
print("X mean :", xb.mean().item())
print("X std :", xb.std().item())