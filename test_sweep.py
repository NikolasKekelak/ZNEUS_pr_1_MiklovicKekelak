import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import random

from config import *

# ----------------------------------------------------------------
# 🔹 Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ----------------------------------------------------------------
# 🔹 Define the neural network
class SteelNet(nn.Module):
    def __init__(self, input_dim, output_dim, binary=False):
        super().__init__()
        layers = []
        if BINARY_CLASSIFICATION:
            layers = MODEL_STRUCTURE_BINARY(input_dim,output_dim)
        else:
            layers = MODEL_STRUCTURE_MULTICLASS(input_dim,output_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
# ----------------------------------------------------------------
# 🔹 Training function (for sweeps)
def train():
    # Initialize W&B
    run = wandb.init(project="ZNEUS_1", config={
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout": 0.2,
        "epochs": 2000,
        "binary_classification": BINARY_CLASSIFICATION
    })

    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------
    # 🔹 Load dataset
    df = pd.read_csv("../faults_reduced.csv")
    fault_columns = ['V28', 'V29', 'V30', 'V31', 'V32', 'V33']

    if config.binary_classification:
        df["target"] = (df[fault_columns].sum(axis=1) > 0).astype(int)
        X = df.drop(columns=fault_columns + ["Class", "target"], errors="ignore")
        y = df["target"]
        criterion = nn.BCELoss()
    else:
        if "Class" in df.columns:
            df.drop(["Class"], axis=1, inplace=True)
        df["target"] = df[fault_columns].idxmax(axis=1)
        X = df.drop(columns=fault_columns + ["target"])
        y = df["target"]
        criterion = nn.CrossEntropyLoss()

    # Normalize + encode
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32 if config.binary_classification else torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32 if config.binary_classification else torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32 if config.binary_classification else torch.long)

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # ----------------------------------------------------------------
    # 🔹 Model + optimizer
    if config.binary_classification:
        model = SteelNet(X_train.shape[1], 1, binary=True, dropout=config.dropout).to(device)
    else:
        model = SteelNet(X_train.shape[1], len(np.unique(y_encoded)), binary=False, dropout=config.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # ----------------------------------------------------------------
    # 🔹 Training loop
    best_val_acc = 0
    best_model = None

    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss, correct = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)

            if config.binary_classification:
                loss = criterion(preds.squeeze(), yb)
                predicted = (preds.squeeze() > 0.5).int()
                correct += (predicted == yb.int()).sum().item()
            else:
                loss = criterion(preds, yb)
                correct += (preds.argmax(1) == yb).sum().item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)

                if config.binary_classification:
                    loss = criterion(preds.squeeze(), yb)
                    val_correct += ((preds.squeeze() > 0.5).int() == yb.int()).sum().item()
                else:
                    loss = criterion(preds, yb)
                    val_correct += (preds.argmax(1) == yb).sum().item()

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # ----------------------------------------------------------------
    # 🔹 Test set evaluation
    model.load_state_dict(best_model)
    model.eval()
    test_correct = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            if config.binary_classification:
                test_correct += ((preds.squeeze() > 0.5).int() == yb.int()).sum().item()
            else:
                test_correct += (preds.argmax(1) == yb).sum().item()

    test_acc = test_correct / len(test_loader.dataset)
    wandb.log({"test_acc": test_acc})
    print(f"✅ Test Accuracy: {test_acc:.3f}")

    run.finish()

# ----------------------------------------------------------------
# 🔹 Define sweep config
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.0005, 0.001, 0.002]},
        'batch_size': {'values': [32, 64, 128]},
        'dropout': {'values': [0.1, 0.2, 0.3]},
        'epochs': {'value': 2000},
        'binary_classification': {'value': BINARY_CLASSIFICATION}
    }
}

# 🔹 Launch sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="ZNEUS_1")
    wandb.agent(sweep_id, function=train, count=5)  # run 5 experiments
