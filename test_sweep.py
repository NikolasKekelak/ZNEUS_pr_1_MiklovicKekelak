# test_sweep.py
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

import wandb
from Model import SteelNet
from config.config import *


def sweep_train():
    # === Initialize run ===
    run = wandb.init(project="ZNEUS_1")
    config = wandb.config

    # === Seed everything for reproducibility ===
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # === Load dataset ===
    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------
    # ⚙️ Decide between binary or multiclass mode
    # --------------------------------------------------
    binary_mode = BINARY_CLASSIFICATION

    if binary_mode:
        # Binary: collapse all fault columns into one 0/1 column
        df["target"] = (df[FAULT_COLUMNS].sum(axis=1) > 0).astype(int)
        X = df.drop(columns=FAULT_COLUMNS + ["Class"], errors="ignore")
        y = df["target"]
        output_dim = 1
        print("⚙️ Running in BINARY mode.")
    else:
        # Multiclass: pick class index with the active fault
        df["target"] = df[FAULT_COLUMNS].idxmax(axis=1)
        X = df.drop(columns=FAULT_COLUMNS + ["target"], errors="ignore")
        y = df["target"]
        output_dim = len(FAULT_COLUMNS)
        print("⚙️ Running in MULTICLASS mode.")

    # === Train/val split ===
    X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
        X, y, test_size=0.3, random_state=int(config.seed), stratify=y
    )

    # === Encode labels ===
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_ser)
    y_val = encoder.transform(y_val_ser)

    # === Convert to tensors ===
    X_train = torch.tensor(X_train_df.to_numpy(), dtype=torch.float32)
    X_val = torch.tensor(X_val_df.to_numpy(), dtype=torch.float32)

    if binary_mode:
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
    else:
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config.batch_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Prepare model parameters ===
    params = {
        "hidden1": int(config.hidden1),
        "hidden2": int(config.hidden2),
        "dropout1": float(config.dropout1),
        "dropout2": float(config.dropout2),
    }

    # === Initialize model ===
    model = SteelNet(
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        params=params,
        binary=binary_mode,
        optimizer=config.optimizer,
        lr=config.learning_rate,
        targets=y_train if not binary_mode else None  # only used for weighted loss
    ).to(device)

    # === Train the model ===
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        logger=wandb,
        logging=True,
        patience=10  # optional early stopping
    )

    wandb.finish()


if __name__ == "__main__":
    wandb.login()
    sweep_train()
