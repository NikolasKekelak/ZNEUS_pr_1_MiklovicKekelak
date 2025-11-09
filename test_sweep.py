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
    X = df.drop(columns=FAULT_COLUMNS + ["Class"], errors="ignore")
    y = df["Class"]

    # === Train/val split based on sweep seed ===
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
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

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
        output_dim=1,
        params=params,
        binary=True,
        optimizer=config.optimizer,
        lr=config.learning_rate
    ).to(device)

    # === Train the model ===
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        logger=wandb,
        logging=True
    )

    wandb.finish()


if __name__ == "__main__":
    wandb.login()
    sweep_train()
