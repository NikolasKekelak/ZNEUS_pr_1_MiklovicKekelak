import torch
import torch.nn as nn
import torch.optim as optim
import wandb

SEED = 42
#===| Wand db related |===#
#==============================================================#
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="my-awesome-team-name",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)


#==============================================================#

#===| Dataset related |===#
#==============================================================#

INPUT_FILE = "faults.csv"
DATA_SPLIT = 0.70 # DATA_SPILT is how much % is training set
FAULT_COLUMNS = ['V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'Class']
#==============================================================#


#===| Model related |===#
#==============================================================#
BINARY_CLASSIFICATION = False
MAX_EPOCHS = 10_000
MULTCLASS_LOSS = ""
BINARY_LOSS = "bceLoss"

BATCH_SIZE = 64
LEARNING_RATE = 3e-3

def MODEL_STRUCTURE_MULTICLASS(input_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

def MODEL_STRUCTURE_BINARY(input_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
#==============================================================#
