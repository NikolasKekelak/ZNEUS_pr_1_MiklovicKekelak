
import wandb
from config.config import *

#===| Wand db related |===#
#==============================================================#
run = wandb.init(
    project="ZNEUS_1",
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "max_epochs": MAX_EPOCHS,
        "binary": BINARY_CLASSIFICATION,
        "goal": "Test what is fucked up"
    }
)
#==============================================================#
