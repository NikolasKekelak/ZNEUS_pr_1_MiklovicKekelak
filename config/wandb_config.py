
import wandb
from config.config import *

#===| Wand db related |===#
#==============================================================#
run = wandb.init(
    entity="nikolaskekelak-fiit-stu",
    project="ZNEUS_1",
    # Track hyperparameters and run metadata.
    config={
        "seed": SEED,
        "goal": "Porovnat rozne typy dropoutu v rammci modelu",
        "testing_batch": "DropiTropiVripiTripi",
        "architecture": "128->64",
        "dataset": "Steel Plates Fault",
        "epochs": MAX_EPOCHS
    },
)
#==============================================================#
