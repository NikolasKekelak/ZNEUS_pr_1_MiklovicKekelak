
import wandb
from config import *

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


#===| Sweep Config |===#
#==============================================================#
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'validation_acc',   # what metric W&B should maximize
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.0005, 0.001, 0.002, 0.005]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3]
        }
    }
}
#==============================================================#
