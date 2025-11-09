from config.wandb_config import *
from config.help_functions import *
from Model import *

#===| Seed preparation |===#
#==============================================================#
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#==============================================================#

X_train, X_val , y_train, y_val, train_loader, val_loader, y = (
    split("faults_reduced_normalized.csv",BINARY_CLASSIFICATION, DATA_SPLIT))

#===| Preparations |===#
#==============================================================#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if BINARY_CLASSIFICATION:
    model = SteelNet(
        X_train.shape[1],
        1,
        PARAMS,
        binary=True
    ).to(device)
else:
    model = SteelNet(
        X_train.shape[1],
        len(np.unique(y)),
        PARAMS,
        binary=False,
        targets=y_train
    ).to(device)
#==============================================================#


#===| Training and Logging|===#
#==============================================================#
best_model, best_val_acc = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    logger=run,
    logging=True,
    device=device,
    max_epochs=MAX_EPOCHS,
)
#==============================================================#
