import header_libs
import wandb
from Model import SteelNet
from config.config import *
from config.help_functions import *

def sweep_train():
    #===| Initialize run |===#
    run = wandb.init(project="ZNEUS_1")
    config = wandb.config

    #===| Seed setting |===#
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #====| TU JE CONFIG PRE TOTO SOWYYYY |===#
    #===| Jedno predom dane co testujeme|====#
    binary_mode = False
    BOTH=1
    #=======================================1#

    X_train, X_val, y_train, y_val, train_loader, val_loader, y = (
        split("faults_reduced_normalized.csv", binary_mode, DATA_SPLIT, BOTH ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===| Prepare model parameters |===#
    params = {
        "hidden1": int(config.hidden1),
        "hidden2": int(config.hidden2),
        "dropout1": float(config.dropout1),
        "dropout2": float(config.dropout2),
    }

    # === Initialize model ===
    model = SteelNet(
        input_dim=X_train.shape[1],
        output_dim=1 if binary_mode else 6+BOTH,
        params=params,
        binary=binary_mode,
        optimizer=config.optimizer,
        lr=config.learning_rate,
        targets=y_train if not binary_mode else None,
        patience_counter=100
    ).to(device)

    # === Train the model ===
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        logger=wandb,
        logging=True,
        console_output=False,

    )

    wandb.finish()


if __name__ == "__main__":
    wandb.login()
    sweep_train()
