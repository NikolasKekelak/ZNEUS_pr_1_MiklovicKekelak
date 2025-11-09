from header_libs import *
import wandb
from config.config import *
from Model import *

# === Load dataset ===
df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=FAULT_COLUMNS + ["Class"])
y = df["Class"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_val   = torch.tensor(X_val.to_numpy(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)


def sweep_train():
    wandb.init(project="ZNEUS_1")
    config = wandb.config

    # Set seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Re-create loaders with batch size from sweep
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SteelNet(
        X_train.shape[1],
        1,
        binary=True,
        optimizer=config.optimizer,
        lr=config.learning_rate,
        params=config  # pass whole sweep config
    ).to(device)

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        logger=wandb,
        logging=True
    )



if __name__ == "__main__":
    wandb.login()

    sweep_id = wandb.sweep("sweep_config.yaml", project="ZNEUS_1")
    wandb.agent(sweep_id, function=sweep_train)
