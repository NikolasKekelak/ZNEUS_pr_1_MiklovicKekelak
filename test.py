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

df = pd.read_csv("faults_reduced_normalized.csv")

#===| CLASSIFICATION MODE SWITCH |===#
#==============================================================#
if BINARY_CLASSIFICATION:
    X = df.drop(columns=FAULT_COLUMNS + ["Class"], errors="ignore")
    y = df["Class"].astype(int)

#===| Multiclass |===#
else:

    if "Class" in df.columns:
        df.drop(["Class"], axis=1, inplace=True)

    df["target"] = df[FAULT_COLUMNS].idxmax(axis=1)
    X = df.drop(columns=FAULT_COLUMNS + ["target"])
    y = df["target"]

    # Encode textual labels into integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

#===| Split dataset |===#
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=1.0 - DATA_SPLIT,
    random_state=int(SEED),
    stratify=y
)

#===| Convert to tensors |===#
X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_val   = torch.tensor(X_val.to_numpy(), dtype=torch.float32)

if BINARY_CLASSIFICATION:
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)
else:
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)

#===| DataLoaders |===#
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
#==============================================================#

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

#===| Training |===#
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

print(f"✅ Best validation accuracy: {best_val_acc:.4f}")
