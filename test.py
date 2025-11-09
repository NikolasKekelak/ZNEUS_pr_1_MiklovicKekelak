from config.wandb_config import *
from config.help_functions import *
from Model import *

#===| Seed preparation |===#
#==============================================================#
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#==============================================================#


df = pd.read_csv("faults_reduced_normalized.csv")

#===| CLASSIFICATION MODE SWITCH |===#
criterion = None

if BINARY_CLASSIFICATION:
    X = df.drop(columns=FAULT_COLUMNS+["Class"])
    y = df["Class"]

else:
    criterion = nn.CrossEntropyLoss()
    if "Class" in df.columns:
        df.drop(["Class"], axis=1, inplace=True)

    df["target"] = df[FAULT_COLUMNS].idxmax(axis=1)
    X = df.drop(columns=FAULT_COLUMNS + ["target"])
    y = df["target"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_val   = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
X_test  = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)


# 3. Dataloaders

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

#===| Preparations |===#
#==============================================================#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(y_train)

model = None

if BINARY_CLASSIFICATION:
    model = SteelNet(X_train.shape[1], 1, PARAMS,binary=True,patience=False).to(device)
else:
    model = SteelNet(X_train.shape[1], len(encoder.classes_),PARAMS, binary=False, targets=y_train,patience=False ).to(device)



best_model = None
#==============================================================#


#===| Training |===#
#==============================================================#
best_model, best_val_acc = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    logger = run,
    logging = True,
    device=device,
    max_epochs=MAX_EPOCHS,
)
#==============================================================#

# === Test Evaluation === #
# #==============================================================#
#model.load_state_dict(best_model)
#test_loss, test_acc = evaluate(model, test_loader, criterion, device, BINARY_CLASSIFICATION)
#print(f"✅ Test Accuracy: {test_acc:.3f}")
#==============================================================#