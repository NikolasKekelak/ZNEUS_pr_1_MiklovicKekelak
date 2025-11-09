
from header_libs import *
from config.config import *


#===| Help functions |===#
#==============================================================#

def get_scaler(type, interval = (0,1)):
    if type == "standard":
        return StandardScaler()
    elif type == "minmax":
        return MinMaxScaler(feature_range=interval)
    elif type == "robust":
        return RobustScaler()
    elif type == "maxabs":
        return MaxAbsScaler()
    elif type == "quantile":
        return QuantileTransformer(output_distribution='uniform')
    elif type == "power":
        return PowerTransformer()
    else:
        raise ValueError(f"Unknown type: {type}")

def get_optimizer(model, type :str, lr):
    if type == "adam":
        return optim.Adam(model.parameters(), lr)
    if type == "sgd":
        return optim.SGD(model.parameters(), lr)
    if type == "rms":
        return optim.RMSprop(model.parameters(), lr)
    else:
        raise ValueError(f"Unknown type: {type}")

#==============================================================#


#===| Function for evaluation of current configuration (that rhymes lol) |===#
def evaluate(model, loader, criterion, device, binary=False):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # --- Fix target shape for binary classification ---
            if binary:
                yb = yb.float().view(-1, 1)    # shape [batch] → [batch,1]

            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)

            if binary:
                predicted = (preds > 0.5).int().view(-1)
                labels = yb.int().view(-1)
            else:
                predicted = preds.argmax(dim=1).view(-1)
                labels = yb.view(-1)

            # accumulate stats
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    #===| Nieco som tu chcel napisat ale sry uz nepamatam |===#
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)

    avg = 'binary' if binary else 'macro'
    prec = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    rec  = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1   = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    return avg_loss, avg_acc, prec, rec, f1

#==============================================================#
def split(path: str, binary: bool, data_split: float, both=0):
    df = pd.read_csv(path)

    # ===| CLASSIFICATION MODE SWITCH |===#
    if binary:
        # ===| Binary classification |===#
        X = df.drop(columns=FAULT_COLUMNS + ["Class"], errors="ignore")

        # if “Class” exists already as {0,1}, use it directly
        if "Class" in df.columns:
            y = df["Class"].astype(int)
        else:
            # otherwise create it by combining all fault columns
            y = (df[FAULT_COLUMNS].sum(axis=1) > 0).astype(int)

    else:
        # ===| Multiclass classification |===#
        if both ==0 and "Class" in df.columns:
            df.drop(["Class"], axis=1, inplace=True)

        # pick the column name (fault type) with the highest value per row
        df["target"] = df[FAULT_COLUMNS].idxmax(axis=1)
        X = df.drop(columns=FAULT_COLUMNS + ["target"])
        y = df["target"]

        # Encode textual fault labels into integers
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # ===| Train / Val split |===#
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=1.0 - data_split,
        random_state=int(SEED),
        stratify=y
    )

    # ===| Convert to torch tensors |===#
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val   = torch.tensor(X_val.to_numpy(), dtype=torch.float32)

    if binary:
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
        y_val   = torch.tensor(np.array(y_val), dtype=torch.float32)
    else:
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        y_val   = torch.tensor(np.array(y_val), dtype=torch.long)

    # ===| Create DataLoaders |===#
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return X_train, X_val, y_train, y_val, train_loader, val_loader, y

#==============================================================#
def plot_confusion_matrix(model, val_loader, device, binary=True):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)

            if binary:
                preds = (preds.squeeze() > 0.5).int()
            else:
                preds = preds.argmax(dim=1)

            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100

    # plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format=".1f")
    ax.set_title("Confusion Matrix (%)")
    plt.tight_layout()
    plt.show()

    return cm

#==============================================================#
