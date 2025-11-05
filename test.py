import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random
import numpy as np
from config import *

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


# 1. Load dataset
df = pd.read_csv("../faults_reduced.csv")

# --------------- CLASSIFICATION MODE SWITCH ---------------
fault_columns = ['V28', 'V29', 'V30', 'V31', 'V32', 'V33']

criterion = None

if BINARY_CLASSIFICATION:

    df["target"] = (df[fault_columns].sum(axis=1) > 0).astype(int)
    X = df.drop(columns=fault_columns + ["Class", "target"], errors="ignore")
    y = df["target"]
    criterion = nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()
    if "Class" in df.columns:
        df.drop(["Class"], axis=1, inplace=True)

    df["target"] = df[fault_columns].idxmax(axis=1)
    X = df.drop(columns=fault_columns + ["target"])
    y = df["target"]





scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 3. Dataloaders

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 4. Define neural network
class SteelNet(nn.Module):
    def __init__(self, input_dim, output_dim, binary=False):
        super().__init__()
        layers = []
        if BINARY_CLASSIFICATION:
            layers = MODEL_STRUCTURE_BINARY(input_dim,output_dim)
        else:
            layers = MODEL_STRUCTURE_MULTICLASS(input_dim,output_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# 5. Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None

if BINARY_CLASSIFICATION:
    model = SteelNet(X_train.shape[1], 1).to(device)   # one output neuron
else:
    model = SteelNet(X_train.shape[1], len(encoder.classes_)).to(device)

best_model = None
#optimizer = optim.SGD(model.parameters(), lr=0.001)
#optimizer = optim.RMSprop(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 6. Training loop
best_acc =0
best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(MAX_EPOCHS):
    # --- TRAINING ---
    model.train()
    total_loss = 0
    correct = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    # --- VALIDATION ---
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item() * xb.size(0)
            val_correct += (preds.argmax(1) == yb).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    # --- LOGGING ---
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    run.log({
        "training_loss": train_loss,
        "validation_loss": val_loss,
        "training_acc": train_acc,
        "validation_acc": val_acc,
        "best_validation_acc": best_val_acc
    })

    # --- BEST MODEL TRACKING ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_acc = max(best_acc, train_acc)
        best_model = model

    # --- PRINT PROGRESS ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# 7. Evaluate on test set
best_model.eval()
test_correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)

        if BINARY_CLASSIFICATION:
            preds = preds.squeeze()
            predicted = (preds > 0.5).int()
            test_correct += (predicted == yb.int()).sum().item()
        else:
            test_correct += (preds.argmax(1) == yb).sum().item()

test_acc = test_correct / len(test_loader.dataset)
print(f"✅ Test Accuracy: {test_acc:.3f}")


