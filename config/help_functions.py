
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

            # --- Prediction ---
            if binary:
                # raw logits → sigmoid → threshold
                probs = torch.sigmoid(preds)
                predicted = (probs > 0.5).int().view(-1)
                labels = yb.int().view(-1)
            else:
                predicted = preds.argmax(dim=1).view(-1)
                labels = yb.view(-1)

            # accumulate stats
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # ----- Metrics -----
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)

    avg = 'binary' if binary else 'macro'
    prec = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    rec  = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1   = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    return avg_loss, avg_acc, prec, rec, f1


#==============================================================#
