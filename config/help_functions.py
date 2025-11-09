
from header_libs import *
from config.config import *
#===| Help functions |===#
#==============================================================#

def get_scaler(type=SCALER_TYPE, interval = MIN_MAX_INTERVAL):
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
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)

            if binary:
                predicted = (preds.squeeze() > 0.5).int()
            else:
                predicted = preds.argmax(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            correct += (predicted == yb).sum().item()

    avg = 'binary' if binary else 'macro'
    loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    prec = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    return loss, acc, prec, rec, f1

#==============================================================#
