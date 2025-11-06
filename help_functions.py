
from header_libs import *
from config import *
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

def get_optimizer(model, type = OPTIMIZER, lr = LEARNING_RATE):
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
            correct += (predicted == yb).sum().item()

    loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return loss, acc

#==============================================================#
