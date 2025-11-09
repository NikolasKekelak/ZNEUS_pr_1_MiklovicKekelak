from config.config import *
from config.help_functions import get_optimizer, evaluate


class SteelNet(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 params,
                 binary=False,
                 optimizer: str = OPTIMIZER,
                 lr=LEARNING_RATE,
                 targets=None):

        super().__init__()

        self.binary = binary

        # === Build network ===
        if self.binary:
            self.net = MODEL_STRUCTURE_BINARY(
                        input_dim,
                        output_dim,
                        h1=params["hidden1"],
                        h2=params["hidden2"],
                        d1=params["dropout1"],
                        d2=params["dropout2"],
                    )
            self.criterion = nn.BCELoss()

        else:
            self.net = MODEL_STRUCTURE_MULTICLASS(
                        input_dim,
                        output_dim,
                        h1=params["hidden1"],
                        h2=params["hidden2"],
                        d1=params["dropout1"],
                        d2=params["dropout2"],
                    )

            # === Auto-compute class weights if targets are provided ===
            if targets is not None:
                unique, counts = torch.unique(targets, return_counts=True)
                counts = counts.float()

                # inverse frequency
                weights = 1.0 / counts
                weights = weights / weights.sum()

                # create weight vector sorted by class index
                # so that weight[class_id] is correct
                weight_vector = torch.zeros(output_dim)
                for cls, w in zip(unique, weights):
                    weight_vector[int(cls.item())] = w

                self.criterion = nn.CrossEntropyLoss(weight=weight_vector)

            else:
                self.criterion = nn.CrossEntropyLoss()

        # === Optimizer ===
        self.optimizer = get_optimizer(self, optimizer, lr)


    def forward(self, x):
        return self.net(x)

    def train_step(self, xb, yb):
        # ---- reshape target for binary classification ----
        if self.binary:
            yb = yb.float().unsqueeze(1)  # [batch] -> [batch,1]

        preds = self(xb)
        loss = self.criterion(preds, yb)

        loss.backward()
        self.optimizer.step()

        if self.binary:
            predicted = (preds > 0.5).int()
            correct = (predicted == yb.int()).sum().item()
        else:
            predicted = preds.argmax(1)
            correct = (predicted == yb).sum().item()

        return loss.item(), correct


    def train_epoch(self, loader, device):
        self.train()
        total_loss, total_correct = 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            self.optimizer.zero_grad()
            loss, correct = self.train_step(xb, yb)

            total_loss += loss * xb.size(0)
            total_correct += correct

        # epoch metrics
        avg_loss = total_loss / len(loader.dataset)
        avg_acc = total_correct / len(loader.dataset)
        return avg_loss, avg_acc



    def validate_epoch(self, loader, device):
        return evaluate(self, loader, self.criterion, device, self.binary)

    # ===| Main Training loop |===#
    def fit(self,
            train_loader,
            val_loader,
            device,
            max_epochs,
            logger,
            logging: bool,
            console_output : bool = True,
            ):

        best_model_state = None
        best_val_acc = 0.0

        for epoch in range(max_epochs):

            train_loss, train_acc = self.train_epoch(train_loader, device)

            val_loss, val_acc, prec, rec, f1 = self.validate_epoch(val_loader, device)

            #===| 🪵 |===#
            if(logging):
                logger.log({
                    "best_validation_acc" : best_val_acc,
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
                    "training_acc": train_acc,
                    "validation_acc": val_acc,
                    "best_validation_acc": best_val_acc,
                    "precission": prec,
                    "recall": rec,
                    "f1": f1,
                })

            # ===| Save best |===#
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.state_dict().copy()

        print("Neural Network stopped neural networking")
        return best_model_state, best_val_acc

