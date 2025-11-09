from sklearn.preprocessing import StandardScaler

from config.config import *
from config.help_functions import get_optimizer


class SteelNet(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 binary=False,
                 optimizer:str = OPTIMIZER,
                 lr = LEARNING_RATE
                 ):
        super().__init__()
        if BINARY_CLASSIFICATION:
            self.net  = MODEL_STRUCTURE_BINARY(input_dim,output_dim)
        else:
            self.net = MODEL_STRUCTURE_MULTICLASS(input_dim,output_dim)


        self.optimizer = get_optimizer(self, optimizer, lr)

    def forward(self, x):
        return self.net(x)

