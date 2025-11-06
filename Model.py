from sklearn.preprocessing import StandardScaler

from config import *


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

