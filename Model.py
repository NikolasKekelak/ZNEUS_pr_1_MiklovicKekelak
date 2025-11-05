import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import *


class SteelFaults(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = MODEL_STRUCTURE

    def forward(self, x):
        return self.net(x)