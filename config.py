import torch
import torch.nn as nn
import torch.optim as optim

#Dataset related
DATA_SPLIT = 0.70


#Model related
INPUT_DIM = 27
OUTPUT_DIM = 8
MODEL_STRUCTURE = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),


            nn.Linear(16, OUTPUT_DIM)
        )
