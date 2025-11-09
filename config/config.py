from header_libs import *

SEED = 69

#===| Dataset related |===#
#==============================================================#
INPUT_FILE = "faults_reduced_normalized.csv"
DATA_SPLIT = 0.70 # DATA_SPILT is how much % is training set
FAULT_COLUMNS = ['V28', 'V29', 'V30', 'V31', 'V32', 'V33']

#Normalisation/scaler
SCALER_TYPE = "minmax"  # options: 'standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power'
MIN_MAX_INTERVAL = (0,1) # ma efekt len pre moznost minmax
#==============================================================#


#===| Model related |===#
#==============================================================#
BINARY_CLASSIFICATION = True
MAX_EPOCHS = 5_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OPTIMIZER = "adam" # options: 'adam' , 'sgd' , 'rms'

PARAMS = {
    "hidden1" : 128,
    "hidden2" : 64,
    "dropout1" : 0.2,
    "dropout2" : 0.1,
}

def MODEL_STRUCTURE_BINARY(input_dim, output_dim, h1, h2, d1, d2):
    return nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.BatchNorm1d(h1),
        nn.Dropout(d1),
        nn.ReLU(),

        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.Dropout(d2),
        nn.ReLU(),

        nn.Linear(h2, output_dim),
        nn.Sigmoid(),
    )


def MODEL_STRUCTURE_MULTICLASS(input_dim, output_dim, h1, h2, d1, d2):
    return nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.BatchNorm1d(h1),
        nn.Dropout(d1),
        nn.ReLU(),

        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.Dropout(d2),
        nn.ReLU(),

        nn.Linear(h2, output_dim),
        nn.Sigmoid(),
    )


#==============================================================#


