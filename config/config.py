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

def MODEL_STRUCTURE_MULTICLASS(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(64, output_dim),
        nn.Sigmoid()
    )

def MODEL_STRUCTURE_BINARY(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.Dropout(0.1),
        nn.ReLU(),

        nn.Linear(64, output_dim),
        nn.Sigmoid()
    )

#==============================================================#


