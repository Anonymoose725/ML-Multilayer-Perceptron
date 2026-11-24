"""Seeds, device, and shared hyperparameter constants."""
import torch

SEED = 1234
VAL_SPLIT = 0.1 # 10% of dataset used to validate

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

EPOCHS_CNN = 50  # Same as MLP from Q1-Q5
EPOCHS_PRE = 50
BATCH = 128
WORKERS = 2
SWEEP_EPOCHS = 8
RES = 64 # images are upscaled to 64x64
