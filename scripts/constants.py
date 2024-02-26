import torch

# Define the ML training device based on GPU availability
ML_TRAINING_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Directory containing raw data
DATA_DIRECTORY = "data/raw/WoundDataset/train"

# Learning rate for training the model
LEARNING_RATE = 0.001

# Batch size for DataLoader
BATCH_SIZE = 32

# Number of epochs for training
NUM_EPOCHS = 15
