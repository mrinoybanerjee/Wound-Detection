import torch

ML_TRAINING_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
DATA_DIRECTORY = "data"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 25
