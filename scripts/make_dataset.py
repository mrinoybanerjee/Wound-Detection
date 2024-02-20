from constants import *
from setup import *


def prepare_dataset():
    preprocess_instance = PreProcess(DATA_DIRECTORY)
    dataset = preprocess_instance.get_dataset()
    number_of_classes = len(dataset.classes)
    train_loader, val_loader = preprocess_instance.execute_etl()
    return train_loader, val_loader, number_of_classes
