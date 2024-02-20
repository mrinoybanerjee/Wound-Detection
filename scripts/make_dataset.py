from constants import *  # Importing all constants from the constants module
from setup import *  # Importing all functions/classes from the setup module


def prepare_dataset():
    """
    This function prepares the dataset for training and validation.
    It uses the PreProcess class from the setup module to preprocess the data.
    It returns the training and validation data loaders, and the number of classes in the dataset.

    Returns:
        train_loader (DataLoader): The DataLoader instance for the training data.
        val_loader (DataLoader): The DataLoader instance for the validation data.
        number_of_classes (int): The number of classes in the dataset.
    """

    # Create an instance of the PreProcess class with the data directory
    preprocess_instance = PreProcess(DATA_DIRECTORY)
    # Get the dataset using the get_dataset method of the PreProcess instance
    dataset = preprocess_instance.get_dataset()
    # Calculate the number of classes in the dataset
    number_of_classes = len(dataset.classes)
    # Execute the ETL (Extract, Transform, Load) process to get the training and validation data loaders
    train_loader, val_loader = preprocess_instance.execute_etl()
    # Return the training and validation data loaders, and the number of classes
    return train_loader, val_loader, number_of_classes
