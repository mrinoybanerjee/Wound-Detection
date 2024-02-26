from torch import nn, optim
from torchvision import models
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

# Local imports
from make_dataset import *
from constants import *


def get_model() -> nn.Module:
    """
    This function creates a pretrained ResNet50 model, freezes its parameters,
    changes the last layer to match the number of classes in the dataset, and
    moves the model to the device specified in the constants.

    Returns:
        model (nn.Module): The prepared model.
    """
    # Model setup
    # Load a pretrained ResNet50 model
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():  # Freeze the parameters of the model
    #     param.requires_grad = False

    # Get the number of classes in the dataset
    _, _, number_of_classes = prepare_dataset()
    model = EfficientNet.from_pretrained(
        'efficientnet-b0',
        num_classes=number_of_classes
    )
    # Move the model to the specified device
    model = model.to(ML_TRAINING_DEVICE)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam
):
    """
    This function trains the model for a number of epochs specified in the constants.
    It prints the loss for each epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Adam): The optimizer.

    Returns:
        model (nn.Module): The trained model.
    """
    # Train the model
    for epoch in tqdm(range(NUM_EPOCHS)):  # For each epoch
        model.train()  # Set the model to training mode
        running_loss = 0  # Initialize the running loss
        for inputs, labels in train_loader:  # For each batch in the training data
            inputs, labels = inputs.to(ML_TRAINING_DEVICE), \
                labels.to(
                    ML_TRAINING_DEVICE)  # Move the inputs and labels to the specified device

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()  # Add the loss for this batch to the running loss

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: \
                {running_loss/len(train_loader)}"  # Print the average loss for this epoch
        )

    return model  # Return the trained model


def validate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    """
    This function validates the model on the validation data.
    It prints the validation loss and accuracy.

    Args:
        model (nn.Module): The model to validate.
        validation_loader (DataLoader): The DataLoader for the validation data.
        criterion (nn.CrossEntropyLoss): The loss function.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize the validation loss
    accuracy = 0  # Initialize the accuracy

    with torch.no_grad():  # No need to calculate gradients for validation
        for inputs, labels in validation_loader:  # For each batch in the validation data
            inputs, labels = inputs.to(ML_TRAINING_DEVICE), \
                labels.to(
                    ML_TRAINING_DEVICE)  # Move the inputs and labels to the specified device
            outputs = model(inputs)  # Forward pass
            # Calculate the loss for this batch
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()  # Add the loss for this batch to the validation loss

            _, predicted = torch.max(outputs, 1)  # Get the predicted classes
            # Calculate the number of correct predictions
            correct = (predicted == labels).type(torch.FloatTensor)
            accuracy += correct.mean()  # Add the accuracy for this batch to the total accuracy

    # Print the average validation loss
    print(f"Validation loss: {val_loss/len(validation_loader)}")
    # Print the average accuracy
    print(f"Validation accuracy: {accuracy/len(validation_loader)}")


def main():
    """
    This function is the main entry point of the script.
    It prepares the model, the loss function, and the optimizer, loads the data,
    trains the model, validates it, and saves it.
    """
    model: nn.Module = get_model()  # Get the model

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )  # Define the optimizer

    # Load data
    train_loader, _, _ = prepare_dataset()  # Load the training data

    # Train the model
    model = train_model(
        model,
        train_loader,
        criterion,
        optimizer
    )  # Train the model

    # Validate the model
    validate_model(model, train_loader, criterion)  # Validate the model
    torch.save(model, "models/model.pth")  # Save the model


if __name__ == "__main__":
    main()  # Call the main function if the script is run directly
