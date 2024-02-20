from torch import nn, optim
from torchvision import models
from tqdm import tqdm

# Local imports
from make_dataset import *
from constants import *


def get_model() -> nn.Module:
    # Model setup
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    _, _, number_of_classes = prepare_dataset()

    # Change the last layer of the model
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    model = model.to(ML_TRAINING_DEVICE)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam
):
    # Train the model
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(ML_TRAINING_DEVICE), \
                labels.to(ML_TRAINING_DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: \
                {running_loss/len(train_loader)}"
        )

    return model


def validate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    criterion: nn.CrossEntropyLoss
):
    model.eval()
    val_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(ML_TRAINING_DEVICE), \
                labels.to(ML_TRAINING_DEVICE)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).type(torch.FloatTensor)
            accuracy += correct.mean()

    print(f"Validation loss: {val_loss/len(validation_loader)}")
    print(f"Validation accuracy: {accuracy/len(validation_loader)}")


def main():
    model: nn.Module = get_model()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Load data
    train_loader, _, _ = prepare_dataset()

    # Train the model
    model = train_model(model, train_loader, criterion, optimizer)

    # Validate the model
    validate_model(model, train_loader, criterion)
    torch.save(model, "models/model.pth")


if __name__ == "__main__":
    main()
