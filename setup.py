from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Local imports
from constants import BATCH_SIZE


class PreProcess:
    dataset_directory_path: str = None
    dataset: datasets.ImageFolder = None

    def __init__(self, dataset_directory_path: str):
        self.dataset_directory_path = dataset_directory_path

    def __load_image_directory__(self):
        self.dataset = datasets.ImageFolder(self.dataset_directory_path)

    def __get_image_transforms__(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __get_data_loader__(self, dataset_subset, batch_size: int = BATCH_SIZE) -> DataLoader:
        return DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=True
        )

    def get_dataset(self) -> datasets.ImageFolder:
        return self.dataset

    def execute_etl(self, train_size: float = 0.8) -> tuple[DataLoader, DataLoader]:
        self.__load_image_directory__()
        dataset_size = len(self.dataset)
        validation_size = int((1-train_size) * dataset_size)

        train_dataset, validation_dataset = random_split(
            self.dataset,
            [train_size, validation_size]
        )

        train_loader = self.__get_data_loader__(train_dataset)
        validation_loader = self.__get_data_loader__(validation_dataset)

        return train_loader, validation_loader
