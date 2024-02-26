import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold

# Local imports
from constants import BATCH_SIZE


class PreProcess:
    dataset_directory_path: str = None
    dataset: datasets.ImageFolder = None

    def __init__(self, dataset_directory_path: str):
        self.dataset_directory_path = dataset_directory_path
        self.__load_image_directory__()

    def __load_image_directory__(self):
        self.dataset = datasets.ImageFolder(
            self.dataset_directory_path,
            transform=self.__get_image_transforms__()
        )

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

        # train_dataset, validation_dataset = random_split(
        #     self.dataset,
        #     [train_size, validation_size]
        # )

        # train_loader = self.__get_data_loader__(train_dataset)
        # validation_loader = self.__get_data_loader__(validation_dataset)

        # Initialize StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Loop through folds for cross-validation
        for fold, (train_index, val_index) in enumerate(
                skf.split(self.dataset.imgs, self.dataset.targets)
        ):
            print(f'Fold [{fold + 1}/5]')

            # Create train and validation datasets based on fold indices
            train_dataset = torch.utils.data.Subset(self.dataset, train_index)
            val_dataset = torch.utils.data.Subset(self.dataset, val_index)

            # Create DataLoader for train and validation datasets
            train_loader = self.__get_data_loader__(
                train_dataset
            )
            validation_loader = self.__get_data_loader__(
                val_dataset
            )

        return train_loader, validation_loader
