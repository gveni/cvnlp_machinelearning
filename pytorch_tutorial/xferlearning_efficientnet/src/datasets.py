# Data preparation for model training (with and without pretrained model)
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

ROOT_DIR = "../input/Chessman-image-dataset/Chess"
VALID_SPLIT = 0.1
IMAGE_SIZE = 224  # Input images to be resized while applying transforms
BATCH_SIZE = 16
NUM_WORKERS = 4  # number of parallel processes while preparing data


# transforms needed while training the model
def get_train_transform(pretrained):
    """
    Apply following augmentations while transforming training data
    :param pretrained: Boolean, True or False.
    :return:
    """
    train_tranform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_tranform

def get_valid_transform(pretrained):
    """
    Apply following augmentations while transforming validation data
    :param pretrained: Boolean, True or False.
    :return:
    """
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

def normalize_transform(pretrained):
    """
    Apply normalization transforms based on the pretrained parameter.
    if pretrained = True, that is using EfficientNet pretrained weights, ImageNet normalizations stats are adopted
    else, standard normalization stats are used
    :param pretrained: Boolean, True or False.
    :return: normalization stats
    """
    if pretrained:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        return normalize

# Prepare training and validation datasets using above defined transforms
def get_datasets(pretrained):
    """
    Prepare train and validation datasets
    :param pretrained: Boolean, True or False.
    :return: training and validation datasets along with class names
    """
    transformed_dataset = datasets.ImageFolder(ROOT_DIR, transform=get_train_transform(pretrained))
    transformed_dataset_valid = datasets.ImageFolder(ROOT_DIR, transform=get_valid_transform(pretrained))
    dataset_size = len(transformed_dataset)

    # partition the dataset into training and valid
    valid_size = int(VALID_SPLIT*dataset_size)
    indices = torch.randperm(dataset_size).tolist()
    dataset_train = Subset(transformed_dataset, indices[:-valid_size])
    dataset_valid = Subset(transformed_dataset_valid, indices[-valid_size:])

    return dataset_train, dataset_valid, transformed_dataset.classes

# Prepare dataloaders for train and validation datasets
def get_dataloaders(dataset_train, dataset_valid):
    """
    Prepare  training and validation dataloaders
    :param dataset_train: training dataset
    :param dataset_valid: validation dataset
    :return: training and validation dataloaders
    """
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return dataloader_train, dataloader_valid


