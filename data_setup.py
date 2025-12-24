# Goal
# Download the data | get the dataloaders and transformations |put this all into a function | should return training, and validation dataloaders and class names and number of classes
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 4

def create_data_setup (BATCH_SIZE:int, num_workers:int,):


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", download=True, transform=transform)
    
    print("Finished downloading training data")
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    testset = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=transform)
    print("Finished downloading test data")
    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    CLASS_NAMES = testset.classes
    

    return train_loader, test_loader, CLASS_NAMES

  