# Goal
# Download the data | get the dataloaders and transformations |put this all into a function | should return training, and validation dataloaders and class names and number of classes
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def create_data_setup (BATCH_SIZE:int, NUM_WORKERS:int,):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", download=True, transform=train_transform)
    
    print("Finished downloading training data")
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    testset = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=True, transform=test_transform)
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

  