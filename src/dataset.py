import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from . import config

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders():
    # Only create dataloaders if directories exist, otherwise return None/Empty for safety
    if not os.path.exists(config.TRAIN_DIR) or not os.path.exists(config.TEST_DIR):
        print(f"Warning: Dataset directories not found at {config.DATASET_DIR}")
        return None, None, None

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=get_transforms('train'))
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=get_transforms('test'))

    # Verify classes
    if len(train_dataset.classes) != config.NUM_CLASSES:
        print(f"Warning: Expected {config.NUM_CLASSES} classes, found {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, train_dataset.classes
