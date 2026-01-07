import os
from src import dataset, config
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def visualize_preprocessing():
    """
    Loads a batch of training data and saves a grid of images
    to visualize the effect of preprocessing and augmentation.
    """
    print("Initializing Data Pipeline...")
    train_loader, _, classes = dataset.get_dataloaders()
    
    if train_loader is None:
        print("Dataset not found. Cannot visualize pipeline.")
        return

    # Get one batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    # Denormalize for visualization
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    vis_images = images.permute(0, 2, 3, 1).numpy()
    vis_images = std * vis_images + mean
    vis_images = np.clip(vis_images, 0, 1)
    
    # Plot grid
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title("Data Pipeline Visualization (Augmented Batch)")
    plt.savefig('pipeline_visualization.png')
    print("Visualization saved to pipeline_visualization.png")

if __name__ == "__main__":
    visualize_preprocessing()
